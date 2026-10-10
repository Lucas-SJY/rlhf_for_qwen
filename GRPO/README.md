# GRPO for labelled chain-of-thought: design notes

This directory holds the GRPO implementation: code, image and Kubernetes manifests. The
entry point is **`../run.sh`**; how to run it is in the [root README](../README.md). The exact per-step computation, losses, reward and
hyperparameters are specified in **[ALGORITHM.md](ALGORITHM.md)**. This file explains
the design decisions and what has been verified.

Only the shared training path has run on a GPU, with Qwen3-0.6B (see
[Verification](#verification)); the 8B LoRA run has not. **Run its smoke test first.**

---

## 1. Why GRPO, and which rLLM trainer

GRPO drops PPO's critic. For each question it samples a group of 8 answers and uses
their mean reward as the baseline. Compared with the earlier PPO version, this means:

- no value model;
- no critic warm-up;
- ~20 GB less host memory;
- no dependency on rLLM's older workflow trainer, whose critic path is broken against
  verl 0.8.0.

The cost is 8 generations per question instead of 1.

The run uses **rLLM's unified trainer** (`rllm.trainer.AgentTrainer` with
`backend="verl"`), rLLM's current, recommended training path at `main` (`3b40c37`). It is
designed for critic-free estimators such as GRPO:

- it groups a question's answers by the task `id` and computes the GRPO advantages
  itself;
- it runs the actor update through verl 0.8.0.

Settings that exist in both frameworks are written in rLLM's `rllm.*` namespace and
mirrored into verl by rLLM's `sync_config`. verl-only settings (model, LoRA, FSDP, vLLM)
use verl's paths.

## 2. The reward

A rule-based score of two things: whether the chain of thought uses only the eight
labels (weight 0.2), and whether the final answer is right (weight 0.8). No reference
trace is used.

```
not trained on                                                       if the answer hits 8,192 tokens
R = 0                                                                if the thought never closes
R = 0.2·label_valid + 0.8·answer_correct                             otherwise
```

An answer cut off at the length limit is dropped from the training batch
(`MASK_TRUNCATED=true`, rLLM's compact filtering) rather than scored 0: it gets no
gradient and does not enter its group's mean and std. Validation still counts it as
wrong.

The tag format is a rule, not a reward term: an answer whose thought has a tag outside
the eight labels is thrown away and sampled again (up to `MAX_LABEL_RETRIES` = 3 times);
only the kept sample is trained on, and one still invalid after the last retry gets
`label_valid` = 0.

- `tag_rate` (logged, not rewarded): the share of paragraphs that open with one of the
  eight tags.
- `label_valid`: 1 if the thought has at least one tag and every tag is one of the eight
  labels, else 0. Which labels are used, and how often, does not matter.
- `label_mix` (logged, not rewarded): 1 minus the total-variation distance between the
  label shares in the thought and their average share in an annotated bespoke-v2 trace
  (`LABEL_PRIOR`, from `src/compute_label_prior.py`).
- `answer_correct`: whether the final answer after `</think>` equals the reference answer.
  Only the final answer is judged, not the reasoning; equivalence comes from Hugging Face
  `math_verify` (0.5 = 1/2), with the string matcher of `../../train/evaluate` as a second
  route and for multiple choice.

Full definitions and worked examples are in [ALGORITHM.md §4](ALGORITHM.md#4-the-reward-r).
Scoring each SFT target as if it were a model answer gives 1.0 for all 5,042.

GRPO fits this reward well. When all 8 answers to a question are equally right or wrong,
only the label terms differ, and group normalisation turns those small differences into
full-size advantages. The same scaling also amplifies noise; `NORM_ADV_BY_STD=false`
switches it off.

## 3. Fitting it on one 48 GB card (L40)

- **LoRA on the actor, added by GRPO.** rank 64, alpha 32, all linear layers, lr 1e-5.
  - The SFT checkpoint is full-parameter and has no adapter. GRPO adds a fresh adapter
    (B = 0, so step 0 equals the SFT model) and trains only that.
  - The SFT weights stay frozen, so the reference policy for the KL term is the same model
    with the adapter switched off; no second 8B model is needed.
  - The frozen weights are kept in bf16 (`fsdp_config.model_dtype`), not verl's default
    fp32. The SFT checkpoint is stored in bf16 and FSDP computes in bf16 anyway, so the
    forward pass is identical, but the actor takes ~16 GB instead of ~33 GB. Only the
    adapter and its optimizer state are fp32.
  - `../run.sh export` merges the adapter back into the full weights.
- **Rollout and training take turns on the GPU.**
  - vLLM uses 70 % of the card while it generates 64 answers, and sleeps during training.
  - The actor's weights and optimizer state are offloaded to host memory while vLLM runs.
- **Dynamic batching by tokens** (12,288 tokens per micro-batch). Memory then scales with
  tokens, not with the number of sequences.

The pod requests 8 CPUs, 160 GiB RAM (including a 24 GiB `/dev/shm` for Ray's 16 GiB
object store) and one GPU chosen by `GPU_TYPE`: L40 / L40S by default, RTX A6000, or
A100-80GB. `GPU_TYPE=a100` runs at priority `opportunistic` (no quota needed,
preemptible) unless `GPU_PRIORITY_CLASS=` is set empty; the namespace has had an A100
quota of 4 since 2026-10-05. The 48 GB cards need no quota. 48 GB is tight for
8,192-token responses; if the full run runs out of memory, lower `MAX_RESPONSE_LENGTH` or
use `GPU_TYPE=a100`.

**Full-parameter training** (`LORA_RANK=0`, `ACTOR_LR=1e-6`) does not fit one card: fp32
weights, gradients and Adam state of the 8B model are ~128 GB. It runs either on
`N_GPUS=4` A100-80GB cards of one node (FSDP1, the state moves to host RAM only while vLLM
generates, 16 CPUs / 256 GiB), or on 1–2 cards with `FSDP_CPU_OFFLOAD=true` (FSDP2
`CPUOffloadPolicy`: the state stays in pinned host RAM, each layer is copied to the GPU
while it is computed, and Adam steps on the CPU; one card: 14 CPUs / 288 GiB, ~240 GB of
host RAM at peak). Both with a separate frozen reference model (CPU-offloaded by verl)
and one vLLM replica per card.

## 4. What is custom, and why

| piece | reason |
|---|---|
| `labelcot/workflow.py` renders the prompt with the checkpoint's own chat template | rLLM's `QwenChatTemplateParser` prepends a default "You are Qwen, created by Alibaba Cloud…" system turn the SFT model never saw. Generation still uses rLLM's token-in/token-out path, so training uses the exact sampled tokens and logprobs. |
| `labelcot/patches.py` saves a final checkpoint | rLLM's verl backend saves only at multiples of `save_freq` and does nothing at the end of training, so up to 19 steps would be lost. It is applied inside the Ray actor that builds the trainer: `workflow.py` calls it on import, and the actor imports that module when it receives the workflow class. |
| data as plain task dicts (`GRPO/data/*.jsonl`) | the unified trainer takes rLLM `Dataset` objects directly: no verl parquet, no dataset registry |
| `docker/patch_verl.py` edits one line of verl at image build time | Under FSDP2 CPU offload (`FSDP_CPU_OFFLOAD=true`), verl's `FSDPEngine.save_checkpoint` moves the module to the GPU, which leaves it half on each device, and `state_dict()` fails ("Attempted to set the storage of a tensor on device cpu to a storage on different device cuda:0"); the first full-parameter run crashed this way at its step-10 checkpoint. The patch skips that move under CPU offload, as verl already does when syncing weights to vLLM (#5995). It runs in verl's GPU worker processes, which never import `labelcot`, so it cannot be a runtime patch; the build fails if the patched code changes. |

rLLM is not edited; verl is, in that one place.

## 5. Data

`src/prepare_data.py` (run through `../run.sh data`) reads `../train/bespoke-v2` and writes
one task per line: `id`, `data_source`, `question`, `answer`, `ref_labels`. The reward no
longer reads `ref_labels`; the field is left in place.

- The prompt is the bare question, exactly as in SFT.
- The split reuses the SFT held-out ids: 5,042 train and 102 validation. The validation
  questions were never trained on in either stage.

`src/prepare_grpo_try.py` builds `data/grpo_try/` from the 9 questions in `../grpo_try/`
(6 train, 3 validation) with only what training needs: `id`, `data_source`, `prompt` (the
chat-format input, one user turn holding the question), `question` and `answer`. No
reference trace or labels are written. Answers come from the upstream harbor solutions.
The workflow uses a task's `prompt` when it has one and otherwise the bare question.
Usage is in the [root README](../README.md#training-on-grpo_try).

`src/prepare_rest_grpo.py` builds `data/rest_grpo/` in the same format from the 11,566
Bespoke-Stratos-17k questions outside the SFT set (`../train/data_labeled_2`), read from
the harbor dataset `bespoke-stratos-rest`, with a seeded 90/10 split. Questions without a
checkable answer (the coding questions and prose or proof answers), which the reward
could score on the label term alone, are moved to `../no_answer/` in the same split, so
`data/rest_grpo/` keeps 4,776 train / 526 validation questions with one checkable answer
(solutions that box several distinct values are moved too); 10 more training questions
are held out in `../manual_inspection/` and never trained on.
Usage is in the [root README](../README.md#training-on-rest_grpo).

## 6. Cost

The time per step is set by the longest of the 64 answers (up to 8,192 tokens), plus
three passes of the 8B actor over all answer tokens: old log-probs, reference log-probs,
and the update. My estimate is **~10–12 minutes per step**, so the default cap of 200 steps
(1,600 questions) is roughly 1.5 days. A full epoch at 8 questions per step would be
630 steps. Treat these numbers as a guess until the smoke test reports `timing_s/*`.

One actor checkpoint is ~33 GB, because FSDP keeps the LoRA-wrapped 8B in fp32. Only the
latest is kept, on the 150 Gi PVC `qwen-grpo-data`. A full-parameter checkpoint would be
~100 GB with the Adam state; verl keeps the previous checkpoint until the next one is
written, so full-parameter runs save only the weights and the training position (~33 GB)
unless `CKPT_SAVE_OPTIMIZER=true`. A resumed run then restarts Adam's moments.

## Verification

Done locally, without a GPU:

- **Reward and data.**
  - `../run.sh test`: 44 unit tests (reward incl. the 0.2 / 0.8 weights, the invalid-label
    rule, cut-off answers left out of the logged reward and the math_verify answer check;
    the workflow's regenerate loop, prompt handling and the overlong termination reason
    the trainer filters on; the grpo_try and rest_grpo task builders and their format
    checks). The 4
    math_verify tests need Python >= 3.10 and the 5 workflow tests need rLLM; both are
    skipped without them.
  - `../run.sh data`: 5,042 / 102 tasks, split identical to SFT.
- **Configuration.** rLLM's `unified` config (verl 0.8.0 `ppo_trainer` underneath) was
  composed with all 58 overrides from `entrypoint.sh`, for both the normal and the smoke
  run. The same config then passed rLLM's `sync_config` and verl's own `validate_config`.
  Resolved values checked: GRPO estimator, group size, KL-in-loss with k3, clip 0.2/0.28,
  token-mean, LoRA 64, hybrid engine.
- **Imports.** verl 0.8.0 and rLLM `3b40c37` were installed in a CPU virtualenv, and
  `train_grpo`, the unified `AgentTrainer`, the workflow and the patched backend import
  cleanly.
- **End-to-end GRPO step, with a fake vLLM server** that replays SFT answers (2 questions × 4
  samples: perfect, cut off, wrong answer, perfect). This ran rLLM's real `VerlEngine`,
  `UnifiedWorkflowEngine`, trajectory grouping, GRPO advantage computation and verl batch
  conversion. Results:
  - rewards `[1.0, 0.0, 0.6, 1.0]` per group (with the weights then in use, 0.3 / 0.3 /
    0.4; the wrong answer scores 0.5 with the current ones);
  - advantages `[0.855, −1.588, −0.122, 0.855]`;
  - groups keyed by question id;
  - the advantage applied to every response token.
- **Final-checkpoint patch.** Run against rLLM's loop semantics: 157 steps with
  save_freq 20 → saves at 20…140 plus 157; 200 steps → no duplicate save.
- **Tokenizer.** The SFT tokenizer and chat template load under the image's transformers
  5.3.0.

On a GPU (NRP, 2026-10-01): the image was built and pushed, then a 3-step smoke test
ran with the same image and entrypoint as the main run, but with Qwen3-0.6B from the Hub,
full-parameter (`LORA_RANK=0`, `ACTOR_LR=1e-6`), on one A10 (24 GB, driver 595, so no
CUDA forward compatibility needed). The Job finished with exit code 0:

- **Pipeline.** vLLM rollout, reward, GRPO advantages, actor update, weight sync,
  checkpoints and wandb logging all ran; about 14 s per step.
- **Checkpoints.** Saved at step 2; the final-checkpoint patch saved step 3, and the
  checkpoint manager then removed step 2.
- **Rollout and training agree.** `offpolicy/ppl_ratio` 1.001 and
  `training/rollout_actor_probs_pearson_corr` 0.9993 between vLLM and FSDP log-probs.
- **Memory.** Peak 14.0 GB allocated (19.8 GB reserved) with 1,024-token responses.
- **Reward.** Every answer was cut off at 1,024 tokens, so all rewards, advantages,
  `pg_loss` and `grad_norm` were 0, as designed for the smoke test.

**Not done:** anything specific to the 8B run on a 48 GB card:

- loading `qwen3-8b-sft-v3` from the SFT PVC;
- the LoRA path: bf16 base weights, adapter training and LoRA weight sync into vLLM 0.20.2;
- memory peaks and throughput at 8B, first with the smoke test's 1,024-token responses,
  then with 8,192;
- `export` merging the LoRA adapter into the base model.

## Files

```
GRPO/
├── README.md                this file
├── ALGORITHM.md             exact computation, losses, reward, hyperparameters
├── Dockerfile               verlai/verl:vllm020.dev2 + verl 0.8.0 + rLLM 3b40c37 (no-deps installs)
├── docker/
│   └── patch_verl.py        build-time fix: verl checkpoints under FSDP2 CPU offload
├── k8s/
│   ├── pvc.yaml             150Gi PVC for outputs
│   ├── job.yaml             training Job, N_GPUS cards of GPU_TYPE (L40 / A6000 / A100)
│   └── export-job.yaml      CPU Job: checkpoint -> HF model
├── src/
│   ├── prepare_data.py      bespoke-v2 -> data/{train,validation}.jsonl
│   ├── prepare_grpo_try.py  ../grpo_try -> data/grpo_try/{train,validation}.jsonl
│   ├── prepare_rest_grpo.py bespoke-stratos-rest (non-SFT questions) -> data/rest_grpo/,
│   │                        unanswerable ones -> ../no_answer/
│   ├── compute_label_prior.py  bespoke-v2 -> target label mix of the reward (LABEL_PRIOR)
│   ├── train_grpo.py        Hydra entry point: datasets, Ray, rLLM AgentTrainer(backend="verl")
│   ├── entrypoint.sh        env -> Hydra overrides -> train_grpo.py
│   ├── export_policy.py     verl checkpoint -> merged HF model
│   └── labelcot/
│       ├── reward.py        the label-following reward (stdlib only)
│       ├── workflow.py      rLLM Workflow: SFT-identical prompt, generation, scoring
│       └── patches.py       final checkpoint for rLLM's verl backend
├── tests/                 test_reward.py, test_prepare_grpo_try.py, test_prepare_rest_grpo.py,
│                          test_workflow.py
└── data/                    generated, gitignored, baked into the image
```
