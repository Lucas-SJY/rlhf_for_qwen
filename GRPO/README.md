# GRPO for labelled chain-of-thought: design notes

This directory holds the GRPO implementation: code, image and Kubernetes manifests. The
entry points are **`../run.sh`** (the main run) and **`example/run.sh`** (a small
Qwen3-0.6B run on NRP with wandb, for testing the pipeline). How to run both is in the
[root README](../README.md). The exact per-step computation, losses, reward and
hyperparameters are specified in **[ALGORITHM.md](ALGORITHM.md)**. This file explains
the design decisions and what has been verified.

Nothing here has run on a GPU yet. **Run the smoke test first.**

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

The reward is unchanged from the PPO version: a rule-based score of how well the chain
of thought follows the annotated labels, plus answer correctness.

```
R = 0                                                                if the thought never closes or hits 8,192 tokens
R = 0.3·tag_rate + 0.3·label_alignment + 0.4·answer_correct            otherwise
```

- `tag_rate`: the share of paragraphs that open with one of the eight tags.
- `label_alignment`: the similarity of the label mix and order to the annotation of the
  same question.
- `answer_correct`: `\boxed{}` after `</think>` against the reference answer, using the
  scorer from `../../train/evaluate`.

Full definitions and worked examples are in [ALGORITHM.md §4](ALGORITHM.md#4-the-reward-r).
Scoring each SFT target as if it were a model answer gives 1.0 for all 5,042.

GRPO fits this reward well. When all 8 answers to a question are equally right or wrong,
only the label terms differ, and group normalisation turns those small differences into
full-size advantages. The same scaling also amplifies noise; `NORM_ADV_BY_STD=false`
switches it off.

## 3. Fitting it on one A100-80GB

- **LoRA on the actor, added by GRPO.** rank 64, alpha 32, all linear layers, lr 1e-5.
  - The SFT checkpoint is full-parameter and has no adapter. GRPO adds a fresh adapter
    (B = 0, so step 0 equals the SFT model) and trains only that.
  - The SFT weights stay frozen, so the reference policy for the KL term is the same model
    with the adapter switched off; no second 8B model is needed.
  - `../run.sh export` merges the adapter back into the full weights.
- **Rollout and training take turns on the GPU.**
  - vLLM uses 70 % of the card while it generates 64 answers, and sleeps during training.
  - The actor's weights and optimizer state are offloaded to host memory while vLLM runs.
- **Dynamic batching by tokens** (12,288 tokens per micro-batch). Memory then scales with
  tokens, not with the number of sequences.

The pod requests 8 CPUs, 160 GiB RAM (including a 24 GiB `/dev/shm` for Ray's 16 GiB
object store) and one A100-80GB (40 GB cards excluded).

## 4. What is custom, and why

| piece | reason |
|---|---|
| `labelcot/workflow.py` renders the prompt with the checkpoint's own chat template | rLLM's `QwenChatTemplateParser` prepends a default "You are Qwen, created by Alibaba Cloud…" system turn the SFT model never saw. Generation still uses rLLM's token-in/token-out path, so training uses the exact sampled tokens and logprobs. |
| `labelcot/patches.py` saves a final checkpoint | rLLM's verl backend saves only at multiples of `save_freq` and does nothing at the end of training, so up to 19 steps would be lost. It is applied inside the Ray actor that builds the trainer: `workflow.py` calls it on import, and the actor imports that module when it receives the workflow class. |
| data as plain task dicts (`GRPO/data/*.jsonl`) | the unified trainer takes rLLM `Dataset` objects directly: no verl parquet, no dataset registry |

Nothing in rLLM or verl is edited.

## 5. Data

`src/prepare_data.py` (run through `../run.sh data`) reads `../train/bespoke-v2` and writes
one task per line: `id`, `data_source`, `question`, `answer`, `ref_labels`.

- The prompt is the bare question, exactly as in SFT.
- The split reuses the SFT held-out ids: 5,042 train and 102 validation. The validation
  questions were never trained on in either stage.

## 6. Cost

The time per step is set by the longest of the 64 answers (up to 8,192 tokens), plus
three passes of the 8B actor over all answer tokens: old log-probs, reference log-probs,
and the update. My estimate is **~10–12 minutes per step**, so the default cap of 200 steps
(1,600 questions) is roughly 1.5 days. A full epoch at 8 questions per step would be
630 steps. Treat these numbers as a guess until the smoke test reports `timing_s/*`.

One actor checkpoint is ~33 GB, because FSDP keeps the LoRA-wrapped 8B in fp32. Only the
latest is kept, on the 150 Gi PVC `qwen-grpo-data`.

## Verification

Done locally, without a GPU:

- **Reward and data.**
  - `../run.sh test`: 11 reward unit tests.
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
  - rewards `[1.0, 0.0, 0.6, 1.0]` per group;
  - advantages `[0.855, −1.588, −0.122, 0.855]`;
  - groups keyed by question id;
  - the advantage applied to every response token.
- **Final-checkpoint patch.** Run against rLLM's loop semantics: 157 steps with
  save_freq 20 → saves at 20…140 plus 157; 200 steps → no duplicate save.
- **Tokenizer.** The SFT tokenizer and chat template load under the image's transformers
  5.3.0.

**Not done:** building the image and any GPU run. The smoke test will settle:

- the CUDA 13 base image on NRP's drivers;
- memory peaks;
- LoRA weight sync into vLLM 0.20.2;
- throughput.

## Files

```
GRPO/
├── README.md                this file
├── ALGORITHM.md             exact computation, losses, reward, hyperparameters
├── Dockerfile               verlai/verl:vllm020.dev2 + verl 0.8.0 + rLLM 3b40c37 (no-deps installs)
├── example/               small NRP example: Qwen3-0.6B, full-parameter, wandb
│   ├── run.sh               data -> image -> secrets -> pvc -> job -> logs, like ../../train/run.sh
│   ├── .env.example         its configuration; copy to .env (gitignored)
│   └── k8s/job.yaml         1x 24-48 GB Ampere/Ada GPU, 64 GiB
├── k8s/
│   ├── pvc.yaml             150Gi PVC for outputs
│   ├── job.yaml             training Job, 1x A100-80GB
│   └── export-job.yaml      CPU Job: checkpoint -> HF model
├── src/
│   ├── prepare_data.py      bespoke-v2 -> data/{train,validation}.jsonl
│   ├── train_grpo.py        Hydra entry point: datasets, Ray, rLLM AgentTrainer(backend="verl")
│   ├── entrypoint.sh        env -> Hydra overrides -> train_grpo.py
│   ├── export_policy.py     verl checkpoint -> merged HF model
│   └── labelcot/
│       ├── reward.py        the label-following reward (stdlib only)
│       ├── workflow.py      rLLM Workflow: SFT-identical prompt, generation, scoring
│       └── patches.py       final checkpoint for rLLM's verl backend
├── tests/test_reward.py
└── data/                    generated, gitignored, baked into the image
```
