# RLHF for Qwen3 labelled chain-of-thought

The RL stage that follows the SFT project in `../train`. It trains the full-parameter SFT
checkpoint `qwen3-8b-sft-v3` with **GRPO** so that the model reasons in steps labelled
with the eight annotation tags, in the label mix of a typical annotated trace, without
losing answer accuracy. It uses the latest rLLM (unified trainer) on verl, and runs as one
Kubernetes Job on NRP, the same way the SFT job does.

- **[GRPO/README.md](GRPO/README.md)**: design decisions and what has been verified
- **[GRPO/ALGORITHM.md](GRPO/ALGORITHM.md)**: exact computation, losses, reward and
  hyperparameters

The shared training path (vLLM rollout, reward, GRPO update, weight sync, checkpoints,
wandb) has run on a GPU once: a 3-step smoke test of Qwen3-0.6B, full-parameter, on an
A10 (see [GRPO/README.md](GRPO/README.md#verification)). The 8B LoRA run itself has not, so
start it with the smoke test.

## Quick start

`./run.sh` is the entry point for everything.

```bash
cp .env.example .env
$EDITOR .env          # K8S_NAMESPACE, IMAGE, NRP_REGISTRY_TOKEN (same values as ../train/.env,
                      # but IMAGE ends in /context-comp-grpo:latest); WANDB_API_KEY is optional

./run.sh test         # unit tests, local, no .env needed
./run.sh data         # build GRPO/data/{train,validation}.jsonl from ../train/bespoke-v2

# 1) smoke test: set SMOKE_TEST=true in .env, then
./run.sh              # data -> image -> secrets -> pvc -> job -> follow logs

# 2) real run: set SMOKE_TEST=false in .env, then
./run.sh secrets && ./run.sh submit && ./run.sh logs

# 3) after training: turn the latest checkpoint into a normal HF model directory
./run.sh export
```

The smoke test runs 3 steps with 2 questions × 4 answers and 1,024-token responses, and
checkpoints at steps 2 and 3. It writes to `<RUN_NAME>-smoke`.

- `SMOKE_TEST=true` overrides the size settings in `.env`, so there is no need to shrink
  them by hand.
- Almost every answer is cut off at 1,024 tokens, so its rewards are ~0; it only proves
  that the pipeline runs. The smoke test keeps cut-off answers in training
  (`MASK_TRUNCATED=false`), so the update step still runs.

| command | what it does |
|---|---|
| `./run.sh` | `data` if missing, then `image`, `secrets`, `pvc`, `submit`, `logs` |
| `./run.sh data` | rebuild the task set |
| `./run.sh test` | unit tests (reward, grpo_try task builder) |
| `./run.sh image` | `docker login`, build (context `GRPO/`), push |
| `./run.sh secrets` | upload `.env` as Secret `grpo-env`, refresh the `nrp-registry` pull secret |
| `./run.sh pvc` | create PVC `qwen-grpo-data` (150Gi) if missing; checks that the SFT PVC exists |
| `./run.sh submit` / `logs` / `status` | (re)submit the training Job, follow it, show jobs, pods and events |
| `./run.sh export` | CPU Job: latest checkpoint → merged HF model under `/grpo/runs/<RUN_NAME>/export/` |
| `./run.sh clean` | delete the Jobs; Secrets and PVCs stay |

After editing `.env`, run `./run.sh secrets && ./run.sh submit`; the Secret is a snapshot.
A restarted pod resumes from the newest checkpoint of its `RUN_NAME`; use a new
`RUN_NAME` to start over.

## Settings

All settings live in `.env` (template: `.env.example`).

| variable | default | notes |
|---|---|---|
| `POLICY_MODEL_PATH` | `/data/runs/qwen3-8b-sft-v3` | the SFT checkpoint on the SFT PVC, mounted read-only at `/data` |
| `POLICY_MODEL_REVISION` | unset | set it to train from the Hub: `POLICY_MODEL_PATH` is then a repo id (e.g. `Lucas-SJY/qwen3-8b-sft-bespoke`) and this a branch, tag or commit (e.g. `v3`). The snapshot is downloaded once into `HF_HOME` (`/grpo/hf`, on the RL PVC) and reused by later runs; a private repo needs `HF_TOKEN` |
| `TRAIN_BATCH_SIZE`, `GROUP_SIZE` | 8, 8 | questions per step, answers per question |
| `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH` | 2048, 8192 | an answer cut off at the limit is dropped from training (see `MASK_TRUNCATED`) |
| `MASK_TRUNCATED` | true | an answer cut off at `MAX_RESPONSE_LENGTH` is left out of the training batch (no reward, no gradient, not part of its group's mean/std; rLLM's compact filtering, DAPO's overlong filtering) instead of scoring 0; validation still counts it as wrong. `false`: train on it with reward 0. Smoke tests use `false` |
| `LORA_RANK`, `LORA_ALPHA`, `ACTOR_LR` | 64, 32, 1e-5 | the LoRA adapter GRPO adds on top of the full SFT weights, whose frozen copy is kept in bf16 (`ACTOR_MODEL_DTYPE=fp32` restores verl's default, which does not fit a 48 GB card). `LORA_RANK=0` trains all parameters instead (with a separate frozen reference model); use a full-fine-tune learning rate such as `ACTOR_LR=1e-6` then. For the 8B model that needs `N_GPUS=4` A100s, see [Full-parameter training](#full-parameter-training) |
| `PPO_MINI_BATCH_SIZE` | 4 | questions per optimizer update, i.e. 2 updates per step |
| `CLIP_LOW`, `CLIP_HIGH`, `KL_BETA`, `NORM_ADV_BY_STD` | 0.2, 0.28, 0.001, true | see ALGORITHM.md §3 |
| `REWARD_W_LABEL`, `REWARD_W_CORRECT` | 0.2, 0.8 | weights of valid labels (the thought has tags and all of them are among the eight) and of answer correctness; no reference trace; `tag_rate` and `label_mix` are logged, not rewarded; see ALGORITHM.md §4 |
| `MAX_LABEL_RETRIES` | 3 | an answer whose thought has a tag outside the eight labels is sampled again, up to this many times; still invalid after that, it gets no label reward (only the correctness weight) |
| `TOTAL_TRAINING_STEPS`, `EPOCHS` | 200, 1 | step cap; -1 = full epochs (630 steps) |
| `SAVE_FREQ`, `TEST_FREQ`, `VAL_BEFORE_TRAIN` | 20, 20, true | a final checkpoint and a final validation always happen |
| `ROLLOUT_GPU_MEM_UTIL` | 0.7 | vLLM's share of the GPU while generating |
| `PPO_MAX_TOKEN_LEN` | prompt + response + 2048 (12,288) | tokens per micro-batch in the update and the log-prob passes. Under CPU offload each micro-batch pays a fixed ~40 s to stream weights in and gradients out (measured ~46 s per 12k micro-batch on one A100), so 32,768 (~60 GB peak on an 80 GB card, logits dominate) cuts the update time to about a third |
| `RUN_NAME`, `OUTPUT_ROOT` | `qwen3-8b-grpo-labels-v1`, `/grpo/runs` | |
| `SMOKE_TEST` | false | true: 3 tiny steps; overrides the size settings above |
| `REPORT_TO`, `WANDB_API_KEY`, `WANDB_PROJECT` | `wandb`, empty, `context-comp-grpo` | as in `../train`: `REPORT_TO=wandb` streams all metrics to wandb.ai and needs `WANDB_API_KEY` (or `WANDB_MODE=offline`); empty `REPORT_TO` = console only |
| `GPU_TYPE`, `GPU_PRIORITY_CLASS` | `l40`, empty | the card: `l40` (L40 / L40S, 48 GB), `a6000` (RTX A6000, 48 GB) or `a100` (A100-80GB). `a100` runs at priority `opportunistic` unless `GPU_PRIORITY_CLASS` is set: no quota needed, but the pod can be preempted (it then resumes from the last checkpoint). The namespace has an A100 quota of 4 since 2026-10-05, so `GPU_PRIORITY_CLASS=` (empty, normal priority) runs without preemption |
| `N_GPUS` | 1 | cards on the one node the pod runs on; the Job requests this many and FSDP shards the actor across them, with one vLLM replica per card |
| `FSDP_CPU_OFFLOAD`, `OMP_NUM_THREADS` | false, unset | true: FSDP2 with CPU offload. The actor's weights, gradients and Adam state stay in host RAM, layers go to the GPU only while computed, and Adam steps on the CPU with `OMP_NUM_THREADS` threads per worker (Ray's default is 1). Much less GPU memory, slower steps; see [Full-parameter training](#full-parameter-training) |
| `POD_CPU`, `POD_MEMORY`, `RAY_OBJECT_STORE_GB`, `RAY_NUM_CPUS` | 8, `160Gi`, 16, unset | training pod size, sized for the 8B LoRA run on one card (the 4-card full-parameter run uses 16 / `256Gi`). A small model fits 4 / `64Gi` / 8 and schedules far more easily; with `POD_CPU` below 8 set `RAY_NUM_CPUS=8`, or Ray runs out of CPUs to hand to verl and vLLM |
| `TRAIN_FILE`, `VAL_FILE` | `/workspace/data/{train,validation}.jsonl` | task files inside the image: bespoke-v2 by default, or [grpo_try](#training-on-grpo_try) / [rest_grpo](#training-on-rest_grpo) (the current runs); all sets and their counts are in [data_distrib.md](data_distrib.md) |

## Full-parameter training

`LORA_RANK=0` updates all 8B weights instead of a LoRA adapter. fp32 weights, gradients and
Adam state come to ~128 GB, more than one card holds. Two ways to fit it:

| | 4 A100s, FSDP1 | 1–2 A100s, FSDP2 CPU offload |
|---|---|---|
| settings | `N_GPUS=4` | `N_GPUS=1` or `2`, `FSDP_CPU_OFFLOAD=true`, `OMP_NUM_THREADS` = ~10 CPUs per card |
| where weights / grads / Adam live during the update | on the GPUs, sharded 4 ways (~49 GB per card at the optimizer step) | in pinned host RAM; only the layers being computed are on the GPU |
| optimizer step | GPU | CPU |
| speed | faster | slower (host-device copies every pass, CPU Adam) |
| verified on a GPU | config only | config only |

Both use:

```bash
LORA_RANK=0
ACTOR_LR=1e-6
GPU_TYPE=a100
GPU_PRIORITY_CLASS=
POD_CPU=16          # 14 for one card
POD_MEMORY=256Gi    # 288Gi with CPU offload on one card
```

With CPU offload the host-RAM need does not shrink with fewer cards: the actor's fp32
weights, gradients and Adam state (~132 GB), the CPU Adam step's temporaries (~33 GB) and
the fp32 reference model (~33 GB) are all in host RAM, ~240 GB at peak.

- The KL reference is a separate frozen copy of the SFT model, kept in host RAM (CPU
  offload) in both modes, and vLLM receives all weights after every update.
- The pod waits (Pending) until one node has `N_GPUS` free A100s and enough free CPU and
  RAM. 4 A100s are the namespace's whole A100 quota.
- Checkpoints keep the weights and the training position but not the Adam state: ~33 GB
  instead of ~100 GB. verl keeps the previous checkpoint until the next one is written, and
  two full checkpoints would not fit on the 150Gi RL PVC. A resumed run therefore restarts
  Adam's moments. `CKPT_SAVE_OPTIMIZER=true` keeps them, after growing the PVC (e.g. to
  300Gi).

## Training on grpo_try

`grpo_try/` holds 9 annotated traces in the bespoke-v2 layout, without reference answers.
`GRPO/src/prepare_grpo_try.py` turns them into the same task format as `GRPO/data/`:

```bash
python3 GRPO/src/prepare_grpo_try.py               # -> GRPO/data/grpo_try/{train,validation}.jsonl
python3 GRPO/src/prepare_grpo_try.py --strict      # stop on the first malformed input file
python3 GRPO/src/prepare_grpo_try.py --check-only  # only check the existing output files
```

- **Output.** One task per line with only `id`, `data_source`, `prompt` (chat format:
  one user turn holding the question, as in SFT), `question` and `answer`. The reference
  trace and its labels are not written: the reward does not use them.
- **Format check.** Every input file must be valid JSON with a string `id` and
  `question` (the labelled spans are not used). A file that fails is reported with the
  file, line and column or field, and skipped (`--strict`: the run stops); an id that
  does not match the file name only warns. The written files are read back and checked
  line by line: field types, no unexpected keys, a well-formed `prompt` whose last user
  turn holds the `question`, and no id twice. Any error there exits with status 1.

- **Answers.** Recovered by sample id from the upstream harbor dataset
  (`../jianhong_harbor/harbor/datasets/bespoke-stratos-rest`), as `../train` did for
  bespoke-v2. 7 of the 9 have a `\boxed{}` answer; the two coding questions do not, so
  their reward uses only the label term.
- **Split.** By trace: 6 train, 3 validation. The validation traces default to the three
  held out by the earlier step-level export (`../train/grpo_test.json`, since removed):
  `sample_008604`, `sample_013628`, `sample_014855` (`--val-ids` to change them). One of
  them is a coding question without an answer, so validation `pass@1` tops out at 2/3.

To train on it, add to `.env` and rebuild the image (the data is baked in):

```bash
TRAIN_FILE=/workspace/data/grpo_try/train.jsonl
VAL_FILE=/workspace/data/grpo_try/validation.jsonl
TRAIN_BATCH_SIZE=2        # at most 6: incomplete batches are dropped, 8 would leave none
PPO_MINI_BATCH_SIZE=1     # at most TRAIN_BATCH_SIZE
EPOCHS=20                 # 3 steps per epoch at TRAIN_BATCH_SIZE=2
```

Validation metrics are then logged as `val/grpo_try/...`. `SMOKE_TEST=true` already uses
2 questions per step, so it works on this set unchanged.

## Training on rest_grpo

`GRPO/data/rest_grpo/` holds the Bespoke-Stratos-17k questions the SFT data did not use:
the full set has 16,710 samples, `../train/data_labeled_2` holds 5,144 of them, and the
harbor dataset `../jianhong_harbor/harbor/datasets/bespoke-stratos-rest` holds exactly
the other 11,566. `GRPO/src/prepare_rest_grpo.py` turns those into tasks in the grpo_try
format (`id`, `data_source`, `prompt`, `question`, `answer`):

```bash
python3 GRPO/src/prepare_rest_grpo.py               # -> GRPO/data/rest_grpo/ and no_answer/
python3 GRPO/src/prepare_rest_grpo.py --strict      # stop on the first malformed source file
python3 GRPO/src/prepare_rest_grpo.py --check-only  # only check the existing output files
python3 GRPO/src/prepare_rest_grpo.py --no-answer-dir ''  # keep every question in rest_grpo
```

- **Source.** The question and the `\boxed{}` answer come from each sample's
  `environment/trajectory.json`; the reference reasoning is not written. Every id in
  `../train/data_labeled_2/*.jsonl` is left out even if the source has it
  (`--exclude-from`), so the set cannot overlap the SFT data (it does not: 0 excluded).
- **Split.** Seeded random by question, 10 % validation: 10,409 train / 1,157 validation
  over all 11,566 questions.
- **Only checkable answers stay.** A question whose answer the reward cannot check
  (`answer_is_checkable`) would be scored on the label term alone, so after the split it
  is moved, unchanged and in the same split, to `no_answer/` at the repository root
  (gitignored): 5,410 questions without an answer (all 5,395 coding questions, "Generate
  an executable Python function ...", and 15 science), 792 prose or proof answers (552
  science, 240 math, e.g. `\text{No}`), and 52 questions whose solution boxes several
  distinct values (several roots, multi-part questions), for which the last `\boxed{}`
  kept as the answer is incomplete. `no_answer/` holds 5,623 / 631.
- **Held out for manual inspection.** 10 of those training questions, drawn at random
  (seeded), are moved to `manual_inspection/` at the repository root and never trained
  on: `tasks.jsonl` in the task format, and `trajectories/<id>.json`, a copy of each
  upstream trajectory (question, DeepSeek-R1 reasoning, solution). The validation set is
  untouched. `GRPO/data/rest_grpo/` therefore has **4,776 train / 526 validation**
  questions, 4,560 / 495 math and 216 / 31 science (`--inspect-count` changes the number
  held out, `--inspect-dir ''` holds out none).
- **Breakdown.** [data_distrib.md](data_distrib.md) tabulates where every one of the
  11,566 questions went, the composition of each set, and the distribution before the
  separation.
- **Format check.** As for grpo_try: malformed source files are reported and skipped
  (`--strict`: the run stops), and the written files are read back and checked.

To train on it, add to `.env` and rebuild the image (the data is baked in):

```bash
TRAIN_FILE=/workspace/data/rest_grpo/train.jsonl
VAL_FILE=/workspace/data/rest_grpo/validation.jsonl
```

Validation metrics are then logged as `val/rest_grpo/...`. One validation pass over all
526 questions (one answer each, up to 8,192 tokens) took ~25 min on one A100 and ~9 min on
two; a smaller `VAL_FILE` keeps it cheaper. The SFT checkpoint scores pass@1 0.80–0.83 on
it before training (three runs), 14–16 % of its answers cut off at the length limit.

## What to watch in wandb

The project is `WANDB_PROJECT` and the run name is `RUN_NAME` (`<RUN_NAME>-smoke` for the
smoke test). Every step logs about 140 metrics; these are the ones to watch:

| metric | meaning |
|---|---|
| `batch/reward`, `reward/policy/mean` | mean reward of the step's answers, without the ones cut off at the length limit (those are not trained on) |
| `val/bespoke_labeled_cot/reward`, `val/bespoke_labeled_cot/pass@1` | reward and accuracy on the 102 held-out questions (step 0, every `TEST_FREQ` steps, end) |
| `batch/answer_correct`, `batch/label_valid`, `batch/truncated` | the reward terms, and how often answers hit the length limit |
| `groups/num_trajs_before_filter`, `groups/num_trajs_after_filter` | answers generated in the step, and how many are left for training once the cut-off ones are dropped |
| `batch/label_mix`, `batch/share_<label>` | how the label shares compare with the average annotated trace (logged only) |
| `batch/tag_rate`, `batch/invented_tag_rate` | share of paragraphs with a valid tag, and with a made-up one (logged only, not rewarded) |
| `batch/label_retries`, `batch/invalid_label` | regenerations per answer because of a tag outside the eight labels, and the share still invalid after the last retry |
| `batch/policy/fractions/effective` | share of questions whose 8 answers got different rewards; only these produce a gradient |
| `actor/ppo_kl`, `actor/pg_clipfrac`, `actor/entropy` | update size and policy entropy |
| `perf/max_memory_allocated_gb`, `timing_s/step` | GPU memory peak and seconds per step |

In the smoke test every answer is cut off at 1,024 tokens, so `batch/truncated` is 1,
the reward and all advantages are 0, and `actor/pg_loss` and `grad_norm` are 0: the
pipeline runs, but the model does not change.

## Troubleshooting

- **`CUDA driver version is insufficient`, or error 803/804.** The image is CUDA 13. On
  drivers older than 580 the entrypoint switches on CUDA forward compatibility. The driver
  version is printed at the top of the log. If that fails, rebuild with another base
  (`docker build --build-arg BASE_IMAGE=...`).
- **CUDA OOM.** The 8B run is tight on a 48 GB card.
  - Lower `MAX_RESPONSE_LENGTH` first.
  - If the OOM happens while generating, lower `ROLLOUT_GPU_MEM_UTIL` or `GROUP_SIZE`.
  - Lower `LORA_RANK`.
  - If none of that helps, move to A100-80GB: `GPU_TYPE=a100`.
- **Pending.** Either no free card of `GPU_TYPE` (`./run.sh status` shows `Insufficient
  nvidia.com/gpu`, `.../rtxa6000` or `.../a100`; try another `GPU_TYPE`), or a
  Multi-Attach error: both PVCs are ReadWriteOnce, so no other pod may hold them on
  another node.
- **Job `Running 0/1` with no pod, and events say `exceeded quota: a100-limit`.** The
  Job asks for more A100s than the namespace's quota has left (4 in total; `kubectl get
  resourcequota a100-limit` shows how many are in use), at normal priority
  (`GPU_PRIORITY_CLASS=` empty). Wait for the other A100 pods to finish, lower `N_GPUS`,
  or leave `GPU_PRIORITY_CLASS` unset (priority `opportunistic`, which bypasses the quota
  but can be preempted).
- **The pod disappears and a new one starts (A100).** With `opportunistic` priority the
  pod can be preempted by higher-priority work. The Job retries up to twice and the new
  pod resumes from the newest checkpoint of its `RUN_NAME`; a lower `SAVE_FREQ` loses less
  work.
- **`ValueError: ... expandable segments`.** A `PYTORCH_*ALLOC_CONF` got through. The
  entrypoint strips `expandable_segments:True`; check `kubectl exec <pod> -- env`.
- **Image build fails in the import check on a Mac.** The emulated amd64 build could not
  run torch. Build with `--build-arg IMPORT_CHECK=0`.

## Layout

```
rlhf/
├── run.sh            entry point
├── .env.example      every setting; copy to .env (gitignored)
├── README.md         this file
├── frameworks.md     which environment standards SkyRL, rLLM, verl and prime-rl accept
└── GRPO/             code, image, k8s manifests, design notes (README.md, ALGORITHM.md)
```
