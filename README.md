# RLHF for Qwen3 labelled chain-of-thought

The RL stage that follows the SFT project in `../train`. It trains the full-parameter SFT
checkpoint `qwen3-8b-sft-v3` with **GRPO** so that the model reasons in steps labelled
with the eight annotation tags, following the annotation of each question, without losing
answer accuracy. It uses the latest rLLM (unified trainer) on verl, and runs as one
Kubernetes Job on NRP, the same way the SFT job does. A small
[example](#small-example-on-nrp-grpoexample) (Qwen3-0.6B, full-parameter, wandb) tests the
whole pipeline cheaply first.

- **[GRPO/README.md](GRPO/README.md)**: design decisions and what has been verified
- **[GRPO/ALGORITHM.md](GRPO/ALGORITHM.md)**: exact computation, losses, reward and
  hyperparameters

Nothing has run on a GPU yet. Start with the smoke test.

## Quick start

`./run.sh` is the entry point for everything.

```bash
cp .env.example .env
$EDITOR .env          # K8S_NAMESPACE, IMAGE, NRP_REGISTRY_TOKEN (same values as ../train/.env,
                      # but IMAGE ends in /context-comp-grpo:latest); WANDB_API_KEY is optional

./run.sh test         # reward unit tests, local, no .env needed
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
  that the pipeline runs.

| command | what it does |
|---|---|
| `./run.sh` | `data` if missing, then `image`, `secrets`, `pvc`, `submit`, `logs` |
| `./run.sh data` | rebuild the task set |
| `./run.sh test` | reward unit tests |
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
| `TRAIN_BATCH_SIZE`, `GROUP_SIZE` | 8, 8 | questions per step, answers per question |
| `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH` | 2048, 8192 | longer answers score 0 |
| `LORA_RANK`, `LORA_ALPHA`, `ACTOR_LR` | 64, 32, 1e-5 | the LoRA adapter GRPO adds on top of the full SFT weights. `LORA_RANK=0` trains all parameters instead (with a separate frozen reference model); use a full-fine-tune learning rate such as `ACTOR_LR=1e-6` then |
| `PPO_MINI_BATCH_SIZE` | 4 | questions per optimizer update, i.e. 2 updates per step |
| `CLIP_LOW`, `CLIP_HIGH`, `KL_BETA`, `NORM_ADV_BY_STD` | 0.2, 0.28, 0.001, true | see ALGORITHM.md §3 |
| `REWARD_W_TAG`, `REWARD_W_ALIGN`, `REWARD_W_CORRECT` | 0.3, 0.3, 0.4 | see ALGORITHM.md §4 |
| `TOTAL_TRAINING_STEPS`, `EPOCHS` | 200, 1 | step cap; -1 = full epochs (630 steps) |
| `SAVE_FREQ`, `TEST_FREQ`, `VAL_BEFORE_TRAIN` | 20, 20, true | a final checkpoint and a final validation always happen |
| `ROLLOUT_GPU_MEM_UTIL` | 0.7 | vLLM's share of the GPU while generating |
| `RUN_NAME`, `OUTPUT_ROOT` | `qwen3-8b-grpo-labels-v1`, `/grpo/runs` | |
| `SMOKE_TEST` | false | true: 3 tiny steps; overrides the size settings above |
| `REPORT_TO`, `WANDB_API_KEY`, `WANDB_PROJECT` | `wandb`, empty, `context-comp-grpo` | as in `../train`: `REPORT_TO=wandb` streams all metrics to wandb.ai and needs `WANDB_API_KEY` (or `WANDB_MODE=offline`); empty `REPORT_TO` = console only |

## Small example on NRP (`GRPO/example/`)

`GRPO/example/` runs the same training as a separate, small Job on NRP. It is configured
exactly like `../train`: one `.env` turned into a Secret, `run.sh` for image, Secrets,
PVC, Job and logs, and wandb switched on with `REPORT_TO=wandb`. Use it to check the
whole pipeline and the wandb dashboards before spending A100-80GB hours on the 8B run.

| | main run (`./run.sh`) | example (`GRPO/example/run.sh`) |
|---|---|---|
| policy | `qwen3-8b-sft-v3` from the SFT PVC | `Qwen/Qwen3-0.6B`, pulled from the Hub into `/grpo/hf` |
| training | LoRA adapter (rank 64) | full-parameter (`LORA_RANK=0`, `ACTOR_LR=1e-6`), separate frozen reference model |
| length | 200 steps | 50 steps (400 questions), validation every 10 |
| GPU | 1× A100-80GB | 1× L40, L40S, RTX 5000 Ada, A10, L4 or RTX 4090 (24–48 GB, `nvidia.com/gpu`, no A100 quota needed); the full 50 steps may need one of the 48 GB cards |
| pod | 8 CPU, 160 GiB | 8 CPU, 64 GiB |
| Secret / Job | `grpo-env` / `qwen-grpo` | `grpo-example-env` / `qwen-grpo-example` |
| data, image, PVC | `GRPO/data`, `context-comp-grpo`, `qwen-grpo-data` | the same |

### Configure

```bash
cd GRPO/example
cp .env.example .env
$EDITOR .env
```

- **Cluster:** `K8S_NAMESPACE`, `IMAGE` and `NRP_REGISTRY_TOKEN`, the same values as the
  root `.env`.
- **wandb:** `WANDB_API_KEY` from https://wandb.ai/authorize. `REPORT_TO=wandb` is already
  set, and `run.sh secrets` refuses to continue while the key is empty. Use
  `WANDB_MODE=offline` to keep the logs on the PVC instead.
- **Everything else** is preset for the example and uses the variable names from the
  [Settings](#settings) table, so any of them can be changed here. `.env` is gitignored.

### Run

```bash
./run.sh              # data if missing -> image -> secrets -> pvc -> job -> follow logs
./run.sh status       # job, pods, recent events
./run.sh clean        # delete the Job
```

- **If the image was already built and pushed by the root `./run.sh`**, skip the build:
  `./run.sh secrets && ./run.sh pvc && ./run.sh submit && ./run.sh logs`.
- **Changes to `.env`** need `./run.sh secrets && ./run.sh submit`, because the Secret is
  a snapshot.
- **PVC sharing:** the example shares the PVC with the main run, which is ReadWriteOnce,
  so do not run both Jobs at the same time.

### What to watch in wandb

The project is `WANDB_PROJECT` and the run name is `RUN_NAME`.

| metric | meaning |
|---|---|
| `batch/reward`, `reward/policy/mean` | mean reward of the step's 64 answers |
| `val/bespoke_labeled_cot/reward`, `val/bespoke_labeled_cot/pass@1` | reward and accuracy on the 102 held-out questions (step 0, every 10 steps, end) |
| `batch/answer_correct`, `batch/tag_rate`, `batch/truncated` | accuracy, label-format use, length-limit hits |
| `actor/ppo_kl`, `actor/pg_clipfrac`, `actor/entropy` | update size and policy entropy |

Qwen3-0.6B was never taught the `[label]` format. `tag_rate` and `label_alignment` stay
near 0, so the reward is mostly 0 (wrong or cut off) or 0.4 (right answer), and
`batch/reward` ≈ 0.4 × accuracy. The example shows that the pipeline and the dashboards
work; the label terms only become meaningful with the SFT model.

## Troubleshooting

- **`CUDA driver version is insufficient`, or error 803/804.** The image is CUDA 13. On
  drivers older than 580 the entrypoint switches on CUDA forward compatibility. The driver
  version is printed at the top of the log. If that fails, rebuild with another base
  (`docker build --build-arg BASE_IMAGE=...`).
- **CUDA OOM.**
  - Lower `MAX_RESPONSE_LENGTH` first.
  - If the OOM happens while generating, lower `ROLLOUT_GPU_MEM_UTIL` or `GROUP_SIZE`.
  - Lower `LORA_RANK`.
- **Pending.** Either no free A100-80GB, or a Multi-Attach error: both PVCs are
  ReadWriteOnce, so no other pod may hold them on another node.
- **Job `Running 0/1` with no pod, and events say `exceeded quota: a100-limit`.** The
  namespace's A100 quota is 0 (`kubectl get resourcequota`), so the pod is rejected before
  it is scheduled. Either ask the NRP admins to raise it (A100 access request), or add
  `priorityClassName: opportunistic` to the pod spec, which bypasses the GPU quota but can
  be preempted at any time. The example avoids this by using generic GPUs
  (`nvidia.com/gpu`).
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
    └── example/      small NRP example (Qwen3-0.6B, full-parameter, wandb): run.sh, .env.example, k8s/job.yaml
```
