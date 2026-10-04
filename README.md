# Labelled-CoT Qwen GRPO training

This is the Dayallen harness, adapted from the team's `master` commit `58624bb`.
It retains the team's **rLLM unified trainer → verl → vLLM** workflow and reward:
50% reference-label alignment and 50% automatically checkable final-answer correctness.
GRPO trains a fresh LoRA adapter on the full Qwen SFT checkpoint.

The architecture, algorithm and reward details are in [GRPO/ALGORITHM.md](GRPO/ALGORITHM.md).
Historical design decisions and the teammate's small-model GPU results are in
[GRPO/README.md](GRPO/README.md). Those GPU results do not validate this modified branch.

## What's added on Dayallen

- Task schema, duplicate-ID and train/validation leakage checks.
- Batch, reward-weight and token-budget checks; all prompts are checked with the actual
  checkpoint tokenizer before launching Ray workers.
- Recipe, data/order, source-code and dependency fingerprints in `run_manifest.json`.
  Incompatible automatic resume is refused; use a new `RUN_NAME` for changed experiments.
- Stable WandB run ID across restarts; two retained checkpoints by default; final-save
  patch preserved. Resume requires model, optimizer, RNG/scheduler and dataloader files.
- Training fails on workflow exceptions and selected non-finite optimization metrics.
  Five consecutive batches without informative reward groups stop a normal run
  (`MAX_ZERO_SIGNAL_STEPS=0` disables this; smoke mode disables it automatically).
- Separate Dayallen Job, Secrets, image tag, PVC and run names.
- Secrets are applied without deleting them; only an explicit runtime allowlist enters
  the training container. Registry credentials stay in the image-pull Secret.
- Job logs follow replacement pods and return failure when the Job fails. Submitting
  refuses to replace an existing Job; `clean` is a separate explicit command.
- Export handles smoke directories, verifies checkpoint shards, refuses missing LoRA
  adapters/overwriting exports, stages output, merges on CPU and validates HF files.
- CPU tests, a scripted gradient diagnostic, and CI. Docker also runs the real rLLM
  workflow tests with the pinned dependencies.

## Local checks (no GPU or cluster)

Python 3.11+ is recommended. The basic tests need only the standard library; math
equivalence and optimizer tests need the optional CPU dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r GRPO/requirements-test.txt
./run.sh test
./run.sh diagnose
```

`diagnose` scores four scripted responses, verifies mixed/zero-variance advantages and,
when torch is installed, performs an actual update of a tiny categorical policy.
It is **not a Qwen/vLLM/FSDP integration test**. To require that update:

```bash
python GRPO/src/diagnose.py --require-torch
```

## Data

The production dataset and SFT weights are external to this repository.
Configure `BESPOKE_DIR` and `SFT_SPLIT_DIR` in `.env` to point to your SFT project.
The prepared task format is:

```json
{"id":"q1","data_source":"bespoke_labeled_cot","question":"What is 4+5?","answer":"9","ref_labels":["logical_deduction","concluding"]}
```

`question` alone is prompted; the gold answer and reference labels are reward metadata,
never appended to the model prompt. The original SFT held-out IDs define validation.

```bash
cp .env.example .env
# Edit plain KEY=value values: no shell quoting or inline comments.
./run.sh data
./run.sh preflight
```

`preflight` prints counts, dataset hashes, effective settings and warnings. The training
driver repeats validation using the final resolved Hydra configuration, including CLI
overrides, then checks every prompt's tokenized length. Overlong prompts are rejected
before worker startup rather than silently losing GRPO groups.

The inherited nine-trace builder is also available:

```bash
python GRPO/src/prepare_grpo_try.py --strict
```

For that dataset set `TRAIN_FILE=/workspace/data/grpo_try/train.jsonl`,
`VAL_FILE=/workspace/data/grpo_try/validation.jsonl`, `TRAIN_BATCH_SIZE=2`,
`PPO_MINI_BATCH_SIZE=1`, and a suitable `EPOCHS` value (for example 20).
Data is baked into the image; rebuild after changes. Local preflight maps
`/workspace/data/...` to `GRPO/data/...`. Other locations can use
`LOCAL_TRAIN_FILE` / `LOCAL_VAL_FILE` for the local check.

## First cluster run

Fill in `K8S_NAMESPACE`, `IMAGE`, `NRP_REGISTRY_TOKEN`, `POLICY_MODEL_PATH`,
the SFT PVC name, and `WANDB_API_KEY` in `.env`. Use versioned model paths and image tags.
The defaults request **one L40/L40S, 8 CPUs, 160 GiB RAM**, with a separate 150 GiB output
PVC. These are a starting configuration, not a guarantee that 8B/8K responses fit.

Set `SMOKE_TEST=true` in `.env`, then:

```bash
./run.sh render      # inspect the training manifest without deploying
./run.sh            # data if needed -> preflight -> image/push -> secrets/PVC -> job/logs
```

Smoke mode uses 2 questions × 4 responses, 1,024 response tokens and 3 steps, with
checkpoints at 2 and 3. It retains the configured model and LoRA settings: it does not
silently switch to Qwen3-0.6B. It writes to `<RUN_NAME>-smoke`.
Truncated outputs score zero, so a successful smoke run can still prove no learning.

After inspecting that run, set `SMOKE_TEST=false` in `.env`. Inspect status before
cleaning up the completed smoke Job:

```bash
./run.sh status
./run.sh clean       # deletes this configured training/export Job, not PVCs or Secrets
./run.sh secrets
./run.sh submit
./run.sh logs
```

For a fresh experiment, choose a new `RUN_NAME`. A completed run exits without training
again. Interrupted runs resume from their latest checkpoint only when data, recipe,
source and recorded dependency versions match. A process lock prevents two drivers
from writing to the same run directory. The namespace's SFT PVC may itself constrain
concurrent scheduling because it is ReadWriteOnce.

To run on a provisioned NVIDIA machine without Kubernetes, use the same built image
with `/data` and `/grpo` mounted, or an equivalent installed environment:

```bash
./run.sh train
# Extra Hydra overrides are appended last:
./run.sh train rllm.rollout.n=4
```

The container/runtime must have the pinned training stack; the lightweight CPU test
environment is not sufficient. The launcher uses `python3` on PATH.

## Monitoring and checkpoints

Defaults: 8 questions × 8 responses per batch, 4 questions per optimizer mini-batch,
200-batch cap, one epoch, validation/checkpoint every 20 batches, validation before
training. Final validation runs when `TEST_FREQ > 0`; smoke mode disables it.
The effective training length is limited by both epoch count and batch cap.

WandB uses `WANDB_PROJECT=context-comp-grpo` and the run name, including the smoke suffix.
The persistent manifest supplies `WANDB_RUN_ID` and `WANDB_RESUME=allow`.
An empty `REPORT_TO` selects console only; `WANDB_MODE=offline` records locally.
After a crash, metrics beyond the last checkpoint may already exist in WandB; replayed
steps can be skipped by WandB's monotonic-step rule. Checkpoint state remains authoritative.

Watch these metrics (the trajectory/role name is normally `policy`):

| Metric | What to watch |
|---|---|
| `batch/reward`, `reward/policy/mean` | Overall reward trend |
| `batch/answer_correct` | Accuracy on checkable answers |
| `batch/label_alignment`, `batch/tag_rate` | Label order/mix and actual tagging coverage |
| `batch/truncated`, `batch/label_retries` | Wasted generation and format rejections |
| `batch/policy/fractions/effective` | Groups with differing rewards that provide GRPO signal |
| `health/zero_signal_steps` | Consecutive batches without informative groups |
| `actor/pg_loss`, `actor/grad_norm`, `actor/ppo_kl` | Optimization health |
| `timing_s/step`, `perf/max_memory_allocated_gb` | Throughput and memory |

The run directory contains:

```text
run_manifest.json       recipe/data/code/dependencies, stable WandB ID
run_status.json         running, failed or completed
episodes/               rLLM rollout traces
hydra/                  composed launch configuration
checkpoints/            model + optimizer + RNG/scheduler + dataloader state
export/global_step_N/   merged Hugging Face model after export
```

`KEEP_CHECKPOINTS=2` retains two actor checkpoints. Monitor PVC free space: checkpoints,
export staging, the merged model and episode logs coexist. Retention is not a backup.

## Export

```bash
./run.sh export
```

After training stops, the CPU Job exports the latest checkpoint from the run selected by `RUN_NAME` and
`SMOKE_TEST`. Set `EXPORT_STEP` before refreshing secrets to select a retained step.
The recorded training model is used as the LoRA base. Existing exports are refused.
Failed staging is retained and its exact path printed; successful staging is removed.
Export holds the run lock so a concurrent trainer cannot delete checkpoint shards.

Native equivalent:

```bash
python GRPO/src/export_policy.py --checkpoint-dir /path/to/run/checkpoints
```

## Verification limits

CPU tests cover reward equivalence, nonzero/zero signal, data leakage, configuration,
resume identity, checkpoint completeness, final-save behavior, export structure, scoped
secrets and Job failure reporting. The final-save tests use contract fixtures.
`tests/check_pinned_config.py` additionally composes and synchronizes the actual launcher
overrides against unpacked rLLM `3b40c37` and verl `v0.8.0` source trees.

The modified branch still needs a real 8B GPU smoke test, a short run with nonzero
advantages, an interrupted-run/resume check, and a real checkpoint merge/load check.
No improved model quality or GPU memory fit is claimed from CPU tests. The inherited
reward checks label mix/order, not the semantic correctness of each reasoning label;
format retries also discard bad samples instead of directly training against them.
