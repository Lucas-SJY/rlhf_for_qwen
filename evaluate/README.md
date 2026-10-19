# MATH-500 evaluation of GRPO checkpoints

The evaluation of [`../../train/evaluate`](../../train/evaluate/README.md), applied to the
GRPO checkpoints so their scores sit next to the SFT v3 baseline. Generation and the
string scorer are the same in every respect that affects the numbers. On top, every run
is also scored with Hugging Face Math-Verify and the LLM judge (see Scoring).

## What is the same

| | here and in `../../train/evaluate` |
|---|---|
| scripts | `eval_math500.py` and `judge_answers.py` as there, plus support for datasets without MATH-500's id, subject and level fields; `math_verify_answers.py` is new |
| image | the SFT training image (`context-comp-sft:latest`: torch 2.9.1, transformers 5.15) |
| dataset | `HuggingFaceH4/MATH-500`, split `test`, all 500 problems |
| prompt | `Return your final response within \boxed{}.` + problem, chat template, thinking on |
| decoding | greedy (`temperature 0`), batch 6, at most 12,288 new tokens, seed 42 |
| string scorer | only the answer after `</think>`, its last `\boxed{}`, LaTeX normalisation + numeric fallback (`accuracy`, `accuracy_string`) |
| LLM judge | `qwen3-small` on NRP's gateway, on the rows the string scorer rejected (`accuracy_judge`) |
| hardware | one RTX A6000 (default), 32Gi |

Batched greedy generation is not invariant to batch size or GPU type, so those are kept
too by default. `EVAL_GPU_TYPE=l40` runs on an L40 / L40S (also 48 GB) when no A6000 is
free; the scores can then differ slightly from the baseline's for numerical reasons. The
Job requests one CPU core by default (`EVAL_CPU=1000` millicores; `../train` used 4) with
a limit of 1.2 x the request, NRP's maximum: generation is GPU-bound, and 48 GB nodes
rarely have several cores left unrequested. A lower `EVAL_CPU` fits more nodes.

The SFT v3 baseline from that setup: `accuracy_string` **0.814**, `accuracy_math_verify`
0.852, `accuracy_judge` 0.846, 64 truncated, mean 3,764 completion tokens
(`../../train/evaluate/results/qwen3-8b-sft-v3`; its Math-Verify score was added here, in
`results/qwen3-8b-sft-v3/`).

## What is different

- **Model**: a GRPO export on the RL PVC (`../run.sh export` writes
  `/grpo/runs/<run>/export/global_step_<N>`), chosen with `EVAL_MODEL`. The default is
  the export of `RUN_NAME` (from `../.env`) at its final step.
- **Results**: `/grpo/eval/<run>-global_step_<N>/` on the RL PVC, fetched to
  `results/<run>-global_step_<N>/` here (gitignored, as in `../../train/evaluate/results`).
- **Names**: Job `qwen-grpo-eval`, ConfigMap `grpo-eval-code`, Secret `grpo-env`, so
  nothing collides with `../train`'s `qwen-eval` / `eval-code` in the same namespace.

## Run

```bash
# 1. generate + string-score on the cluster (~26-28 h for 500 problems on an A6000)
EVAL_MODEL=/grpo/runs/<run>/export/global_step_<N> ./evaluate/run_eval.sh submit
# 2. when the Job is done: copy the results here and score them (math_verify, LLM judge)
EVAL_MODEL=/grpo/runs/<run>/export/global_step_<N> ./evaluate/run_eval.sh fetch
#    -> evaluate/results/<run>-global_step_<N>/
```

`./evaluate/run_eval.sh` without a subcommand does code, submit, logs and fetch in one go
(it then follows the log for the whole run). `score` repeats only the scoring of the
fetched results.

- `EVAL_LIMIT=20 ./evaluate/run_eval.sh` is a smoke test; `never_closed_think` and
  `no_boxed_answer` should be near zero before a full run.
- `EVAL_DATASET` / `EVAL_SPLIT` evaluate on another Hugging Face dataset whose rows have
  `problem` and `answer`, with the same prompt, decoding and scorers; results go to
  `results/<dataset name>/<run>-global_step_<N>/`. A row without MATH-500's `unique_id` is
  named `<split>/<index>`; `subject` and `level` are optional. For example the first 20
  BeyondAIME problems (integer answers):
  `EVAL_DATASET=ByteDance-Seed/BeyondAIME EVAL_LIMIT=20 ./evaluate/run_eval.sh submit`,
  then `fetch` with the same variables. `EVAL_DATASET_CONFIG` picks a config (subset).
- `EVAL_SUITE` runs several datasets in one Job, loading the model once: entries
  `dataset|config|split` separated by `;`. Each goes to
  `results/<dataset name>[-<config>]/<EVAL_RUN_NAME>/`; `fetch` and `score` handle every
  entry, and a dataset that fails does not stop the others. The SFT v3 run on seven
  olympiad sets (first 20 problems each):

  ```bash
  export EVAL_MODEL=/grpo/hf/hub/models--Lucas-SJY--qwen3-8b-sft-bespoke/snapshots/e24173c027eefa9e976908041d6f7858cb1e8634
  export EVAL_RUN_NAME=qwen3-8b-sft-v3 EVAL_LIMIT=20
  export EVAL_SUITE="ByteDance-Seed/BeyondAIME||test;KbsdJames/Omni-MATH||test;RUC-AIBOX/OlymMATH|en-hard|test;MathArena/hmmt_feb_2025||train;MathArena/hmmt_feb_2026||train;math-ai/aime25||test;MathArena/aime_2026||train"
  ./evaluate/run_eval.sh submit      # later, with the same variables: fetch
  ```
- `EVAL_TEMPERATURE` / `EVAL_TOP_P` / `EVAL_TOP_K` change the decoding (default greedy, as
  the baseline). Greedy decoding sends Qwen3's thinking mode into repetition loops (40 of
  the 104 cut-off answers of the run above); Qwen3 recommends temperature 0.6, top-p 0.95,
  top-k 20, e.g. with `EVAL_RUN_NAME=qwen3-8b-sft-v3-t0.6` so the results sit beside the
  greedy ones. One sample per problem, so a 20-problem score is noisy.
- `EVAL_RUN_NAME` overrides the result directory name.
- The other subcommands are those of `../../train/evaluate/run_eval.sh`: `code`, `submit`,
  `logs`, `status`, `fetch`, `clean`; plus `score`.
- `fetch` copies only `EVAL_RUN_NAME`'s directory, so other runs' scored files stay.
- The RL PVC is ReadWriteOnce: the Job cannot start on another node while a training
  pod holds it, so evaluate between training runs.

## Scoring

Every fetched run gets three accuracies in `summary.json`:

| field | scorer | runs |
|---|---|---|
| `accuracy`, `accuracy_string` | `eval_math500.py`'s string scorer, unchanged | in the Job |
| `accuracy_math_verify` | Hugging Face Math-Verify 0.9.0 (`math_verify_answers.py`): the text after `</think>` against `\boxed{gold}`, the check the GRPO reward uses, alone | locally |
| `accuracy_judge` | the LLM judge (`judge_answers.py`), on the rows the string scorer rejected | locally |

- The local steps run in docker, in the GRPO image (`IMAGE` in `../.env`; `SCORE_IMAGE`
  overrides it), which has math-verify and datasets; the local Python need not.
- The judge needs `OPENAI_BASE_URL` and `OPENAI_API_KEY` (NRP's LLM gateway). They come
  from the environment or `../.env`, else from `../../train/.env`, of which only those two
  keys are read. `EVAL_JUDGE_MODEL` changes the judge (default `qwen3-small`, the
  baseline's).
- Math-Verify is lenient with answers that are several values: it accepted `1` for
  `-2,1` and `3 - 2\sqrt{2}` for `3 \pm 2\sqrt{2}` (V1, where the judge said wrong).
  `math_verify_flipped_to_correct` counts the rows it accepts and the string scorer does
  not; read those before trusting a gain.

## Rollout outcomes of a training run

`rollout_stats.sh` counts, for every training step and question, how many of its
`GROUP_SIZE` rollouts were correct, how many were cut off at the token limit and whether
GRPO got a gradient from the group (the finished rollouts' rewards differ). It reads
rLLM's episode logs (`/grpo/runs/<run>/episodes/episodes/train_step_<N>_epoch_0/`, one
JSON file per rollout with the full response, ~1 GB for a 200-step run) where they are:
a small `python:3.12-alpine` pod mounts the RL PVC read-only and runs `rollout_stats.py`,
and only the summary, one row per step and question, comes back.

```bash
./evaluate/rollout_stats.sh                          # RUN_NAME from ../.env
STATS_RUN_NAME=<run> ./evaluate/rollout_stats.sh     # another run
STATS_MODE=val ./evaluate/rollout_stats.sh           # the validation rollouts
# -> eval-results/rollout-stats/<run>-<mode>.json
```

- A cut-off rollout counts as wrong, as in `batch/answer_correct`.
- If another pod (training, an answer Job) holds the ReadWriteOnce PVC, the pod is
  scheduled onto that node, where the PVC can be mounted twice.
- Reading runs at the PVC's speed: ~9 min for V1's 25,600 rollouts (~1 GB).

## Output

As in `../../train/evaluate`: `summary.json` (accuracy, token statistics, breakdown by
level and subject), `predictions.jsonl` (one row per problem with the full completion)
and `predictions_judged.jsonl` (the same rows with `judge` and `judge_raw`). Scoring adds
a `math_verify` field to every row and `accuracy_math_verify`,
`math_verify_flipped_to_correct` / `_to_wrong`, the judge fields and `accuracy_string`
to `summary.json`.
