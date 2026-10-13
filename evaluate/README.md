# MATH-500 evaluation of GRPO checkpoints

The evaluation of [`../../train/evaluate`](../../train/evaluate/README.md), applied to the
GRPO checkpoints so their scores sit next to the SFT v3 baseline. The method is the same
in every respect that affects the numbers; only where the model and the results live is
different.

## What is the same

| | here and in `../../train/evaluate` |
|---|---|
| scripts | `eval_math500.py` and `judge_answers.py` are byte-identical copies |
| image | the SFT training image (`context-comp-sft:latest`: torch 2.9.1, transformers 5.15) |
| dataset | `HuggingFaceH4/MATH-500`, split `test`, all 500 problems |
| prompt | `Return your final response within \boxed{}.` + problem, chat template, thinking on |
| decoding | greedy (`temperature 0`), batch 6, at most 12,288 new tokens, seed 42 |
| scoring | only the answer after `</think>`, LaTeX normalisation + numeric fallback; then the LLM judge on the rows the string scorer rejected |
| hardware | one RTX A6000, 4 CPU, 32Gi |

Batched greedy generation is not invariant to batch size or GPU type, so those are kept
too. The SFT v3 baseline from that setup: `accuracy_string` **0.814**, `accuracy_judge`
0.846, 64 truncated, mean 3,764 completion tokens
(`../../train/evaluate/results/qwen3-8b-sft-v3`).

## What is different

- **Model**: a GRPO export on the RL PVC (`../run.sh export` writes
  `/grpo/runs/<run>/export/global_step_<N>`), chosen with `EVAL_MODEL`. The default is
  the export of `RUN_NAME` (from `../.env`) at its final step.
- **Results**: `/grpo/eval/<run>-global_step_<N>/` on the RL PVC, fetched to
  `../eval-results/` (gitignored).
- **Names**: Job `qwen-grpo-eval`, ConfigMap `grpo-eval-code`, Secret `grpo-env`, so
  nothing collides with `../train`'s `qwen-eval` / `eval-code` in the same namespace.

## Run

```bash
# 1. generate + string-score on the cluster (~26 h for 500 problems, as for the baseline)
EVAL_MODEL=/grpo/runs/qwen3-8b-v3-full-offload-2gpu-rest-grpo-v1/export/global_step_200 \
  ./evaluate/run_eval.sh
./evaluate/run_eval.sh fetch      # -> eval-results/<run>-global_step_<N>/

# 2. re-score the near-misses with the LLM judge, locally
python evaluate/judge_answers.py --run-dir eval-results/<run>-global_step_<N>
```

- `EVAL_LIMIT=20 ./evaluate/run_eval.sh` is a smoke test; `never_closed_think` and
  `no_boxed_answer` should be near zero before a full run.
- `EVAL_RUN_NAME` overrides the result directory name.
- The other subcommands are those of `../../train/evaluate/run_eval.sh`: `code`, `submit`,
  `logs`, `status`, `fetch`, `clean`.
- The RL PVC is ReadWriteOnce: the Job cannot start on another node while a training
  pod holds it, so evaluate between training runs.
- Stage 2 reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the environment or from
  `../.env` (which does not have them by default; `../../train/.env` does).

## Output

As in `../../train/evaluate`: `summary.json` (accuracy, token statistics, breakdown by
level and subject), `predictions.jsonl` (one row per problem with the full completion),
and after stage 2 `predictions_judged.jsonl` plus the judge fields in `summary.json`.
