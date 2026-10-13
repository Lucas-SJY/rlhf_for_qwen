#!/usr/bin/env python3
"""Answer a task file with one or more models and score every answer, for manual inspection.

Each question is answered --samples times by each model, with the validation sampling of
the training run (temperature 0.6, top-p 0.95, top-k 20, at most MAX_RESPONSE_LENGTH new
tokens), from the same prompt the trainer uses: the task's chat-format prompt rendered
with the model's own chat template. Answers are kept as generated: there is no
regeneration on invalid labels. Every answer is scored with labelcot.reward, as in
training: label_valid, answer_correct (only the final answer after </think> is judged),
and the reward, which a cut-off answer does not get.

Models default to the run's latest export (label "grpo", made by export_policy.py /
../run.sh export) and the SFT checkpoint the run started from (label "sft"), so the two
can be compared. Each model runs in its own subprocess, which frees its GPU memory before
the next one loads.

Output under --output-dir (default /grpo/answers/<RUN_NAME>):
    <label>/<id>.json  the question, the reference answer and every sample with its score
    summary.json       per model and question: correct / truncated / valid-label counts
    summary.md         the same as a table

Usage (inside the image; ../run.sh answer runs it as a GPU Job):
    python3 answer_tasks.py
    python3 answer_tasks.py --model grpo=/grpo/runs/x/export/global_step_200 --samples 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from labelcot.reward import invalid_labels, score_completion, split_completion
from prepare_grpo_try import check_jsonl

# Validation sampling of the training run (entrypoint.sh: rllm.rollout.val.*).
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def parse_args() -> argparse.Namespace:
    run_dir = Path(os.environ.get("OUTPUT_ROOT", "/grpo/runs")) / os.environ.get("RUN_NAME", "qwen3-8b-grpo-labels-v1")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=os.environ.get("ANSWER_TASKS_FILE", "/tasks/tasks.jsonl"))
    p.add_argument("--model", action="append", default=[],
                   help="LABEL=PATH, repeatable; default: grpo=<latest export> and sft=<the SFT checkpoint>")
    p.add_argument("--run-dir", default=str(run_dir), help="training run whose latest export is the grpo model")
    p.add_argument("--samples", type=int, default=int(os.environ.get("ANSWER_SAMPLES", "4")))
    p.add_argument("--max-prompt-tokens", type=int, default=int(os.environ.get("MAX_PROMPT_LENGTH", "2048")))
    p.add_argument("--max-response-tokens", type=int, default=int(os.environ.get("MAX_RESPONSE_LENGTH", "8192")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    p.add_argument("--output-dir", default="", help=f"default: /grpo/answers/{run_dir.name}")
    p.add_argument("--one", default="", help=argparse.SUPPRESS)  # internal: answer with this LABEL=PATH only
    return p.parse_args()


def parse_model(spec: str) -> tuple[str, str]:
    label, sep, path = spec.partition("=")
    if not sep or not label or not path:
        raise SystemExit(f"--model expects LABEL=PATH, got {spec!r}")
    return label, path


def default_models(run_dir: Path) -> list[tuple[str, str]]:
    """grpo = the latest export of run_dir, sft = the checkpoint the run started from."""
    tracker = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise SystemExit(f"no {tracker}: the run has no checkpoint to answer with")
    step = int(tracker.read_text().strip())
    export = run_dir / "export" / f"global_step_{step}"
    if not (export / "config.json").is_file():
        raise SystemExit(f"{export} is not exported yet; run ../run.sh export first")
    base = os.environ.get("POLICY_MODEL_PATH", "/data/runs/qwen3-8b-sft-v3")
    revision = os.environ.get("POLICY_MODEL_REVISION", "")
    if revision and not Path(base).is_dir():
        from huggingface_hub import snapshot_download

        base = snapshot_download(base, revision=revision)
    return [("grpo", str(export)), ("sft", base)]


def score_sample(text: str, finish_reason: str, response_tokens: int, task: dict) -> dict:
    """One generated answer with its reward breakdown, as training would score it."""
    truncated = finish_reason == "length"
    b = score_completion(text, task, truncated=truncated)
    return {
        "finish_reason": finish_reason,
        "response_tokens": response_tokens,
        "truncated": truncated,
        "closed_think": b.closed,
        "label_valid": b.label_valid,
        "invalid_tags": invalid_labels(text),
        "n_steps": b.n_steps,
        "tag_rate": round(b.tag_rate, 4),
        "label_counts": b.label_counts,
        "answer_checkable": b.answer_checkable,
        "answer_correct": b.answer_correct,
        # A cut-off answer is not scored in training (it is masked out of the batch).
        "reward": None if truncated else round(b.reward, 4),
        "final_answer": split_completion(text)[1],
        "response": text,
    }


def question_summary(record: dict) -> dict:
    samples = record["samples"]
    return {
        "id": record["id"],
        "reference_answer": record["reference_answer"],
        "samples": len(samples),
        "correct": sum(s["answer_correct"] for s in samples),
        "truncated": sum(s["truncated"] for s in samples),
        "label_valid": sum(s["label_valid"] for s in samples),
        "mean_response_tokens": round(sum(s["response_tokens"] for s in samples) / max(1, len(samples))),
    }


def summary_markdown(summary: dict[str, list[dict]]) -> str:
    """One table: a row per question, a column per model with correct / truncated / valid labels."""
    labels = list(summary)
    by_id: dict[str, dict[str, dict]] = {}
    for label, rows in summary.items():
        for row in rows:
            by_id.setdefault(row["id"], {})[label] = row
    lines = ["| question | reference answer | " + " | ".join(f"{label}: correct / cut off / valid labels / mean tokens"
                                                       for label in labels) + " |",
             "|---|---|" + "---|" * len(labels)]
    totals = {label: [0, 0, 0, 0] for label in labels}
    for task_id, cells in by_id.items():
        reference = next(iter(cells.values()))["reference_answer"].replace("|", "\\|")
        row = [task_id, f"`{reference}`"]
        for label in labels:
            c = cells.get(label)
            if c is None:
                row.append("–")
                continue
            row.append(f"{c['correct']}/{c['samples']} / {c['truncated']} / {c['label_valid']} / {c['mean_response_tokens']}")
            t = totals[label]
            t[0] += c["correct"]
            t[1] += c["samples"]
            t[2] += c["truncated"]
            t[3] += c["label_valid"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| **total** | | " + " | ".join(
        f"**{t[0]}/{t[1]}** / {t[2]} / {t[3]}" for t in totals.values()) + " |")
    return "\n".join(lines) + "\n"


def answer_with(label: str, model_path: str, tasks: list[dict], args: argparse.Namespace, out_dir: Path) -> None:
    """Generate and score every answer of one model; writes <out_dir>/<label>/<id>.json."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", seed=args.seed, gpu_memory_utilization=0.85,
              max_model_len=args.max_prompt_tokens + args.max_response_tokens)
    tokenizer = llm.get_tokenizer()
    prompts = []
    for task in tasks:
        messages = list(task.get("prompt") or [{"role": "user", "content": task["question"]}])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > args.max_prompt_tokens:
            raise SystemExit(f"{task['id']}: prompt has {len(ids)} tokens, more than {args.max_prompt_tokens}")
        prompts.append({"prompt_token_ids": ids})
    params = SamplingParams(n=args.samples, max_tokens=args.max_response_tokens, seed=args.seed, **SAMPLING)
    outputs = llm.generate(prompts, params)

    model_dir = out_dir / label
    model_dir.mkdir(parents=True, exist_ok=True)
    for task, output in zip(tasks, outputs, strict=True):
        record = {
            "id": task["id"],
            "question": task["question"],
            "reference_answer": task.get("answer", ""),
            "model": label,
            "model_path": model_path,
            "sampling": dict(SAMPLING, n=args.samples, max_tokens=args.max_response_tokens, seed=args.seed),
            "samples": [dict(index=i, **score_sample(c.text, c.finish_reason or "", len(c.token_ids), task))
                        for i, c in enumerate(output.outputs)],
        }
        (model_dir / f"{task['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(f"[answer] {label} {task['id']}: " + json.dumps(question_summary(record)), flush=True)


def main() -> None:
    args = parse_args()
    tasks, errors = check_jsonl(Path(args.tasks))
    if errors:
        raise SystemExit("task file failed the format check:\n" + "\n".join(errors))
    out_dir = Path(args.output_dir or f"/grpo/answers/{Path(args.run_dir).name}")

    if args.one:
        label, path = parse_model(args.one)
        answer_with(label, path, tasks, args, out_dir)
        return

    models = [parse_model(m) for m in args.model] or default_models(Path(args.run_dir))
    print(f"[answer] {len(tasks)} questions x {args.samples} samples from {args.tasks} -> {out_dir}", flush=True)
    for label, path in models:
        print(f"[answer] model {label}: {path}", flush=True)
        # A fresh process per model, so vLLM's GPU memory is released in between.
        cmd = [sys.executable, __file__, "--one", f"{label}={path}", "--tasks", args.tasks,
               "--samples", str(args.samples), "--max-prompt-tokens", str(args.max_prompt_tokens),
               "--max-response-tokens", str(args.max_response_tokens), "--seed", str(args.seed),
               "--output-dir", str(out_dir), "--run-dir", args.run_dir]
        subprocess.run(cmd, check=True)

    summary = {}
    for label, _ in models:
        summary[label] = [question_summary(json.loads((out_dir / label / f"{t['id']}.json").read_text())) for t in tasks]
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "summary.md").write_text(
        f"# Answers to {Path(args.tasks).name}\n\n"
        f"{len(tasks)} questions, {args.samples} samples per question and model; sampling "
        f"{json.dumps(SAMPLING)}, at most {args.max_response_tokens} new tokens. Models: "
        + ", ".join(f"`{label}` = `{path}`" for label, path in models) + ".\n\n"
        + summary_markdown(summary))
    print(f"[answer] done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
