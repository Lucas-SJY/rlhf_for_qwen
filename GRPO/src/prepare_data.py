#!/usr/bin/env python3
"""Build the GRPO task set from the bespoke-v2 span annotations.

Each output line is one task dict, which rLLM's unified trainer hands to the workflow
unchanged:

    id           source sample id; rLLM also uses it to group the GRPO samples of a question
    data_source  "bespoke_labeled_cot" (groups validation metrics)
    question     the bare question, exactly as in SFT
    answer       reference final answer (the \\boxed{} content)
    ref_labels   the annotated label of every span of the reference trace, in order

The question text already starts with "Return your final response within \\boxed{}.",
and no label instruction is added, so the prompt matches the SFT prompt.

The validation split reuses the ids of the SFT held-out set (``--split-from``), so the
validation questions were never trained on in either stage. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

LABELS = {
    "planning_next_step",
    "restating_problem",
    "recalling_knowledge",
    "logical_deduction",
    "reflecting",
    "verifying",
    "correcting_itself",
    "concluding",
}
DATA_SOURCE = "bespoke_labeled_cot"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="../train/bespoke-v2", help="directory holding sample_*.json")
    p.add_argument("--split-from", default="../train/data_labeled_2",
                   help="SFT dataset dir whose validation.jsonl ids become the validation set; "
                        "empty falls back to a seeded random split")
    p.add_argument("--output-dir", default="GRPO/data")
    p.add_argument("--val-ratio", type=float, default=0.02, help="only used without --split-from")
    p.add_argument("--max-question-chars", type=int, default=6000,
                   help="drop questions longer than this; the workflow also drops prompts over "
                        "MAX_PROMPT_LENGTH tokens")
    p.add_argument("--max-samples", type=int, default=0, help="only take the first N files, 0 means all")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_val_ids(split_dir: str) -> set[str] | None:
    if not split_dir:
        return None
    path = Path(split_dir) / "validation.jsonl"
    if not path.is_file():
        raise SystemExit(f"--split-from: {path} not found (pass --split-from '' for a random split)")
    return {json.loads(line)["id"] for line in path.open() if line.strip()}


def build_task(sample: dict, max_question_chars: int) -> dict | None:
    question = (sample.get("question") or "").strip()
    answer = str(sample.get("answer") or "").strip()
    spans = sample.get("spans") or sample.get("steps") or []
    ref_labels = [s.get("label") for s in spans if (s.get("text") or s.get("content") or "").strip()]
    ref_labels = [label for label in ref_labels if label in LABELS]
    if not question or not answer or not ref_labels or len(question) > max_question_chars:
        return None
    return {
        "id": str(sample.get("id") or sample.get("example_id")),
        "data_source": DATA_SOURCE,
        "question": question,
        "answer": answer,
        "ref_labels": ref_labels,
    }


def main() -> None:
    args = parse_args()
    files = sorted(Path(args.input_dir).glob("sample_*.json"))
    if not files:
        raise SystemExit(f"no sample_*.json found under {args.input_dir}")
    if args.max_samples:
        files = files[: args.max_samples]

    tasks, skipped = [], 0
    for path in files:
        try:
            task = build_task(json.loads(path.read_text()), args.max_question_chars)
        except json.JSONDecodeError:
            task = None
        if task is None:
            skipped += 1
            continue
        tasks.append(task)

    val_ids = load_val_ids(args.split_from)
    if val_ids is not None:
        val = [t for t in tasks if t["id"] in val_ids]
        train = [t for t in tasks if t["id"] not in val_ids]
    else:
        random.Random(args.seed).shuffle(tasks)
        n_val = max(1, int(len(tasks) * args.val_ratio))
        val, train = tasks[:n_val], tasks[n_val:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("validation", val)):
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for task in split:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

    label_counts = Counter(label for t in train for label in t["ref_labels"])
    total = sum(label_counts.values())
    print(f"input files : {len(files)} ({skipped} skipped)")
    print(f"train / val : {len(train)} / {len(val)}  -> {out_dir}/"
          f"  (val ids {'from ' + args.split_from if val_ids is not None else 'random'})")
    print("reference label shares (train):")
    for label, count in label_counts.most_common():
        print(f"  {label:22s} {count / total:6.1%}")


if __name__ == "__main__":
    main()
