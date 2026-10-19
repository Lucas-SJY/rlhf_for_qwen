#!/usr/bin/env python3
"""Build a GRPO task set from Omni-MATH (KbsdJames/Omni-MATH on Hugging Face).

Omni-MATH has 4,428 olympiad problems in one `test` split, each with `problem`,
`answer`, `solution`, `domain`, `difficulty` (1-10) and `source`. This script turns them
into tasks in the format of prepare_grpo_try.py:

    id           omni_<row index>, the row's position in the source split
    data_source  "omni_math" (groups validation metrics: val/omni_math/...)
    prompt       [{"role": "user", "content": question}]
    question     "Return your final response within \\boxed{}. " + the problem, the
                 instruction every SFT and rest_grpo question starts with
    answer       the source's `answer`, unwrapped from display-math delimiters and an
                 outer \\boxed{} that some rows carry (`\\[ \\boxed{2, 3, 7} \\]` -> `2, 3, 7`)

Dropped, and counted in the report:
  - the first --exclude-first (20) rows: the slice ../../evaluate scores Omni-MATH on
    (EVAL_LIMIT=20), so training never sees an evaluation problem;
  - answers the reward cannot check (labelcot.reward.answer_is_checkable: empty, longer
    than 40 characters, \\text / \\mbox, prose), mostly proofs and yes/no conclusions;
  - several answers: the answer lists several values at its top level (`2, 3`,
    `(2, 1), (3, 1)`; a single tuple or interval such as `(2, 4, 6)` is one value, as is
    `3,840`), or the solution boxes more than one distinct value (as in
    prepare_rest_grpo.py). math_verify accepts one element of such a list as the whole
    answer, so the reward would pay for a partial answer.

The rest is split at random (seeded) by problem, --val-ratio (10 %) for validation. The
written files are read back and checked with prepare_grpo_try's task check.

The source is read with the `datasets` library (the GRPO image has it), or from a local
JSON-lines export with --input. From the repository root:

    docker run --rm -v "$PWD/GRPO:/w" -w /w/src --entrypoint python3 <GRPO image> \\
        prepare_omni_math.py --output-dir ../data/omni_math
    python3 GRPO/src/prepare_omni_math.py --input omni.jsonl   # offline, stdlib only
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from labelcot.reward import answer_is_checkable
from prepare_grpo_try import check_outputs, extract_boxed, report
from prepare_rest_grpo import boxed_answers

INSTRUCTION = "Return your final response within \\boxed{}."
KINDS = ("checkable", "evaluation slice", "answer not checkable", "several answers")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="KbsdJames/Omni-MATH")
    p.add_argument("--split", default="test")
    p.add_argument("--input", default="", help="a JSON-lines export of the split instead of the Hub")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "omni_math"))
    p.add_argument("--data-source", default="omni_math")
    p.add_argument("--exclude-first", type=int, default=20, help="rows held out for evaluation")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_rows(args) -> list[dict]:
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    from datasets import load_dataset

    return [dict(r) for r in load_dataset(args.dataset, split=args.split)]


def clean_answer(raw) -> str:
    """The answer without \\[ \\], $ delimiters and an outer \\boxed{} around all of it."""
    answer = str(raw or "").strip()
    for left, right in (("\\[", "\\]"), ("$$", "$$"), ("$", "$"), ("\\(", "\\)")):
        if answer.startswith(left) and answer.endswith(right) and len(answer) > len(left) + len(right):
            answer = answer[len(left):-len(right)].strip()
    if answer.startswith("\\boxed"):
        inner = extract_boxed(answer)
        if inner is not None and answer.rstrip().endswith("}") and answer.count("\\boxed") == 1:
            answer = inner.strip()
    return answer


def top_level_values(answer: str) -> list[str]:
    """The answer split on commas outside (), [] and {}; a thousands separator stays."""
    parts, depth, current = [], 0, ""
    for i, ch in enumerate(answer):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0 and not (current[-1:].isdigit() and answer[i + 1:i + 4].isdigit()
                                             and not answer[i + 4:i + 5].isdigit()):
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    parts.append(current.strip())
    return [x for x in parts if x]


def classify(index: int, row: dict, exclude_first: int) -> str:
    if index < exclude_first:
        return "evaluation slice"
    answer = clean_answer(row.get("answer"))
    if not answer_is_checkable(answer):
        return "answer not checkable"
    if ("\\boxed" in answer or len(top_level_values(answer)) > 1
            or len(set(boxed_answers(row.get("solution") or ""))) > 1):
        return "several answers"
    return "checkable"


def build_task(index: int, row: dict, data_source: str) -> dict:
    question = f"{INSTRUCTION} {row['problem'].strip()}"
    return {
        "id": f"omni_{index:04d}",
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "question": question,
        "answer": clean_answer(row["answer"]),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args)
    kinds = [classify(i, r, args.exclude_first) for i, r in enumerate(rows)]
    tasks = [build_task(i, r, args.data_source) for i, (r, k) in enumerate(zip(rows, kinds)) if k == "checkable"]
    if not tasks:
        raise SystemExit("no checkable problems")

    shuffled = list(tasks)
    random.Random(args.seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_ratio))
    splits = {"train": sorted(shuffled[n_val:], key=lambda t: t["id"]),
              "validation": sorted(shuffled[:n_val], key=lambda t: t["id"])}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for task in split:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

    counts, errors = check_outputs(out_dir)
    if errors:
        report("error", errors)
        raise SystemExit("the written files failed the format check")

    by_kind = Counter(kinds)
    print(f"source        : {len(rows)} problems ({args.input or f'{args.dataset} [{args.split}]'})")
    print("kept / dropped: " + ", ".join(f"{by_kind[k]} {k}" for k in KINDS if by_kind[k]))
    print(f"split         : {counts['train']} train / {counts['validation']} validation (seed {args.seed}) -> {out_dir}/")


if __name__ == "__main__":
    main()
