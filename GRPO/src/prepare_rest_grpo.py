#!/usr/bin/env python3
"""Build a GRPO task set from the Bespoke-Stratos questions the SFT data did not use.

Bespoke-Stratos-17k has 16,710 samples (sample_000000 .. sample_016709). The SFT set
../train/data_labeled_2 holds 5,144 of them; the harbor dataset bespoke-stratos-rest
holds exactly the other 11,566, one directory per sample with
<id>/environment/trajectory.json = {id, model, question, thought_trace, segments,
solution}. This script turns those into tasks in the same format as
prepare_grpo_try.py:

    id           source sample id; rLLM also uses it to group the GRPO samples of a question
    data_source  "rest_grpo" (groups validation metrics: val/rest_grpo/...)
    prompt       [{"role": "user", "content": question}], exactly as in SFT
    question     the bare question
    answer       the \\boxed{} content of the upstream solution; "" when there is none,
                 e.g. for coding questions, whose reward then has only the label term

The reference reasoning (thought_trace, segments) is not written. Ids listed in
--exclude-from (default: the SFT set) are dropped even if they appear in the source, so
the GRPO data can never overlap the SFT data. The split is a seeded random one by
question, --val-ratio (10 %) of the questions for validation.

Only questions with one answer the reward can check (labelcot.reward.answer_is_checkable)
stay in --output-dir. The others are written to --no-answer-dir instead, unchanged and in
the same train/validation split (the split is made before they are separated, so the ids
per split do not depend on it):
  - no answer: the coding questions; the reward would score the label term alone;
  - answer not checkable: prose or proof answers; likewise label term alone;
  - several answers: the solution boxes more than one distinct value (several roots,
    multi-part questions), so the last \boxed{} kept as the answer is incomplete and a
    complete model answer could be scored wrong.
--no-answer-dir '' keeps everything in --output-dir.

Format check. Every trajectory.json must be a JSON object with a string id, question and
solution and a non-empty question; a file that fails is reported and skipped, or stops
the run with --strict. An id that does not match its directory name only produces a
warning. The written files are read back and checked with prepare_grpo_try's task check
(also ids across the two splits); any error fails the run. --check-only runs just that
check. Stdlib only.

Usage (from the repository root):
    python3 GRPO/src/prepare_rest_grpo.py
    python3 GRPO/src/prepare_rest_grpo.py --strict
    python3 GRPO/src/prepare_rest_grpo.py --check-only
    python3 GRPO/src/prepare_rest_grpo.py --no-answer-dir ''   # keep unanswerable questions
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from labelcot.reward import answer_is_checkable
from prepare_grpo_try import _type_errors, check_outputs, extract_boxed, report

SOURCE_FIELDS = {"id": str, "question": str, "solution": str}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", default="../jianhong_harbor/harbor/datasets/bespoke-stratos-rest",
                   help="harbor dataset root holding <id>/environment/trajectory.json")
    p.add_argument("--exclude-from", default="../train/data_labeled_2",
                   help="directory of *.jsonl files whose 'id's are left out (the SFT set); empty: none")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--output-dir", default="GRPO/data/rest_grpo")
    p.add_argument("--no-answer-dir", default="no_answer",
                   help="where questions without a checkable answer go; empty: keep them in --output-dir")
    p.add_argument("--data-source", default="rest_grpo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict", action="store_true",
                   help="stop when a source file fails the format check instead of skipping it")
    p.add_argument("--check-only", action="store_true",
                   help="only run the format check on the existing files in --output-dir")
    return p.parse_args()


def load_excluded_ids(exclude_dir: str) -> set[str]:
    if not exclude_dir:
        return set()
    files = sorted(Path(exclude_dir).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"--exclude-from: no *.jsonl under {exclude_dir} (pass --exclude-from '' to exclude nothing)")
    ids = set()
    for path in files:
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"--exclude-from: {path}:{lineno} is not valid JSON: {e.msg}")
                if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                    raise SystemExit(f"--exclude-from: {path}:{lineno} has no string 'id'")
                ids.add(row["id"])
    return ids


def check_source(record: object, where: str, dir_name: str) -> tuple[list[str], list[str]]:
    """Check one parsed trajectory.json. Returns (errors, warnings)."""
    if not isinstance(record, dict):
        return [f"{where}: top level is {type(record).__name__}, expected an object"], []
    errors = _type_errors(record, SOURCE_FIELDS, where)
    if errors:
        return errors, []
    warnings = []
    if not record["question"].strip():
        errors.append(f"{where}.question: empty")
    if record["id"] != dir_name:
        warnings.append(f"{where}.id: {record['id']!r} does not match the directory name {dir_name!r}")
    return errors, warnings


ANSWER_KINDS = ("checkable", "answer not checkable", "several answers", "no answer")


def boxed_answers(text: str) -> list[str]:
    """The contents of every \\boxed{...} in text, in order, matching nested braces."""
    answers = []
    start = text.find("\\boxed")
    while start >= 0:
        # extract_boxed reads the last \boxed of its input, so cut the text before the next.
        nxt = text.find("\\boxed", start + 1)
        content = extract_boxed(text[start:nxt] if nxt >= 0 else text[start:])
        if content is not None:
            answers.append(content.strip())
        start = nxt
    return answers


def answer_kind(task: dict, several: frozenset[str] | set[str] = frozenset()) -> str:
    if not task["answer"]:
        return "no answer"
    if not answer_is_checkable(task["answer"]):
        return "answer not checkable"
    return "several answers" if task["id"] in several else "checkable"


def overlap_errors(dirs: list[Path]) -> list[str]:
    """Ids that appear in more than one of the output directories."""
    owner: dict[str, Path] = {}
    errors = []
    for d in dirs:
        for name in ("train", "validation"):
            path = d / f"{name}.jsonl"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    task_id = json.loads(line).get("id") if line.strip() else None
                    if task_id in owner and owner[task_id] != d:
                        errors.append(f"id {task_id} is in both {owner[task_id]}/ and {d}/")
                    owner.setdefault(task_id, d)
    return errors


def build_task(record: dict, sample_id: str, data_source: str) -> dict:
    question = record["question"].strip()
    return {
        "id": sample_id,
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "question": question,
        "answer": (extract_boxed(record["solution"]) or "").strip(),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    out_dirs = [out_dir] + ([Path(args.no_answer_dir)] if args.no_answer_dir else [])

    if args.check_only:
        errors = []
        for d in out_dirs:
            counts, dir_errors = check_outputs(d)
            errors.extend(dir_errors)
            status = "OK" if not dir_errors else f"{len(dir_errors)} error(s)"
            print(f"format check  : {d}/ {status} "
                  f"({counts['train']} train / {counts['validation']} validation tasks valid)")
        errors.extend(overlap_errors(out_dirs))
        report("error", errors)
        raise SystemExit(1 if errors else 0)

    source = Path(args.source_dir)
    sample_dirs = sorted(p for p in source.glob("sample_*") if p.is_dir())
    if not sample_dirs:
        raise SystemExit(f"no sample_* directories under {source}")
    excluded_ids = load_excluded_ids(args.exclude_from)

    tasks, skipped, excluded, input_errors, warnings = [], [], [], [], []
    several: set[str] = set()  # ids whose solution boxes more than one distinct value
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        if sample_id in excluded_ids:
            excluded.append(sample_id)
            continue
        path = sample_dir / "environment" / "trajectory.json"
        where = f"{sample_id}/environment/trajectory.json"
        if not path.is_file():
            errors = [f"{where}: missing"]
        else:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                errors = [f"{where}: invalid JSON at line {e.lineno} column {e.colno}: {e.msg}"]
            except UnicodeDecodeError as e:
                errors = [f"{where}: not UTF-8 ({e.reason} at byte {e.start})"]
            else:
                errors, record_warnings = check_source(record, where, sample_id)
                warnings.extend(record_warnings)
        if errors:
            input_errors.extend(errors)
            skipped.append(sample_id)
            continue
        tasks.append(build_task(record, sample_id, args.data_source))
        if len(set(boxed_answers(record["solution"]))) > 1:
            several.add(sample_id)

    report("warning", warnings)
    report("error", input_errors)
    if skipped and args.strict:
        raise SystemExit(f"{len(skipped)} source file(s) failed the format check; stopping (--strict)")
    if not tasks:
        raise SystemExit("no source file passed the format check")

    shuffled = sorted(tasks, key=lambda t: t["id"])
    random.Random(args.seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_ratio))
    splits = {"train": shuffled[n_val:], "validation": shuffled[:n_val]}

    # Separate the questions without a checkable answer after the split.
    outputs = {d: {name: [] for name in splits} for d in out_dirs}
    for name, split in splits.items():
        for task in sorted(split, key=lambda t: t["id"]):
            keep = not args.no_answer_dir or answer_kind(task, several) == "checkable"
            outputs[out_dir if keep else out_dirs[1]][name].append(task)

    for d, files in outputs.items():
        d.mkdir(parents=True, exist_ok=True)
        for name, split in files.items():
            with (d / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for task in split:
                    f.write(json.dumps(task, ensure_ascii=False) + "\n")

    # Read the files back: what rLLM will load is what gets checked.
    output_errors = []
    for d in out_dirs:
        # An empty no-answer set is fine; only the main set must have training tasks.
        output_errors.extend(e for e in check_outputs(d)[1] if d == out_dir or not e.endswith(": no tasks"))
    output_errors.extend(overlap_errors(out_dirs))
    if output_errors:
        report("error", output_errors)
        raise SystemExit("the written files failed the format check")

    print(f"source        : {len(sample_dirs)} samples in {source}")
    print(f"excluded      : {len(excluded)} (ids in {args.exclude_from or 'nothing'})")
    print(f"format check  : {len(sample_dirs) - len(excluded) - len(skipped)} source files OK"
          + (f", {len(skipped)} skipped" if skipped else "") + f", {len(warnings)} warning(s); output OK")
    print(f"split         : {len(splits['train'])} train / {len(splits['validation'])} validation (seed {args.seed})")
    for d, files in outputs.items():
        kinds = {name: Counter(answer_kind(t, several) for t in split) for name, split in files.items()}
        print(f"{str(d) + '/':30s}: {len(files['train'])} train / {len(files['validation'])} validation")
        for name in files:
            print(f"  {name:10s}: " + ", ".join(f"{kinds[name][k]} {k}" for k in ANSWER_KINDS if kinds[name][k]))


if __name__ == "__main__":
    main()
