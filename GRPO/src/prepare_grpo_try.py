#!/usr/bin/env python3
"""Build a GRPO task set from the span annotations in grpo_try/.

grpo_try/ holds sample_*.json files in the bespoke-v2 layout (question + labelled spans),
but without the reference answer. The answer is recovered by sample id from the upstream
harbor dataset (``--solutions-dir``), whose ``solution`` ends in \\boxed{}, the same way
../train/src/merge_answers.py backfilled bespoke-v2.

Each output line is one task dict in the format GRPO/src/train_grpo.py and the
LabeledCoTWorkflow read, plus the labelled reference trace:

    id           source sample id; rLLM also uses it to group the GRPO samples of a question
    data_source  "grpo_try" (groups validation metrics: val/grpo_try/...)
    question     the bare question, exactly as in SFT
    answer       reference final answer (the \\boxed{} content of the upstream solution);
                 "" when there is none, e.g. for coding questions. The reward then scores
                 only the label terms for that task.
    ref_labels   the annotated label of every span, in order (what the reward compares to)
    ref_cot      the reference reasoning with each span's label in front of its text, in
                 the SFT target format: "[label] text" blocks separated by blank lines
    solution     the upstream reference solution ("" if not found). The full SFT-style
                 target is "<think>\\n" + ref_cot + "\\n</think>\\n\\n" + solution.

The split is by trace. By default the validation traces are the ones in
../train/grpo_test.json, the held-out set of the step-level export of the same data, so
both views of grpo_try hold out the same questions.

Format check. Every input file must be valid JSON with a string id and question and a
list of spans, each an object with a string label and text (and integer start, end and
token_count when present). A file that fails is reported with its location and skipped,
or stops the run with --strict. A span with an unknown label or no text, or an id that
does not match the file name, only produces a warning (the span is dropped, as in
SFT). The written files are then read back line by line and checked against the task
format above; any error there fails the run. --check-only runs just that last check on
existing output. Stdlib only.

Usage (from the repository root):
    python3 GRPO/src/prepare_grpo_try.py
    python3 GRPO/src/prepare_grpo_try.py --strict
    python3 GRPO/src/prepare_grpo_try.py --check-only
    python3 GRPO/src/prepare_grpo_try.py --split-from '' --val-ratio 0.3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

LABELS = (
    "planning_next_step",
    "restating_problem",
    "recalling_knowledge",
    "logical_deduction",
    "reflecting",
    "verifying",
    "correcting_itself",
    "concluding",
)
_BLANK_LINES = re.compile(r"\n\s*\n")
_COT_BLOCK = re.compile(r"^\[([a-z_]+)\] \S")

# Expected JSON types, for the format check.
SAMPLE_FIELDS = {"id": str, "question": str, "spans": list}
SPAN_FIELDS = {"label": str, "text": str}
SPAN_INT_FIELDS = ("start", "end", "token_count")
TASK_FIELDS = {
    "id": str,
    "data_source": str,
    "question": str,
    "answer": str,
    "ref_labels": list,
    "ref_cot": str,
    "solution": str,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="grpo_try", help="directory holding sample_*.json")
    p.add_argument("--solutions-dir",
                   default="../jianhong_harbor/harbor/datasets/bespoke-stratos-rest",
                   help="upstream dataset root holding <id>/environment/trajectory.json with a "
                        "'solution' field; empty skips answer recovery")
    p.add_argument("--split-from", default="../train/grpo_test.json",
                   help="step-level test file whose trace ids become the validation set; "
                        "empty falls back to a seeded random split by trace")
    p.add_argument("--val-ratio", type=float, default=0.3, help="only used without --split-from")
    p.add_argument("--output-dir", default="GRPO/data/grpo_try")
    p.add_argument("--data-source", default="grpo_try")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict", action="store_true",
                   help="stop when an input file fails the format check instead of skipping it")
    p.add_argument("--check-only", action="store_true",
                   help="only run the format check on the existing files in --output-dir")
    return p.parse_args()


# ---------------------------------------------------------------------------
# format check
# ---------------------------------------------------------------------------
def _type_errors(obj: dict, fields: dict, where: str) -> list[str]:
    errors = []
    for key, expected in fields.items():
        if key not in obj:
            errors.append(f"{where}: missing key '{key}'")
        elif not isinstance(obj[key], expected):
            errors.append(f"{where}.{key}: expected {expected.__name__}, got {type(obj[key]).__name__}")
    return errors


def check_sample(sample: object, name: str) -> tuple[list[str], list[str]]:
    """Check one parsed input file. Returns (errors, warnings).

    An error makes the file unusable; a warning means a span is dropped or something
    looks off but the task can still be built.
    """
    if not isinstance(sample, dict):
        return [f"{name}: top level is {type(sample).__name__}, expected an object"], []
    errors = _type_errors(sample, SAMPLE_FIELDS, name)
    if errors:
        return errors, []

    warnings = []
    if not sample["id"].strip():
        errors.append(f"{name}.id: empty")
    elif sample["id"] != Path(name).stem:
        warnings.append(f"{name}.id: {sample['id']!r} does not match the file name")
    if not sample["question"].strip():
        errors.append(f"{name}.question: empty")

    usable = 0
    for i, span in enumerate(sample["spans"]):
        where = f"{name}.spans[{i}]"
        if not isinstance(span, dict):
            errors.append(f"{where}: expected an object, got {type(span).__name__}")
            continue
        span_errors = _type_errors(span, SPAN_FIELDS, where)
        for key in SPAN_INT_FIELDS:
            value = span.get(key)
            # bool is a subclass of int in Python, but not a valid offset or count.
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                span_errors.append(f"{where}.{key}: expected int, got {type(value).__name__}")
        if span_errors:
            errors.extend(span_errors)
            continue
        if span["label"] not in LABELS:
            warnings.append(f"{where}.label: unknown label {span['label']!r}, span dropped")
        elif not span["text"].strip():
            warnings.append(f"{where}.text: empty, span dropped")
        else:
            usable += 1
    if not errors and not usable:
        errors.append(f"{name}.spans: no span with a known label and text")
    return errors, warnings


def check_task(task: object, where: str) -> list[str]:
    """Check one output task against the task format in the module docstring."""
    if not isinstance(task, dict):
        return [f"{where}: expected an object, got {type(task).__name__}"]
    errors = _type_errors(task, TASK_FIELDS, where)
    extra = sorted(set(task) - set(TASK_FIELDS))
    if extra:
        errors.append(f"{where}: unexpected keys {extra}")
    if errors:
        return errors

    for key in ("id", "data_source", "question"):
        if not task[key].strip():
            errors.append(f"{where}.{key}: empty")
    labels = task["ref_labels"]
    if not labels:
        errors.append(f"{where}.ref_labels: empty")
    unknown = [label for label in labels if label not in LABELS]
    if unknown:
        errors.append(f"{where}.ref_labels: unknown labels {unknown}")
    # ref_cot must be exactly one "[label] text" block per ref_labels entry, in order.
    cot_labels = []
    for block in _BLANK_LINES.split(task["ref_cot"]):
        match = _COT_BLOCK.match(block)
        cot_labels.append(match.group(1) if match else None)
    if cot_labels != labels:
        if len(cot_labels) != len(labels):
            detail = f"{len(cot_labels)} blocks for {len(labels)} ref_labels"
        else:
            i = next(i for i, (a, b) in enumerate(zip(cot_labels, labels)) if a != b)
            detail = f"block {i} is {cot_labels[i]!r}, ref_labels[{i}] is {labels[i]!r}"
        errors.append(f"{where}.ref_cot: '[label] text' blocks do not match ref_labels ({detail})")
    expected_answer = (extract_boxed(task["solution"]) or "").strip()
    if task["answer"] != expected_answer:
        errors.append(f"{where}.answer: {task['answer']!r} is not the \\boxed{{}} content of "
                      f"solution ({expected_answer!r})")
    return errors


def check_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """Parse and check every line of one output file. Returns (valid tasks, errors)."""
    if not path.is_file():
        return [], [f"{path}: missing"]
    tasks, errors = [], []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            where = f"{path}:{lineno}"
            if not line.strip():
                errors.append(f"{where}: blank line")
                continue
            try:
                task = json.loads(line.rstrip("\r\n"))
            except json.JSONDecodeError as e:
                errors.append(f"{where}: invalid JSON at column {e.colno}: {e.msg}")
                continue
            task_errors = check_task(task, where)
            errors.extend(task_errors)
            if not task_errors:
                tasks.append(task)
    return tasks, errors


def check_outputs(out_dir: Path) -> tuple[dict[str, int], list[str]]:
    """Check train.jsonl and validation.jsonl, including ids across the two splits."""
    splits, errors = {}, []
    for name in ("train", "validation"):
        splits[name], split_errors = check_jsonl(out_dir / f"{name}.jsonl")
        errors.extend(split_errors)
    seen: dict[str, str] = {}
    for name, tasks in splits.items():
        for task in tasks:
            first = seen.get(task["id"])
            if first is None:
                seen[task["id"]] = name
            elif first == name:
                errors.append(f"id {task['id']} is duplicated in {name}")
            else:
                errors.append(f"id {task['id']} appears in both {first} and {name}")
    if not splits["train"] and not errors:
        errors.append(f"{out_dir / 'train.jsonl'}: no tasks")
    return {name: len(tasks) for name, tasks in splits.items()}, errors


def report(kind: str, messages: list[str]) -> None:
    for message in messages:
        print(f"{kind}: {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, matching braces so nested ones survive."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


def load_solution(solutions_dir: str, sample_id: str, question: str) -> str:
    """The upstream reference solution, or "" if it is missing, malformed or for another question."""
    if not solutions_dir:
        return ""
    path = Path(solutions_dir) / sample_id / "environment" / "trajectory.json"
    if not path.is_file():
        return ""
    try:
        upstream = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        report("warning", [f"{path}: not valid JSON ({e}); no answer for {sample_id}"])
        return ""
    if not isinstance(upstream, dict) or not isinstance(upstream.get("solution", ""), str):
        report("warning", [f"{path}: expected an object with a string 'solution'; no answer for {sample_id}"])
        return ""
    # Ids are only trusted when the question matches as well.
    if str(upstream.get("question") or "").strip() != question:
        return ""
    return (upstream.get("solution") or "").strip()


def labelled_reasoning(spans: list[dict]) -> str:
    """'[label] text' per span, blank-line separated, as in the SFT targets.

    The reward splits a trace into steps at blank lines, so a blank line inside a span
    is folded into a single newline to keep each labelled span one step.
    """
    pieces = [f"[{s['label']}] {_BLANK_LINES.sub(chr(10), s['text'].strip())}" for s in spans]
    return "\n\n".join(pieces)


def build_task(sample: dict, solutions_dir: str, data_source: str) -> dict | None:
    sample_id = str(sample.get("id") or "")
    question = (sample.get("question") or "").strip()
    spans = [
        s for s in (sample.get("spans") or [])
        if (s.get("text") or "").strip() and s.get("label") in LABELS
    ]
    if not sample_id or not question or not spans:
        return None
    solution = load_solution(solutions_dir, sample_id, question)
    return {
        "id": sample_id,
        "data_source": data_source,
        "question": question,
        "answer": (extract_boxed(solution) or "").strip(),
        "ref_labels": [s["label"] for s in spans],
        "ref_cot": labelled_reasoning(spans),
        "solution": solution,
    }


def load_val_ids(split_from: str) -> set[str] | None:
    if not split_from:
        return None
    path = Path(split_from)
    if not path.is_file():
        raise SystemExit(f"--split-from: {path} not found (pass --split-from '' for a random split)")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"--split-from: {path} is not valid JSON (line {e.lineno} column {e.colno}): {e.msg}")
    if not isinstance(rows, list) or not all(isinstance(r, dict) and isinstance(r.get("id"), str) for r in rows):
        raise SystemExit(f"--split-from: {path} must be a JSON list of objects with a string 'id'")
    # Step ids look like sample_007228_step_000; the trace id is the part before _step_.
    return {re.sub(r"_step_\d+$", "", row["id"]) for row in rows}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    if args.check_only:
        counts, errors = check_outputs(out_dir)
        report("error", errors)
        status = "OK" if not errors else f"{len(errors)} error(s)"
        print(f"format check  : {out_dir}/ {status} "
              f"({counts['train']} train / {counts['validation']} validation tasks valid)")
        raise SystemExit(1 if errors else 0)

    files = sorted(Path(args.input_dir).glob("sample_*.json"))
    if not files:
        raise SystemExit(f"no sample_*.json found under {args.input_dir}")

    tasks, skipped, input_errors, warnings = [], [], [], []
    for path in files:
        try:
            sample = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors = [f"{path.name}: invalid JSON at line {e.lineno} column {e.colno}: {e.msg}"]
        except UnicodeDecodeError as e:
            errors = [f"{path.name}: not UTF-8 ({e.reason} at byte {e.start})"]
        else:
            errors, sample_warnings = check_sample(sample, path.name)
            warnings.extend(sample_warnings)
        task = None if errors else build_task(sample, args.solutions_dir, args.data_source)
        if task is None:
            input_errors.extend(errors)
            skipped.append(path.name)
            continue
        tasks.append(task)

    report("warning", warnings)
    report("error", input_errors)
    if skipped and args.strict:
        raise SystemExit(f"{len(skipped)} input file(s) failed the format check; stopping (--strict)")
    if not tasks:
        raise SystemExit("no input file passed the format check")

    val_ids = load_val_ids(args.split_from)
    if val_ids is not None:
        val = [t for t in tasks if t["id"] in val_ids]
        train = [t for t in tasks if t["id"] not in val_ids]
    else:
        shuffled = sorted(tasks, key=lambda t: t["id"])
        random.Random(args.seed).shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * args.val_ratio))
        val, train = shuffled[:n_val], shuffled[n_val:]
    if not train:
        raise SystemExit("the split left no training tasks")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("validation", val)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for task in split:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

    # Read the files back: what rLLM will load is what gets checked.
    _, output_errors = check_outputs(out_dir)
    if output_errors:
        report("error", output_errors)
        raise SystemExit(f"the written files in {out_dir}/ failed the format check")

    print(f"input files   : {len(files)}" + (f" ({len(skipped)} skipped: {', '.join(skipped)})" if skipped else ""))
    print(f"format check  : {len(files) - len(skipped)} input files OK, {len(warnings)} warning(s); "
          f"output OK")
    print(f"train / val   : {len(train)} / {len(val)} -> {out_dir}/"
          f" (val traces {'from ' + args.split_from if val_ids is not None else 'random'})")
    print(f"val traces    : {', '.join(t['id'] for t in val)}")
    no_answer = [t["id"] for t in tasks if not t["answer"]]
    print(f"with answer   : {len(tasks) - len(no_answer)} / {len(tasks)}"
          + (f" (label-only reward for: {', '.join(no_answer)})" if no_answer else ""))
    counts = Counter(label for t in train for label in t["ref_labels"])
    total = sum(counts.values())
    print(f"reference label shares (train, {total} spans):")
    for label, count in counts.most_common():
        print(f"  {label:22s} {count / total:6.1%}")


if __name__ == "__main__":
    main()
