#!/usr/bin/env python3
"""Build a GRPO task set from the questions in grpo_try/.

grpo_try/ holds sample_*.json files in the bespoke-v2 layout (question + labelled spans),
but without the reference answer. The answer is recovered by sample id from the upstream
harbor dataset (``--solutions-dir``), whose ``solution`` ends in \\boxed{}, the same way
../train/src/merge_answers.py backfilled bespoke-v2.

Only what training needs is kept; the reference trace and its labels are not written,
because the reward does not compare against a reference reasoning. Each output line is:

    id           source sample id; rLLM also uses it to group the GRPO samples of a question
    data_source  "grpo_try" (groups validation metrics: val/grpo_try/...)
    prompt       the model input in chat format: [{"role": "user", "content": question}],
                 exactly as in SFT (no system turn, no label instruction)
    question     the bare question
    answer       reference final answer (the \\boxed{} content of the upstream solution);
                 "" when there is none, e.g. for coding questions. The reward then scores
                 only the label term for that task.

The split is by trace. By default the validation traces are the three held out by the
earlier step-level export of the same data (../train/grpo_test.json, since removed), so
results stay comparable with the runs made on it. --split-from reads the ids from such
a file instead; with neither, the split is random by trace.

Format check. Every input file must be valid JSON with a string id and question; the
labelled spans are not used. A file that fails is reported with its location and
skipped, or stops the run with --strict. An id that does not match the file name only
produces a warning. The written files are then read back line by line and checked
against the task format above; any error there fails the run. --check-only runs just
that last check on existing output. Stdlib only.

Usage (from the repository root):
    python3 GRPO/src/prepare_grpo_try.py
    python3 GRPO/src/prepare_grpo_try.py --strict
    python3 GRPO/src/prepare_grpo_try.py --check-only
    python3 GRPO/src/prepare_grpo_try.py --val-ids '' --val-ratio 0.3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Expected JSON types, for the format check.
SAMPLE_FIELDS = {"id": str, "question": str}
TASK_FIELDS = {
    "id": str,
    "data_source": str,
    "prompt": list,
    "question": str,
    "answer": str,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="grpo_try", help="directory holding sample_*.json")
    p.add_argument("--solutions-dir",
                   default="../jianhong_harbor/harbor/datasets/bespoke-stratos-rest",
                   help="upstream dataset root holding <id>/environment/trajectory.json with a "
                        "'solution' field; empty skips answer recovery")
    p.add_argument("--val-ids", default="sample_008604,sample_013628,sample_014855",
                   help="comma-separated validation trace ids; empty (and no --split-from) "
                        "falls back to a seeded random split by trace")
    p.add_argument("--split-from", default="",
                   help="step-level test file (JSON list of {'id': 'sample_X_step_N'}) whose "
                        "trace ids become the validation set; overrides --val-ids")
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

    An error makes the file unusable; a warning means something looks off but the task
    can still be built.
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
    prompt = task["prompt"]
    if not prompt:
        errors.append(f"{where}.prompt: empty")
    for i, message in enumerate(prompt):
        if not (isinstance(message, dict) and set(message) == {"role", "content"}
                and isinstance(message["role"], str) and isinstance(message["content"], str)):
            errors.append(f"{where}.prompt[{i}]: expected {{'role': str, 'content': str}}")
        elif message["role"] not in ("system", "user", "assistant"):
            errors.append(f"{where}.prompt[{i}].role: unknown role {message['role']!r}")
    if not errors and (prompt[-1]["role"] != "user" or prompt[-1]["content"] != task["question"]):
        errors.append(f"{where}.prompt: the last turn must be the user turn holding the question")
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


def load_answer(solutions_dir: str, sample_id: str, question: str) -> str:
    """The \\boxed{} answer of the upstream solution, or "" if there is none.

    "" also when the upstream file is missing, malformed or for another question.
    """
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
    return (extract_boxed(upstream.get("solution") or "") or "").strip()


def build_task(sample: dict, solutions_dir: str, data_source: str) -> dict | None:
    sample_id = str(sample.get("id") or "")
    question = (sample.get("question") or "").strip()
    if not sample_id or not question:
        return None
    return {
        "id": sample_id,
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "question": question,
        "answer": load_answer(solutions_dir, sample_id, question),
    }


def load_val_ids(split_from: str, val_ids: str) -> set[str] | None:
    if not split_from:
        ids = {i.strip() for i in val_ids.split(",") if i.strip()}
        return ids or None
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

    val_ids = load_val_ids(args.split_from, args.val_ids)
    if val_ids is not None:
        missing = sorted(val_ids - {t["id"] for t in tasks})
        if missing:
            report("warning", [f"validation ids not among the tasks: {', '.join(missing)}"])
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
          f" (val traces {'random' if val_ids is None else 'from ' + (args.split_from or '--val-ids')})")
    print(f"val traces    : {', '.join(t['id'] for t in val)}")
    no_answer = [t["id"] for t in tasks if not t["answer"]]
    print(f"with answer   : {len(tasks) - len(no_answer)} / {len(tasks)}"
          + (f" (label-only reward for: {', '.join(no_answer)})" if no_answer else ""))


if __name__ == "__main__":
    main()
