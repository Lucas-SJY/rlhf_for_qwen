"""Strict task validation shared by preparation, preflight and training (no GPU imports)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from labelcot.reward import LABELS, answer_is_checkable


def validate_tasks(tasks: list[dict], source: str) -> None:
    if not tasks:
        raise ValueError(f"{source}: no tasks")
    seen = set()
    for i, task in enumerate(tasks, 1):
        where = f"{source}:{i}"
        if not isinstance(task, dict):
            raise ValueError(f"{where}: expected an object")
        for key in ("id", "question", "data_source"):
            if not isinstance(task.get(key), str) or not task[key].strip():
                raise ValueError(f"{where}: {key} must be a nonempty string")
        if task["id"] in seen or task["id"] in ("None", "null"):
            raise ValueError(f"{where}: duplicate or invalid id {task['id']!r}")
        seen.add(task["id"])
        if not isinstance(task.get("answer"), str):
            raise ValueError(f"{where}: answer must be a string (empty is allowed)")
        labels = task.get("ref_labels")
        if not isinstance(labels, list) or not labels or any(label not in LABELS for label in labels):
            raise ValueError(f"{where}: ref_labels must be a nonempty list of canonical labels")


def load_tasks(path: str | Path) -> list[dict]:
    path = Path(path)
    tasks = []
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
    validate_tasks(tasks, str(path))
    return tasks


def validate_splits(train: list[dict], val: list[dict]) -> None:
    validate_tasks(train, "train")
    validate_tasks(val, "validation")
    overlap = {t["id"] for t in train} & {t["id"] for t in val}
    if overlap:
        raise ValueError(f"train/validation id leakage: {sorted(overlap)[:5]}")
    questions = {" ".join(t["question"].split()) for t in train}
    if any(" ".join(t["question"].split()) in questions for t in val):
        raise ValueError("train/validation question leakage (even though ids may differ)")


def dataset_summary(tasks: list[dict]) -> dict:
    # Order affects sampling on resume; intentionally preserve it in the fingerprint.
    raw = json.dumps(tasks, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "count": len(tasks),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "answer_checkable": sum(answer_is_checkable(t["answer"]) for t in tasks),
        "sources": sorted({t["data_source"] for t in tasks}),
    }
