#!/usr/bin/env python3
"""Per-question outcome of a GRPO run's rollouts, counted from rLLM's episode logs.

rLLM writes every rollout of a step to its own JSON file (rllm/utils/episode_logger.py):

    <episodes>/train_step_<N>_epoch_<E>/episode_hash<task hash>_id<uid>_<rollout>.json
    <episodes>/val_step_<N>_epoch_<E>/...

Each file holds the task (with its id and reference answer), the termination reason and
the reward breakdown of src/labelcot/reward.py as `metrics`. This groups the files by
step and question and counts, per question:

    rollouts     files found (GROUP_SIZE unless an episode was lost)
    correct      answer_correct, the reward's correctness term; a cut-off answer is wrong
    truncated    cut off at MAX_RESPONSE_LENGTH (dropped from training)
    label_valid  valid tags (the reward's label term)
    signal       the finished rollouts' rewards differ, i.e. GRPO has a nonzero advantage
                 for this question (needs at least two finished rollouts)

Stdlib only, so it runs in a plain python image; ./rollout_stats.sh runs it in a small
pod that mounts the RL PVC and copies back only the summary written to --output.

    python3 rollout_stats.py --episodes /grpo/runs/<run>/episodes/episodes --output stats.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STEP_DIR = re.compile(r"^(?P<mode>train|val)_step_(?P<step>\d+)_epoch_(?P<epoch>\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", required=True, help="rLLM's episode directory (<run>/episodes/episodes)")
    p.add_argument("--mode", choices=["train", "val"], default="train")
    p.add_argument("--run", default="", help="run name recorded in the output")
    p.add_argument("--output", required=True)
    return p.parse_args()


def step_dirs(root: Path, mode: str) -> list[tuple[int, int, Path]]:
    found = []
    for d in root.iterdir():
        m = STEP_DIR.match(d.name)
        if d.is_dir() and m and m["mode"] == mode:
            found.append((int(m["step"]), int(m["epoch"]), d))
    return sorted(found)


def rollout(ep: dict) -> dict:
    """The parts of one episode file that are counted."""
    metrics = ep.get("metrics") or {}
    reason = ep.get("termination_reason") or ""
    truncated = bool(metrics.get("truncated")) or "max_response_length" in reason
    correct = bool(metrics["answer_correct"]) if "answer_correct" in metrics else bool(ep.get("is_correct"))
    return {
        "correct": correct and not truncated,
        "truncated": truncated,
        "label_valid": bool(metrics.get("label_valid")),
        # Absent for a cut-off answer, which training never scores.
        "reward": None if truncated else metrics.get("reward"),
        "reason": reason,
    }


def question_row(step: int, epoch: int, task: dict, rollouts: list[dict]) -> dict:
    rewards = [round(float(r["reward"]), 6) for r in rollouts if r["reward"] is not None]
    return {
        "step": step,
        "epoch": epoch,
        "id": task.get("id", ""),
        "data_source": task.get("data_source", ""),
        "answer": task.get("answer", ""),
        "rollouts": len(rollouts),
        "correct": sum(r["correct"] for r in rollouts),
        "truncated": sum(r["truncated"] for r in rollouts),
        "label_valid": sum(r["label_valid"] for r in rollouts),
        "signal": len(rewards) >= 2 and len(set(rewards)) > 1,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.episodes)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory; is the run's episode logging on?")
    dirs = step_dirs(root, args.mode)
    if not dirs:
        raise SystemExit(f"no {args.mode}_step_* directories in {root}")

    rows, reasons = [], Counter()
    files = unreadable = 0
    start = time.time()
    for i, (step, epoch, d) in enumerate(dirs, 1):
        by_task: dict[str, tuple[dict, list[dict]]] = {}
        for f in sorted(d.glob("episode_*.json")):
            try:
                ep = json.loads(f.read_text())
            except (OSError, ValueError) as e:
                unreadable += 1
                print(f"[stats] skipping {f}: {e}", file=sys.stderr)
                continue
            files += 1
            task = ep.get("task") or {}
            key = ep.get("task_hash") or task.get("id") or f.name
            r = rollout(ep)
            reasons[r["reason"] or "none"] += 1
            by_task.setdefault(key, (task, []))[1].append(r)
        rows.extend(question_row(step, epoch, task, rs) for task, rs in by_task.values())
        if i % 10 == 0 or i == len(dirs):
            print(f"[stats] {i}/{len(dirs)} steps, {files} rollouts, {time.time() - start:.0f}s", file=sys.stderr, flush=True)

    hist = Counter(r["correct"] for r in rows)
    summary = {
        "run": args.run,
        "mode": args.mode,
        "episodes_dir": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps": len(dirs),
        "questions": len(rows),
        "rollouts": files,
        "unreadable_files": unreadable,
        "termination_reasons": dict(reasons.most_common()),
        "correct_histogram": {str(k): hist[k] for k in sorted(hist)},
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    total = sum(r["rollouts"] for r in rows)
    print(f"[stats] {len(rows)} questions over {len(dirs)} steps, {total} rollouts: "
          f"{sum(r['correct'] for r in rows) / max(1, total):.1%} correct, "
          f"{sum(r['truncated'] for r in rows) / max(1, total):.1%} cut off; "
          f"correct-count histogram {summary['correct_histogram']}", file=sys.stderr)


if __name__ == "__main__":
    main()
