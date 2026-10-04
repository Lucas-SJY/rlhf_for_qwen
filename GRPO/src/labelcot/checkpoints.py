"""Validate the pinned verl FSDP checkpoint layout before resume/export."""

from __future__ import annotations

import json
from pathlib import Path


def resolve_step(checkpoint_dir: Path, step: int = 0) -> int:
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    if step == 0:
        tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
        try:
            step = int(tracker.read_text().strip())
        except (OSError, ValueError) as exc:
            raise ValueError(f"no valid checkpoint tracker at {tracker}") from exc
    if step <= 0:
        raise ValueError("checkpoint step must be positive")
    return step


def validate_checkpoint(checkpoint_dir: Path, step: int, for_resume: bool = False) -> Path:
    actor = checkpoint_dir / f"global_step_{step}" / "actor"
    try:
        metadata = json.loads((actor / "fsdp_config.json").read_text())
        world = metadata["world_size"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"missing/invalid FSDP metadata in {actor}") from exc
    if type(world) is not int or world < 1:
        raise ValueError(f"invalid world_size in {actor}")
    required = [actor / "huggingface" / "config.json"]
    kinds = ("model", "optim", "extra_state") if for_resume else ("model",)
    for kind in kinds:
        required.extend(actor / f"{kind}_world_size_{world}_rank_{rank}.pt" for rank in range(world))
    if for_resume:
        required.append(actor.parent / "data.pt")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"incomplete checkpoint: {missing}; inspect retained checkpoints before resuming")
    return actor
