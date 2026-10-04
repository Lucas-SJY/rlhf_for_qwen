"""Run provenance and resume guards. No credentials or arbitrary environment dumps."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from labelcot.data import dataset_summary, validate_splits
from labelcot.checkpoints import resolve_step, validate_checkpoint
from labelcot.reward import RewardWeights


def validate_settings(settings: dict, train: list[dict], val: list[dict]) -> list[str]:
    validate_splits(train, val)
    positive = ("batch_size", "group_size", "mini_batch_size", "max_prompt_length",
                "max_response_length", "token_budget", "epochs", "gpus", "keep_checkpoints")
    for key in positive:
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if settings["group_size"] < 2:
        raise ValueError("GRPO needs GROUP_SIZE >= 2 for a group-relative baseline")
    batch, mini = settings["batch_size"], settings["mini_batch_size"]
    if len(train) < batch:
        raise ValueError(f"{len(train)} training tasks < TRAIN_BATCH_SIZE={batch}; all batches would be dropped")
    if batch % mini:
        raise ValueError("TRAIN_BATCH_SIZE must be divisible by PPO_MINI_BATCH_SIZE")
    if (mini * settings["group_size"]) % settings["gpus"]:
        raise ValueError("mini-batch response count must be divisible by N_GPUS")
    if settings["token_budget"] < settings["max_prompt_length"] + settings["max_response_length"]:
        raise ValueError("PPO_MAX_TOKEN_LEN must hold one full prompt plus response")
    for key in ("lr", "gpu_memory", "clip_low", "clip_high", "kl_beta"):
        if not math.isfinite(settings[key]):
            raise ValueError(f"{key} must be finite")
    if settings["lr"] <= 0 or not 0 < settings["gpu_memory"] < 1:
        raise ValueError("ACTOR_LR must be > 0 and ROLLOUT_GPU_MEM_UTIL between 0 and 1")
    if not 0 < settings["clip_low"] < 1 or settings["clip_high"] <= 0 or settings["kl_beta"] < 0:
        raise ValueError("invalid clipping bounds or KL coefficient")
    for key in ("lora_rank", "max_label_retries", "max_zero_signal_steps"):
        if settings[key] < 0:
            raise ValueError(f"{key} cannot be negative")
    if settings["lora_rank"] > 0 and settings["lora_alpha"] <= 0:
        raise ValueError("LORA_ALPHA must be positive")
    if settings["total_steps"] != -1 and settings["total_steps"] <= 0:
        raise ValueError("TOTAL_TRAINING_STEPS must be -1 or positive")
    if settings["save_freq"] <= 0:
        raise ValueError("SAVE_FREQ must be positive so interrupted runs can resume")
    if settings["test_freq"] != -1 and settings["test_freq"] <= 0:
        raise ValueError("TEST_FREQ must be -1 or positive")
    RewardWeights(**settings["reward_weights"])
    warnings = []
    if len(train) % batch:
        warnings.append(f"{len(train) % batch} training tasks are dropped per epoch by full-batch loading")
    available = len(train) // batch * settings["epochs"]
    if settings["total_steps"] > available:
        warnings.append(f"epoch limit permits {available} batches, below step cap {settings['total_steps']}")
    if settings["lora_rank"] == 0 and settings["lr"] > 1e-6:
        warnings.append("full-parameter training: consider ACTOR_LR=1e-6 instead of the LoRA default")
    if settings["keep_checkpoints"] < 2:
        warnings.append("only one checkpoint retained; a second provides a recovery fallback")
    return warnings


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def run_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another training driver holds {run_dir}; use another RUN_NAME") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")) + [root / "entrypoint.sh"]:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_run(run_dir: Path, settings: dict, train: list[dict], val: list[dict],
                resolved_recipe: dict | None = None) -> dict:
    """Call under run_lock. Require the same recipe/data/code for automatic resume."""
    dependencies = {}
    for name in ("rllm", "verl", "vllm", "torch", "transformers", "math-verify"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not installed"
    identity = {
        "settings": settings,
        "datasets": {"train": dataset_summary(train), "validation": dataset_summary(val)},
        "code_sha256": code_fingerprint(),
        "dependencies": dependencies,
        "recipe": resolved_recipe or {},
    }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("run recipe/data/code changed; use a new RUN_NAME to avoid incompatible resume")
    else:
        if any((run_dir / "checkpoints").glob("global_step_*")):
            raise ValueError("existing checkpoints have no run manifest; use a new RUN_NAME")
        manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "wandb_run_id": uuid.uuid4().hex[:12], "identity": identity}
        atomic_json(manifest_path, manifest)
    checkpoint_dir = run_dir / "checkpoints"
    if (checkpoint_dir / "latest_checkpointed_iteration.txt").exists():
        validate_checkpoint(checkpoint_dir, resolve_step(checkpoint_dir), for_resume=True)
    elif any(checkpoint_dir.glob("global_step_*")):
        raise ValueError("checkpoint directories exist without a committed tracker; inspect the interrupted save")
    os.environ["WANDB_RUN_ID"] = manifest["wandb_run_id"]
    os.environ["WANDB_RESUME"] = "allow"
    return manifest


def mark_run(run_dir: Path, status: str) -> None:
    atomic_json(run_dir / "run_status.json", {
        "status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
    })
