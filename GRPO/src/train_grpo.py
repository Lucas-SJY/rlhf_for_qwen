#!/usr/bin/env python3
"""GRPO on the SFT checkpoint with rLLM's unified trainer on the verl backend.

Every setting is a Hydra override on rLLM's ``unified`` config (rLLM's backend-agnostic
``rllm.*`` settings plus verl's ``ppo_trainer``); ``entrypoint.sh`` assembles them from
environment variables. This file only wires three things together:

    data       GRPO/data/{train,validation}.jsonl  -> rllm.data.Dataset (one task per line);
               TRAIN_FILE / VAL_FILE override the paths
    workflow   labelcot.workflow.LabeledCoTWorkflow (rollout + label reward)
    ray        a local single-node cluster sized for one pod

Run it through entrypoint.sh; calling it directly needs the same overrides.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import hydra
import ray

from labelcot.config import recipe_identity, settings_from_config
from labelcot.data import load_tasks
from labelcot.runtime import mark_run, prepare_run, run_lock, validate_settings

# GRPO/data next to src/: /workspace/data in the image, the checkout's GRPO/data natively.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def reward_weights_from_env() -> dict[str, float]:
    return {
        "align": float(os.environ.get("REWARD_W_ALIGN", "0.5")),
        "correct": float(os.environ.get("REWARD_W_CORRECT", "0.5")),
    }


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config) -> None:
    from rllm.trainer.verl.utils import sync_config
    sync_config(config)
    settings = settings_from_config(config)
    train_tasks = load_tasks(os.environ.get("TRAIN_FILE", str(DATA_DIR / "train.jsonl")))
    val_tasks = load_tasks(os.environ.get("VAL_FILE", str(DATA_DIR / "validation.jsonl")))
    for warning in validate_settings(settings, train_tasks, val_tasks):
        print(f"[preflight] warning: {warning}")
    # Prevent backend/CLI changes from silently turning this into another algorithm.
    if config.rllm.algorithm.adv_estimator != "grpo" or config.rllm.async_training.enable:
        raise ValueError("this harness requires synchronous GRPO (rllm.async_training.enable=false)")
    if config.trainer.resume_mode != "auto":
        raise ValueError("resume_mode must be auto; choose a new RUN_NAME for a fresh run")
    if config.actor_rollout_ref.actor.checkpoint.get("async_save", False):
        raise ValueError("async checkpoint saving is unsupported by the completion/resume guard")
    for operation in ("save_contents", "load_contents"):
        if not {"model", "optimizer", "extra"}.issubset(config.actor_rollout_ref.actor.checkpoint[operation]):
            raise ValueError(f"checkpoint.{operation} must include model, optimizer and extra state")
    if config.actor_rollout_ref.actor.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("this harness exports FSDP checkpoints; select fsdp or fsdp2")
    if config.actor_rollout_ref.model.lora_adapter_path is not None:
        raise ValueError("start from the merged full SFT model; a preloaded adapter changes the reference policy")
    if config.rllm.trainer.val_only:
        raise ValueError("val_only cannot mark a training run complete; use val_before_train for its baseline")

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < settings["gpus"]:
        raise RuntimeError(f"training needs {settings['gpus']} visible NVIDIA GPU(s); CPU diagnostics use ./run.sh diagnose")

    from labelcot.reward import MATH_VERIFY_AVAILABLE
    if not MATH_VERIFY_AVAILABLE:
        raise RuntimeError("training requires math-verify; use the pinned Docker image")
    from transformers import AutoTokenizer
    from preflight import check_prompts
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], trust_remote_code=False)
    check_prompts(train_tasks + val_tasks, tokenizer, settings["max_prompt_length"])

    run_dir = Path(config.trainer.default_local_dir).resolve().parent
    with run_lock(run_dir):
        prepare_run(run_dir, settings, train_tasks, val_tasks, recipe_identity(config))
        status_file = run_dir / "run_status.json"
        if status_file.is_file() and json.loads(status_file.read_text()).get("status") == "completed":
            print(f"[train_grpo] {run_dir} already completed; use a new RUN_NAME for another experiment")
            return
        mark_run(run_dir, "running")
        try:
            train(config, train_tasks, val_tasks)
        except BaseException:
            mark_run(run_dir, "failed")
            raise
        else:
            mark_run(run_dir, "completed")


def train(config, train_tasks, val_tasks) -> None:
    from rllm.data.dataset import Dataset
    from rllm.trainer import AgentTrainer
    from rllm.trainer.ray_init_utils import get_ray_init_settings
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

    from labelcot.workflow import LabeledCoTWorkflow

    train_dataset = Dataset(data=train_tasks, name="bespoke_labeled_cot", split="train")
    val_dataset = Dataset(data=val_tasks, name="bespoke_labeled_cot", split="validation")
    print(f"[train_grpo] train / val tasks: {len(train_tasks)} / {len(val_tasks)}")

    if not ray.is_initialized():
        settings = get_ray_init_settings(config)
        # The object store lives in /dev/shm; k8s/job.yaml mounts a memory-backed
        # emptyDir there. Keep this below that sizeLimit.
        settings.setdefault("object_store_memory", int(float(os.environ.get("RAY_OBJECT_STORE_GB", "16")) * 1024**3))
        # Ray sizes its CPU pool from the pod's CPU limit, and verl alone reserves 3 CPUs per
        # GPU plus 1 for the task runner. RAY_NUM_CPUS lets a pod with fewer real CPUs
        # still advertise enough; Ray's CPUs only gate scheduling, so oversubscribing is fine.
        if os.environ.get("RAY_NUM_CPUS"):
            settings.setdefault("num_cpus", int(os.environ["RAY_NUM_CPUS"]))
        runtime_env = get_ppo_ray_runtime_env()
        runtime_env.setdefault("env_vars", {}).update({
            name: os.environ[name] for name in ("WANDB_RUN_ID", "WANDB_RESUME", "MAX_ZERO_SIGNAL_STEPS")
            if name in os.environ
        })
        ray.init(runtime_env=runtime_env, **settings)

    weights = reward_weights_from_env()
    print(f"[train_grpo] reward weights: {weights}")

    try:
        trainer = AgentTrainer(
            workflow_class=LabeledCoTWorkflow,
            workflow_args={"reward_weights": weights, "max_label_retries": int(os.environ.get("MAX_LABEL_RETRIES", "3"))},
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            backend="verl",
        )
        trainer.train()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
