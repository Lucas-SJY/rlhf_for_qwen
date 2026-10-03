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

import json
import os
from pathlib import Path

import hydra
import ray

# GRPO/data next to src/: /workspace/data in the image, the checkout's GRPO/data natively.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def reward_weights_from_env() -> dict[str, float]:
    return {
        "align": float(os.environ.get("REWARD_W_ALIGN", "0.5")),
        "correct": float(os.environ.get("REWARD_W_CORRECT", "0.5")),
    }


def load_tasks(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config) -> None:
    from rllm.data.dataset import Dataset
    from rllm.trainer import AgentTrainer
    from rllm.trainer.ray_init_utils import get_ray_init_settings
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

    from labelcot.workflow import LabeledCoTWorkflow

    train = Dataset(data=load_tasks(os.environ.get("TRAIN_FILE", str(DATA_DIR / "train.jsonl"))), name="bespoke_labeled_cot", split="train")
    val = Dataset(data=load_tasks(os.environ.get("VAL_FILE", str(DATA_DIR / "validation.jsonl"))), name="bespoke_labeled_cot", split="validation")
    print(f"[train_grpo] train / val tasks: {len(train)} / {len(val)}")

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
        ray.init(runtime_env=get_ppo_ray_runtime_env(), **settings)

    weights = reward_weights_from_env()
    print(f"[train_grpo] reward weights: {weights}")

    trainer = AgentTrainer(
        workflow_class=LabeledCoTWorkflow,
        workflow_args={"reward_weights": weights, "max_label_retries": int(os.environ.get("MAX_LABEL_RETRIES", "3"))},
        config=config,
        train_dataset=train,
        val_dataset=val,
        backend="verl",
    )
    trainer.train()


if __name__ == "__main__":
    main()
