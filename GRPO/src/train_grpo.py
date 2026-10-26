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
        "label": float(os.environ.get("REWARD_W_LABEL", "0.2")),
        "correct": float(os.environ.get("REWARD_W_CORRECT", "0.8")),
    }


# The sampling keys rLLM's verl engine (VerlEngine) reads from the rollout config; it drops
# every other key of rllm.rollout.{train,val} (rLLM passes those through to the backend
# only on its AgentFlow path).
ENGINE_SAMPLING_KEYS = {"temperature", "top_p", "top_k", "max_tokens"}


def extra_sampling_params(config) -> dict[str, dict]:
    """rllm.rollout.train / val keys the verl engine leaves out (presence_penalty, min_p, ...).

    The workflow hands them to get_token_output_from_token_input, which merges them into
    vLLM's SamplingParams.
    """
    from omegaconf import OmegaConf

    rollout = OmegaConf.to_container(config.rllm.rollout, resolve=True)
    return {mode: {k: v for k, v in (rollout.get(mode) or {}).items() if k not in ENGINE_SAMPLING_KEYS}
            for mode in ("train", "val")}


def repeat_label_args_from_env() -> dict[str, int]:
    """The repeated-label retry (labelcot.workflow): a run of REPEAT_MIN_PARAGRAPHS consecutive
    paragraphs with the same label, over REPEAT_MIN_TOKENS tokens, stops a training completion,
    which is sampled again up to REPEAT_LABEL_RETRIES times (0 turns it off); completions are
    generated in chunks of GEN_CHUNK_TOKENS tokens so the run is caught while it is written.
    REPEAT_MASK_FINAL=true drops a last attempt that still has a run from training."""
    return {
        "repeat_label_retries": int(os.environ.get("REPEAT_LABEL_RETRIES", "2")),
        "repeat_min_paragraphs": int(os.environ.get("REPEAT_MIN_PARAGRAPHS", "3")),
        "repeat_min_tokens": int(os.environ.get("REPEAT_MIN_TOKENS", "200")),
        "gen_chunk_tokens": int(os.environ.get("GEN_CHUNK_TOKENS", "1024")),
        # Drop a last attempt that still has a run from training (entrypoint.sh masks it).
        "repeat_mask_final": os.environ.get("REPEAT_MASK_FINAL", "true").lower() == "true",
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
    extra_sampling = extra_sampling_params(config)
    print(f"[train_grpo] sampling params beyond temperature / top_p / top_k: {extra_sampling}")
    repeat_args = repeat_label_args_from_env()
    print(f"[train_grpo] repeated-label retry: {repeat_args}")

    trainer = AgentTrainer(
        workflow_class=LabeledCoTWorkflow,
        workflow_args={"reward_weights": weights, "max_label_retries": int(os.environ.get("MAX_LABEL_RETRIES", "3")),
                       "extra_sampling": extra_sampling, **repeat_args},
        config=config,
        train_dataset=train,
        val_dataset=val,
        backend="verl",
    )
    trainer.train()


if __name__ == "__main__":
    main()
