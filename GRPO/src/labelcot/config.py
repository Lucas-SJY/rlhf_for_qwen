"""Small audited settings projection for preflight and run identity."""

from __future__ import annotations

import os


# Local preflight uses environment defaults; the training driver uses resolved Hydra
# values so appended CLI overrides cannot bypass validation or the resume guard.
FIELDS = {
    "batch_size": ("TRAIN_BATCH_SIZE", "rllm.data.train_batch_size", 8, int),
    "group_size": ("GROUP_SIZE", "rllm.rollout.n", 8, int),
    "mini_batch_size": ("PPO_MINI_BATCH_SIZE", "actor_rollout_ref.actor.ppo_mini_batch_size", 4, int),
    "max_prompt_length": ("MAX_PROMPT_LENGTH", "rllm.data.max_prompt_length", 2048, int),
    "max_response_length": ("MAX_RESPONSE_LENGTH", "rllm.data.max_response_length", 8192, int),
    "token_budget": ("PPO_MAX_TOKEN_LEN", "actor_rollout_ref.actor.ppo_max_token_len_per_gpu", 12288, int),
    "epochs": ("EPOCHS", "rllm.trainer.total_epochs", 1, int),
    "total_steps": ("TOTAL_TRAINING_STEPS", "rllm.trainer.total_batches", 200, int),
    "save_freq": ("SAVE_FREQ", "rllm.trainer.save_freq", 20, int),
    "test_freq": ("TEST_FREQ", "rllm.trainer.test_freq", 20, int),
    "gpus": ("N_GPUS", "trainer.n_gpus_per_node", 1, int),
    "keep_checkpoints": ("KEEP_CHECKPOINTS", "trainer.max_actor_ckpt_to_keep", 2, int),
    "lr": ("ACTOR_LR", "actor_rollout_ref.actor.optim.lr", 1e-5, float),
    "gpu_memory": ("ROLLOUT_GPU_MEM_UTIL", "actor_rollout_ref.rollout.gpu_memory_utilization", 0.7, float),
    "clip_low": ("CLIP_LOW", "rllm.algorithm.eps_clip", 0.2, float),
    "clip_high": ("CLIP_HIGH", "rllm.algorithm.eps_clip_high", 0.28, float),
    "kl_beta": ("KL_BETA", "rllm.algorithm.kl_beta", 0.001, float),
    "lora_rank": ("LORA_RANK", "actor_rollout_ref.model.lora_rank", 64, int),
    "lora_alpha": ("LORA_ALPHA", "actor_rollout_ref.model.lora_alpha", 32, int),
    "model": ("POLICY_MODEL_PATH", "actor_rollout_ref.model.path", "/data/runs/qwen3-8b-sft-v3", str),
}


def settings_from_env() -> dict:
    settings = {key: cast(os.environ.get(env, default)) for key, (env, _, default, cast) in FIELDS.items()}
    if os.environ.get("SMOKE_TEST", "false") == "true":
        settings.update(batch_size=2, group_size=4, mini_batch_size=1, max_response_length=1024,
                        total_steps=3, save_freq=2, test_freq=-1)
    if "PPO_MAX_TOKEN_LEN" not in os.environ:
        settings["token_budget"] = settings["max_prompt_length"] + settings["max_response_length"] + 2048
    settings.update(custom_settings())
    if os.environ.get("SMOKE_TEST", "false") == "true":
        settings["max_zero_signal_steps"] = 0
    return settings


def custom_settings() -> dict:
    return {
        "reward_weights": {"align": float(os.environ.get("REWARD_W_ALIGN", "0.5")),
                           "correct": float(os.environ.get("REWARD_W_CORRECT", "0.5"))},
        "max_label_retries": int(os.environ.get("MAX_LABEL_RETRIES", "3")),
        "max_zero_signal_steps": int(os.environ.get("MAX_ZERO_SIGNAL_STEPS", "5")),
    }


def settings_from_config(config) -> dict:
    from omegaconf import OmegaConf

    settings = {key: cast(OmegaConf.select(config, path, default=default))
                for key, (_, path, default, cast) in FIELDS.items()}
    settings.update(custom_settings())
    return settings


def recipe_identity(config) -> dict:
    """Cover all algorithm/backend knobs, not just the human-friendly projection."""
    from omegaconf import OmegaConf

    recipe = {}
    for key in ("rllm", "actor_rollout_ref", "algorithm"):
        recipe[key] = OmegaConf.to_container(config[key], resolve=True)
    # Output paths and logging destination do not define the learned policy. They can
    # contain absolute paths which vary between hosts; omit those from identity.
    recipe["rllm"].pop("trainer", None)
    recipe["rllm"].pop("episode_logging", None)
    return redact_credentials(recipe)


def redact_credentials(value):
    if isinstance(value, dict):
        return {key: ("<redacted>" if key.lower() in
                {"api_key", "password", "secret", "token", "access_token", "hf_token", "env_vars"}
                else redact_credentials(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_credentials(item) for item in value]
    return value
