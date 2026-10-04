#!/usr/bin/env bash
# Training entrypoint: one node, GRPO via rLLM's unified trainer on verl.
#
# Used by the NRP Job (k8s/job.yaml) inside the image. It does
# not depend on where it lives: paths are resolved relative to this file, so it also runs
# from a checkout with a suitable Python environment.
#
# Every knob comes from the environment (the experiment's Secret on the cluster, or an env
# file) and is turned into a Hydra override below. Backend-agnostic settings use rLLM's
# rllm.* paths; verl-only settings (model, LoRA, FSDP, vLLM) use verl's own paths.
# Extra CLI arguments are appended last, so they win:
#   bash src/entrypoint.sh rllm.trainer.total_batches=2
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
case ":${PYTHONPATH:-}:" in
  *":${SRC_DIR}:"*) ;;
  *) export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
# Without ENV_FILE, fall back to .env in the project root (two levels above this file).
# Variables already set win, so `RUN_NAME=x bash src/entrypoint.sh` stays an override.
# Plain KEY=value lines only.
ENV_FILE="${ENV_FILE:-${SRC_DIR}/../../.env}"
if [ -f "${ENV_FILE}" ]; then
  echo "[entrypoint] loading ${ENV_FILE} (existing env vars take precedence)"
  while IFS= read -r line || [ -n "${line}" ]; do
    case "${line}" in '' | '#'*) continue ;; esac
    key="${line%%=*}"
    case "${key}" in '' | *[!A-Za-z0-9_]*) continue ;; esac
    [ -z "${!key+set}" ] && export "${key}=${line#*=}"
  done < "${ENV_FILE}"
fi

# The image is built on CUDA 13. On a host driver older than the 580 branch, use the
# forward-compatibility libraries the image ships (datacenter GPUs such as the A100
# support this on the 535/550/570 LTS branches). On a new enough driver, leave them out:
# loading them there fails. Only inside the image (the Dockerfile sets CUDA_COMPAT_AUTO=1);
# a native environment brings its own CUDA stack.
if [ "${CUDA_COMPAT_AUTO:-0}" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
  echo "[entrypoint] host driver ${driver:-unknown}"
  if [ -n "${driver}" ] && [ "${driver%%.*}" -lt 580 ] && [ -d /usr/local/cuda/compat ]; then
    echo "[entrypoint] driver < 580: enabling CUDA forward compatibility"
    export LD_LIBRARY_PATH="/usr/local/cuda/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

# vLLM's sleep mode (rollout and training time-share the GPU) refuses expandable segments,
# which the SFT .env turns on.
for var in PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF; do
  if [[ "${!var:-}" == *expandable_segments:True* ]]; then
    echo "[entrypoint] unsetting ${var}=${!var} (incompatible with vLLM sleep mode)"
    unset "${var}"
  fi
done

RUN_NAME="${RUN_NAME:-qwen3-8b-grpo-dayallen-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/grpo/runs}"
case "${RUN_NAME}" in
  '' | *[!a-zA-Z0-9_.-]* | . | ..) echo "invalid RUN_NAME" >&2; exit 1 ;;
esac
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
export MAX_ZERO_SIGNAL_STEPS="${MAX_ZERO_SIGNAL_STEPS:-5}"

# A smoke test proves the whole loop (group rollout, reward, advantage, update,
# checkpoint) in a few minutes. It overrides the size settings from the env file on
# purpose: those are filled in for the real run, and a smoke test must stay small. Extra
# Hydra arguments on the command line still win.
if [ "${SMOKE_TEST:-false}" = "true" ]; then
  echo "[entrypoint] SMOKE_TEST: 2 questions x 4 samples, short responses, 3 steps"
  TRAIN_BATCH_SIZE=2 GROUP_SIZE=4 PPO_MINI_BATCH_SIZE=1 MAX_RESPONSE_LENGTH=1024
  TOTAL_TRAINING_STEPS=3 SAVE_FREQ=2 TEST_FREQ=-1 VAL_BEFORE_TRAIN=false
  export MAX_ZERO_SIGNAL_STEPS=0
  RUN_DIR="${RUN_DIR}-smoke"
fi
mkdir -p "${RUN_DIR}"

# Monitoring, configured the same way as ../train: REPORT_TO=wandb streams every metric to
# wandb.ai, which survives a dropped kubectl connection or an evicted pod. Empty REPORT_TO
# means console only. Left unset, wandb is used whenever WANDB_API_KEY is present.
case "${REPORT_TO-__unset__}" in
  __unset__)
    if [ -n "${WANDB_API_KEY:-}" ]; then LOGGER='["console","wandb"]'; else LOGGER='["console"]'; fi
    ;;
  *wandb*)
    if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
      echo "[entrypoint] error: REPORT_TO=wandb needs WANDB_API_KEY (or WANDB_MODE=offline)" >&2
      exit 1
    fi
    LOGGER='["console","wandb"]'
    ;;
  *)
    LOGGER='["console"]'
    ;;
esac
# Offline runs and wandb's local files go to the run directory on the PVC.
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}}"
echo "[entrypoint] logger ${LOGGER}"

# ---------------------------------------------------------------------------
# Hydra overrides
# ---------------------------------------------------------------------------
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
# Token budget per micro-batch; must hold one full prompt + response.
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 2048))}"

# Data and GRPO sampling: TRAIN_BATCH_SIZE questions per step, GROUP_SIZE answers each.
# Training samples at temperature 1 with no top-p/top-k, so the rollout distribution is
# the policy the actor computes logprobs for; validation uses Qwen3's thinking defaults.
DATA=(
  rllm.data.train_batch_size="${TRAIN_BATCH_SIZE:-8}"
  rllm.data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  rllm.data.max_response_length="${MAX_RESPONSE_LENGTH}"
  rllm.data.seed="${SEED:-42}"
  rllm.rollout.n="${GROUP_SIZE:-8}"
  rllm.rollout.n_val=1
  rllm.rollout.train.temperature=1.0
  rllm.rollout.train.top_p=1.0
  rllm.rollout.val.temperature=0.6
  rllm.rollout.val.top_p=0.95
  +rllm.rollout.val.top_k=20
  rllm.workflow.raise_on_error=true
)

# GRPO: advantage = (reward - group mean) / group std; clipped policy loss with a higher
# upper clip bound; KL to the SFT policy as a loss term (k3 estimator).
ALGORITHM=(
  rllm.async_training.enable=false
  rllm.algorithm.adv_estimator=grpo
  rllm.algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD:-true}"
  rllm.algorithm.loss_agg_mode=token-mean
  rllm.algorithm.eps_clip="${CLIP_LOW:-0.2}"
  rllm.algorithm.eps_clip_high="${CLIP_HIGH:-0.28}"
  rllm.algorithm.kl_beta="${KL_BETA:-0.001}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
)

# Policy: the full-parameter SFT checkpoint plus a LoRA adapter that GRPO adds and trains.
# The reference policy is the same weights with the adapter switched off.
# LORA_RANK=0 trains all parameters instead; verl then keeps a separate frozen copy of
# the model as the reference policy.
LORA_RANK="${LORA_RANK:-64}"
ACTOR=(
  actor_rollout_ref.model.path="${POLICY_MODEL_PATH:-/data/runs/qwen3-8b-sft-v3}"
  actor_rollout_ref.model.lora_rank="${LORA_RANK}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-5}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-4}"
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}"
  # One GPU shared by rollout and training: the actor leaves the card while vLLM runs.
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}"
)

# Rollout: vLLM in the same process group, sleeping while the actor trains.
ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}"
)

# LoRA only: vLLM loads the frozen base weights from disk once and afterwards receives
# just the adapter (verl's tested LoRA setup). Full-parameter training keeps verl's
# defaults, which sync all weights into vLLM after every update.
# The frozen base weights are also kept in bf16 instead of verl's fp32 default. The SFT
# checkpoint is stored in bf16 and FSDP computes in bf16 either way, so the forward pass
# is unchanged, but the 8B actor takes ~16 GB on the GPU instead of ~33 GB, which is what
# targets a 48 GB card (measure actual peaks in a GPU smoke run). PEFT keeps the adapter in fp32.
if [ "${LORA_RANK}" -gt 0 ]; then
  ACTOR+=(
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-32}"
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE:-bf16}"
  )
  ROLLOUT+=(
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
  )
fi

TRAINER=(
  # The Job requests one GPU. With N_GPUS > 1, FSDP shards the actor and vLLM runs one
  # replica per GPU.
  trainer.n_gpus_per_node="${N_GPUS:-1}"
  trainer.nnodes=1
  rllm.trainer.total_epochs="${EPOCHS:-1}"
  rllm.trainer.total_batches="${TOTAL_TRAINING_STEPS:-200}"
  rllm.trainer.save_freq="${SAVE_FREQ:-20}"
  rllm.trainer.test_freq="${TEST_FREQ:-20}"
  rllm.trainer.val_before_train="${VAL_BEFORE_TRAIN:-true}"
  rllm.trainer.project_name="${WANDB_PROJECT:-context-comp-grpo}"
  rllm.trainer.experiment_name="$(basename "${RUN_DIR}")"
  rllm.trainer.logger="${LOGGER}"
  rllm.episode_logging.log_episodes=true
  rllm.episode_logging.episode_log_dir="${RUN_DIR}/episodes"
  trainer.resume_mode=auto
  trainer.default_local_dir="${RUN_DIR}/checkpoints"
  trainer.max_actor_ckpt_to_keep="${KEEP_CHECKPOINTS:-2}"
  hydra.run.dir="${RUN_DIR}/hydra"
)

echo "[entrypoint] run dir ${RUN_DIR}"
echo "[entrypoint] policy ${POLICY_MODEL_PATH:-/data/runs/qwen3-8b-sft-v3}  gpus ${N_GPUS:-1}"

# python3 on PATH, not PYTHON_BIN: PYTHON_BIN is a local path for ../run.sh and also
# is a local-only setting, excluded from the training Secret.
exec python3 "${SRC_DIR}/train_grpo.py" \
  "${DATA[@]}" "${ALGORITHM[@]}" "${ACTOR[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" "$@"
