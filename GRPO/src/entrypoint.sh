#!/usr/bin/env bash
# Training entrypoint: one node, GRPO via rLLM's unified trainer on verl.
#
# Used by the NRP Job (k8s/job.yaml) inside the image. It does
# not depend on where it lives: paths are resolved relative to this file, so it also runs
# from a checkout with a suitable Python environment.
#
# Every knob comes from the environment (the grpo-env Secret on the cluster, or an env
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

RUN_NAME="${RUN_NAME:-qwen3-8b-grpo-labels-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/grpo/runs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

# A smoke test proves the whole loop (group rollout, reward, advantage, update,
# checkpoint) in a few minutes. It overrides the size settings from the env file on
# purpose: those are filled in for the real run, and a smoke test must stay small. Extra
# Hydra arguments on the command line still win.
if [ "${SMOKE_TEST:-false}" = "true" ]; then
  echo "[entrypoint] SMOKE_TEST: 2 questions x 4 samples, short responses, 3 steps"
  TRAIN_BATCH_SIZE=2 GROUP_SIZE=4 PPO_MINI_BATCH_SIZE=1 MAX_RESPONSE_LENGTH=1024
  TOTAL_TRAINING_STEPS=3 SAVE_FREQ=2 TEST_FREQ=-1 VAL_BEFORE_TRAIN=false
  # At 1024 tokens most answers are cut off; masking them would skip the very update the
  # smoke test is meant to exercise.
  MASK_TRUNCATED=false
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
  rllm.workflow.raise_on_error=false
)

# Overlong answers: an answer cut off at MAX_RESPONSE_LENGTH is dropped from the training
# batch (rLLM's compact filtering, the "overlong filtering" of DAPO) instead of being
# trained on with reward 0: it does not enter its group's mean/std and gets no gradient.
# The rest of its group is trained as usual; a step whose answers are all cut off is
# skipped. Failed episodes (errors, timeouts) are dropped as well. Validation still counts
# a cut-off answer as wrong. MASK_TRUNCATED=false trains on them with reward 0 again.
if [ "${MASK_TRUNCATED:-true}" = "true" ]; then
  DATA+=(
    rllm.compact_filtering.enable=true
    rllm.compact_filtering.mask_max_response_length_exceeded=true
  )
fi

# GRPO: advantage = (reward - group mean) / group std; clipped policy loss with a higher
# upper clip bound; KL to the SFT policy as a loss term (k3 estimator).
ALGORITHM=(
  rllm.algorithm.adv_estimator=grpo
  rllm.algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD:-true}"
  rllm.algorithm.loss_agg_mode=token-mean
  rllm.algorithm.eps_clip="${CLIP_LOW:-0.2}"
  rllm.algorithm.eps_clip_high="${CLIP_HIGH:-0.28}"
  rllm.algorithm.kl_beta="${KL_BETA:-0.001}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
)

# Policy: the full-parameter SFT checkpoint. LORA_RANK > 0 adds a LoRA adapter that GRPO
# trains, and the reference policy is the same weights with the adapter switched off.
# LORA_RANK=0 trains all parameters instead; verl then keeps a separate frozen copy of
# the model as the reference policy. For the 8B model that needs several GPUs (N_GPUS):
# fp32 weights, gradients and Adam state are ~128 GB, sharded across the cards by FSDP.
LORA_RANK="${LORA_RANK:-64}"
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/data/runs/qwen3-8b-sft-v3}"
# A Hugging Face repo id plus POLICY_MODEL_REVISION (a branch, tag or commit, e.g. v3) is
# downloaded once into HF_HOME (on the PVC, so later runs reuse it; private repos need
# HF_TOKEN) and trained from the local snapshot: verl and vLLM take model.path without a
# revision.
if [ -n "${POLICY_MODEL_REVISION:-}" ]; then
  echo "[entrypoint] fetching ${POLICY_MODEL_PATH}@${POLICY_MODEL_REVISION} into ${HF_HOME:-the default HF cache}"
  POLICY_MODEL_PATH="$(python3 -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))' "${POLICY_MODEL_PATH}" "${POLICY_MODEL_REVISION}")"
fi
ACTOR=(
  actor_rollout_ref.model.path="${POLICY_MODEL_PATH}"
  actor_rollout_ref.model.lora_rank="${LORA_RANK}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-5}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-4}"
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}"
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}"
)

# Rollout and training share the GPUs, so the actor has to leave them while vLLM runs.
# Default (FSDP1): weights and optimizer state move to host RAM during rollout and come
# back whole for training, so the update still needs the full shard of weights, gradients
# and Adam state on each GPU (~128 GB / N_GPUS for full-parameter 8B).
# FSDP_CPU_OFFLOAD=true (FSDP2 with CPUOffloadPolicy): weights, gradients and Adam state
# stay in pinned host RAM throughout; each layer's weights are copied to the GPU only
# while it is computed, and the optimizer step runs on the CPU. GPU memory then holds just
# the current layers and activations, so full-parameter 8B fits on 2 cards, at the cost
# of host-device copies every pass and a CPU Adam step (OMP_NUM_THREADS sets its threads;
# Ray workers default to 1). The reference model follows the actor's strategy and is
# CPU-offloaded in both modes.
if [ "${FSDP_CPU_OFFLOAD:-false}" = "true" ]; then
  ACTOR+=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  )
else
  ACTOR+=(
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  )
fi

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
# lets it fit on a 48 GB card. PEFT keeps the adapter itself in fp32.
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
else
  # Full-parameter checkpoints: the fp32 weights alone are ~33 GB for the 8B model, the
  # Adam state another ~66 GB. verl keeps the previous checkpoint until the next one is
  # written, so two full checkpoints (~200 GB) would not fit the 150Gi RL PVC. Without the
  # optimizer a resumed run restarts Adam's moments but keeps the weights, the data
  # position and the step. CKPT_SAVE_OPTIMIZER=true saves it too (needs a larger PVC).
  if [ "${CKPT_SAVE_OPTIMIZER:-false}" != "true" ]; then
    ACTOR+=(actor_rollout_ref.actor.checkpoint.save_contents=[model,extra])
  fi
fi

TRAINER=(
  # The Job requests N_GPUS cards on one node (../run.sh). With N_GPUS > 1, FSDP shards
  # the actor and vLLM runs one replica per GPU.
  trainer.n_gpus_per_node="${N_GPUS:-1}"
  trainer.nnodes=1
  rllm.trainer.total_epochs="${EPOCHS:-1}"
  rllm.trainer.total_batches="${TOTAL_TRAINING_STEPS:--1}"
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
  trainer.max_actor_ckpt_to_keep=1
  hydra.run.dir="${RUN_DIR}/hydra"
)

echo "[entrypoint] run dir ${RUN_DIR}"
echo "[entrypoint] policy ${POLICY_MODEL_PATH}  gpus ${N_GPUS:-1}"

# python3 on PATH, not PYTHON_BIN: PYTHON_BIN is a local path for ../run.sh and also
# reaches the pod through the grpo-env Secret.
exec python3 "${SRC_DIR}/train_grpo.py" \
  "${DATA[@]}" "${ALGORITHM[@]}" "${ACTOR[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" "$@"
