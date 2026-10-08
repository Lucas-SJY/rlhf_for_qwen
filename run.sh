#!/usr/bin/env bash
# Entry point of the project: GRPO on the SFT checkpoint, as one Job on NRP.
# Same shape as ../train/run.sh:  data -> image -> secrets -> pvc -> job -> logs
#
#   ./run.sh              everything, then follow the logs
#   ./run.sh data         only rebuild GRPO/data/{train,validation}.jsonl
#   ./run.sh test         run the reward unit tests locally (stdlib only)
#   ./run.sh image        only docker login + build + push (build context: GRPO/)
#   ./run.sh secrets      only refresh the grpo-env and nrp-registry Secrets
#   ./run.sh pvc          only create the GRPO PVC if it is missing
#   ./run.sh submit       only (re)submit the training Job
#   ./run.sh logs         follow the training Job's logs
#   ./run.sh status       jobs, pods and recent events
#   ./run.sh export       convert the latest checkpoint to a HF model (CPU Job) and follow it
#   ./run.sh clean        delete the training and export Jobs (Secrets and PVCs are kept)
#
# Everything is configured through .env next to this file; see .env.example.
set -euo pipefail

cd "$(dirname "$0")"
APP_DIR=GRPO

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
# `data` and `test` work without .env; every cluster step needs it.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

JOB_NAME="${JOB_NAME:-qwen-grpo}"
SFT_PVC_NAME="${SFT_PVC_NAME:-qwen-sft-data}"
RL_PVC_NAME="${RL_PVC_NAME:-qwen-grpo-data}"
# Training pod size; the defaults fit the 8B LoRA run on one GPU.
POD_CPU="${POD_CPU:-8}"
POD_MEMORY="${POD_MEMORY:-160Gi}"
# Cards on the one node the pod runs on (the entrypoint reads the same N_GPUS).
N_GPUS="${N_GPUS:-1}"
# GPU_TYPE picks the card. Without GPU_PRIORITY_CLASS in .env, a100 runs at priority
# "opportunistic", which bypasses the GPU quota but can be preempted at any time; set
# GPU_PRIORITY_CLASS= (empty) to run at normal priority within the namespace's A100 quota.
GPU_TYPE="${GPU_TYPE:-l40}"
case "${GPU_TYPE}" in
  l40) GPU_RESOURCE=nvidia.com/gpu GPU_PRODUCTS="[NVIDIA-L40, NVIDIA-L40S]" ;;
  a6000) GPU_RESOURCE=nvidia.com/rtxa6000 GPU_PRODUCTS="[NVIDIA-RTX-A6000]" ;;
  a100)
    GPU_RESOURCE=nvidia.com/a100 GPU_PRODUCTS="[NVIDIA-A100-SXM4-80GB, NVIDIA-A100-80GB-PCIe]"
    GPU_PRIORITY_CLASS="${GPU_PRIORITY_CLASS-opportunistic}"
    ;;
  *)
    echo "error: GPU_TYPE must be l40, a6000 or a100 (got '${GPU_TYPE}')" >&2
    exit 1
    ;;
esac
GPU_PRIORITY_CLASS="${GPU_PRIORITY_CLASS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BESPOKE_DIR="${BESPOKE_DIR:-../train/bespoke-v2}"
SFT_SPLIT_DIR="${SFT_SPLIT_DIR:-../train/data_labeled_2}"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

cluster_config() {
  if [ ! -f .env ]; then
    echo "error: .env not found. Run: cp .env.example .env && \$EDITOR .env" >&2
    exit 1
  fi
  : "${IMAGE:?set IMAGE in .env}"
  : "${K8S_NAMESPACE:?set K8S_NAMESPACE in .env}"
  NRP_REGISTRY="${IMAGE%%/*}"
  # The path segment after the host is the GitLab namespace, which is also the registry
  # login user for a personal access token.
  NRP_REGISTRY_USER="${NRP_REGISTRY_USER:-$(echo "${IMAGE}" | cut -d/ -f2)}"
  KUBECTL=(kubectl --namespace "${K8S_NAMESPACE}")
}

# k8s cannot read .env, so the manifests carry ${...} placeholders filled in here.
render() {
  sed -e "s|\${IMAGE}|${IMAGE}|g" \
      -e "s|\${JOB_NAME}|${JOB_NAME}|g" \
      -e "s|\${SFT_PVC_NAME}|${SFT_PVC_NAME}|g" \
      -e "s|\${RL_PVC_NAME}|${RL_PVC_NAME}|g" \
      -e "s|\${POD_CPU}|${POD_CPU}|g" \
      -e "s|\${POD_MEMORY}|${POD_MEMORY}|g" \
      -e "s|\${GPU_RESOURCE}|${GPU_RESOURCE}|g" \
      -e "s|\${N_GPUS}|${N_GPUS}|g" \
      -e "s|\${GPU_PRODUCTS}|${GPU_PRODUCTS}|g" \
      -e "s|\${GPU_PRIORITY_CLASS}|${GPU_PRIORITY_CLASS}|g" \
      "$1"
}

# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
step_data() {
  log "Building ${APP_DIR}/data/{train,validation}.jsonl from ${BESPOKE_DIR}"
  "${PYTHON_BIN}" "${APP_DIR}/src/prepare_data.py" \
    --input-dir "${BESPOKE_DIR}" --split-from "${SFT_SPLIT_DIR}" --output-dir "${APP_DIR}/data"
}

step_test() {
  log "Reward unit tests"
  "${PYTHON_BIN}" -m unittest discover -s "${APP_DIR}/tests" -v
}

step_image() {
  : "${NRP_REGISTRY_TOKEN:?set NRP_REGISTRY_TOKEN in .env}"
  [ -f "${APP_DIR}/data/train.jsonl" ] || step_data
  log "docker login ${NRP_REGISTRY} as ${NRP_REGISTRY_USER}"
  printf '%s' "${NRP_REGISTRY_TOKEN}" |
    docker login "${NRP_REGISTRY}" --username "${NRP_REGISTRY_USER}" --password-stdin

  log "Building ${IMAGE}"
  docker build --platform linux/amd64 -t "${IMAGE}" "${APP_DIR}"

  log "Pushing ${IMAGE}"
  docker push "${IMAGE}"
}

step_secrets() {
  : "${NRP_REGISTRY_TOKEN:?set NRP_REGISTRY_TOKEN in .env}"

  log "Refreshing Secret grpo-env (from .env)"
  "${KUBECTL[@]}" delete secret grpo-env --ignore-not-found
  "${KUBECTL[@]}" create secret generic grpo-env --from-env-file=.env

  # Same name as the SFT project's pull secret; it holds the same registry credentials.
  log "Refreshing Secret nrp-registry (image pull)"
  "${KUBECTL[@]}" delete secret nrp-registry --ignore-not-found
  "${KUBECTL[@]}" create secret docker-registry nrp-registry \
    --docker-server="${NRP_REGISTRY}" \
    --docker-username="${NRP_REGISTRY_USER}" \
    --docker-password="${NRP_REGISTRY_TOKEN}"
}

step_pvc() {
  if ! "${KUBECTL[@]}" get pvc "${SFT_PVC_NAME}" >/dev/null 2>&1; then
    echo "error: SFT PVC ${SFT_PVC_NAME} not found; it must hold ${POLICY_MODEL_PATH:-the SFT checkpoint}" >&2
    exit 1
  fi
  if "${KUBECTL[@]}" get pvc "${RL_PVC_NAME}" >/dev/null 2>&1; then
    log "PVC ${RL_PVC_NAME} already exists"
  else
    log "Creating PVC ${RL_PVC_NAME}"
    render "${APP_DIR}/k8s/pvc.yaml" | "${KUBECTL[@]}" apply -f -
  fi
}

step_submit() {
  log "Submitting Job ${JOB_NAME}"
  # A Job's pod template is immutable, so an existing Job has to go first.
  "${KUBECTL[@]}" delete job "${JOB_NAME}" --ignore-not-found
  render "${APP_DIR}/k8s/job.yaml" | "${KUBECTL[@]}" apply -f -
}

job_pod() {
  "${KUBECTL[@]}" get pods -l "job-name=$1" \
    -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true
}

follow_job() {
  local job="$1" pod="" last="" phase reason msg
  log "Waiting for the ${job} pod to start"
  for i in $(seq 1 120); do
    pod="$(job_pod "${job}")"
    if [ -n "${pod}" ]; then
      phase="$("${KUBECTL[@]}" get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
      case "${phase}" in
        Running | Succeeded | Failed) break ;;
      esac
      # Pending is where NRP jobs get stuck (no free GPU, quota, PVC attach).
      reason="$("${KUBECTL[@]}" get pod "${pod}" \
        -o jsonpath='{.status.conditions[?(@.type=="PodScheduled")].message}' 2>/dev/null || true)"
      [ -z "${reason}" ] && reason="${phase}"
      msg="${phase}: $(printf '%s' "${reason}" | cut -c1-150)"
      if [ "${msg}" != "${last}" ]; then
        printf '\n  [%3ds] %s\n' "$((i * 5))" "${msg}"
        last="${msg}"
      else
        printf '.'
      fi
    fi
    sleep 5
  done
  echo
  if [ -z "${pod}" ]; then
    echo "no pod created; check './run.sh status'" >&2
    exit 1
  fi
  phase="$("${KUBECTL[@]}" get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  if [ "${phase}" = "Pending" ]; then
    echo "still Pending after 10 minutes. Full reason:" >&2
    "${KUBECTL[@]}" describe pod "${pod}" | sed -n '/Events:/,$p' >&2
    exit 1
  fi
  "${KUBECTL[@]}" logs -f "${pod}"
}

step_status() {
  "${KUBECTL[@]}" get job "${JOB_NAME}" "${JOB_NAME}-export" 2>/dev/null || true
  "${KUBECTL[@]}" get pods -l "job-name in (${JOB_NAME},${JOB_NAME}-export)" || true
  "${KUBECTL[@]}" get events --sort-by=.lastTimestamp | tail -15 || true
}

step_export() {
  log "Submitting Job ${JOB_NAME}-export"
  "${KUBECTL[@]}" delete job "${JOB_NAME}-export" --ignore-not-found
  render "${APP_DIR}/k8s/export-job.yaml" | "${KUBECTL[@]}" apply -f -
  follow_job "${JOB_NAME}-export"
}

step_clean() {
  log "Deleting Jobs ${JOB_NAME} and ${JOB_NAME}-export"
  "${KUBECTL[@]}" delete job "${JOB_NAME}" "${JOB_NAME}-export" --ignore-not-found
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
case "${1:-all}" in
  data) step_data ;;
  test) step_test ;;
  image) cluster_config; step_image ;;
  secrets) cluster_config; step_secrets ;;
  pvc) cluster_config; step_pvc ;;
  submit) cluster_config; step_submit ;;
  logs) cluster_config; follow_job "${JOB_NAME}" ;;
  status) cluster_config; step_status ;;
  export) cluster_config; step_export ;;
  clean) cluster_config; step_clean ;;
  all)
    cluster_config
    [ -f "${APP_DIR}/data/train.jsonl" ] || step_data
    step_image
    step_secrets
    step_pvc
    step_submit
    follow_job "${JOB_NAME}"
    ;;
  *)
    echo "usage: $0 [all|data|test|image|secrets|pvc|submit|logs|status|export|clean]" >&2
    exit 1
    ;;
esac
