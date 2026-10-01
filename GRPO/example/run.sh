#!/usr/bin/env bash
# Small GRPO example on NRP: Qwen3-0.6B, full-parameter (no LoRA), monitored in wandb.
# Same shape as ../../run.sh and ../../../train/run.sh:  data -> image -> secrets -> pvc -> job -> logs
#
#   ./run.sh              everything, then follow the logs
#   ./run.sh data         only rebuild ../data/{train,validation}.jsonl
#   ./run.sh image        only docker login + build + push (build context: ..)
#   ./run.sh secrets      only refresh the grpo-example-env and nrp-registry Secrets
#   ./run.sh pvc          only create the PVC if it is missing
#   ./run.sh submit       only (re)submit the Job
#   ./run.sh logs         follow the Job's logs
#   ./run.sh status       job, pods and recent events
#   ./run.sh clean        delete the Job (Secrets and the PVC are kept)
#
# Everything is configured through .env next to this file; see .env.example.
set -euo pipefail

cd "$(dirname "$0")"
GRPO_DIR=..

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
  echo "error: .env not found. Run: cp .env.example .env && \$EDITOR .env" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
. ./.env
set +a

: "${IMAGE:?set IMAGE in .env}"
: "${K8S_NAMESPACE:?set K8S_NAMESPACE in .env}"
JOB_NAME="${JOB_NAME:-qwen-grpo-example}"
RL_PVC_NAME="${RL_PVC_NAME:-qwen-grpo-data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BESPOKE_DIR="${BESPOKE_DIR:-../../../train/bespoke-v2}"
SFT_SPLIT_DIR="${SFT_SPLIT_DIR:-../../../train/data_labeled_2}"
NRP_REGISTRY="${IMAGE%%/*}"
# The path segment after the host is the GitLab namespace, which is also the registry
# login user for a personal access token.
NRP_REGISTRY_USER="${NRP_REGISTRY_USER:-$(echo "${IMAGE}" | cut -d/ -f2)}"

KUBECTL=(kubectl --namespace "${K8S_NAMESPACE}")

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# k8s cannot read .env, so the manifests carry ${...} placeholders filled in here.
render() {
  sed -e "s|\${IMAGE}|${IMAGE}|g" \
      -e "s|\${JOB_NAME}|${JOB_NAME}|g" \
      -e "s|\${RL_PVC_NAME}|${RL_PVC_NAME}|g" \
      "$1"
}

# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
step_data() {
  log "Building ${GRPO_DIR}/data/{train,validation}.jsonl from ${BESPOKE_DIR}"
  "${PYTHON_BIN}" "${GRPO_DIR}/src/prepare_data.py" \
    --input-dir "${BESPOKE_DIR}" --split-from "${SFT_SPLIT_DIR}" --output-dir "${GRPO_DIR}/data"
}

step_image() {
  : "${NRP_REGISTRY_TOKEN:?set NRP_REGISTRY_TOKEN in .env}"
  [ -f "${GRPO_DIR}/data/train.jsonl" ] || step_data
  log "docker login ${NRP_REGISTRY} as ${NRP_REGISTRY_USER}"
  printf '%s' "${NRP_REGISTRY_TOKEN}" |
    docker login "${NRP_REGISTRY}" --username "${NRP_REGISTRY_USER}" --password-stdin

  log "Building ${IMAGE}"
  docker build --platform linux/amd64 -t "${IMAGE}" "${GRPO_DIR}"

  log "Pushing ${IMAGE}"
  docker push "${IMAGE}"
}

step_secrets() {
  : "${NRP_REGISTRY_TOKEN:?set NRP_REGISTRY_TOKEN in .env}"
  if [ "${REPORT_TO:-}" = "wandb" ] && [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
    echo "error: REPORT_TO=wandb but WANDB_API_KEY is empty (get one at https://wandb.ai/authorize)" >&2
    exit 1
  fi

  log "Refreshing Secret grpo-example-env (from .env)"
  "${KUBECTL[@]}" delete secret grpo-example-env --ignore-not-found
  "${KUBECTL[@]}" create secret generic grpo-example-env --from-env-file=.env

  # Same name as the SFT and main GRPO pull secret; it holds the same registry credentials.
  log "Refreshing Secret nrp-registry (image pull)"
  "${KUBECTL[@]}" delete secret nrp-registry --ignore-not-found
  "${KUBECTL[@]}" create secret docker-registry nrp-registry \
    --docker-server="${NRP_REGISTRY}" \
    --docker-username="${NRP_REGISTRY_USER}" \
    --docker-password="${NRP_REGISTRY_TOKEN}"
}

step_pvc() {
  if "${KUBECTL[@]}" get pvc "${RL_PVC_NAME}" >/dev/null 2>&1; then
    log "PVC ${RL_PVC_NAME} already exists"
  else
    log "Creating PVC ${RL_PVC_NAME}"
    render "${GRPO_DIR}/k8s/pvc.yaml" | "${KUBECTL[@]}" apply -f -
  fi
}

step_submit() {
  log "Submitting Job ${JOB_NAME}"
  # A Job's pod template is immutable, so an existing Job has to go first.
  "${KUBECTL[@]}" delete job "${JOB_NAME}" --ignore-not-found
  render k8s/job.yaml | "${KUBECTL[@]}" apply -f -
}

job_pod() {
  "${KUBECTL[@]}" get pods -l "job-name=${JOB_NAME}" \
    -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true
}

step_logs() {
  local pod="" last="" phase reason msg
  log "Waiting for the ${JOB_NAME} pod to start"
  for i in $(seq 1 120); do
    pod="$(job_pod)"
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
  "${KUBECTL[@]}" get job "${JOB_NAME}" || true
  "${KUBECTL[@]}" get pods -l "job-name=${JOB_NAME}" || true
  "${KUBECTL[@]}" get events --sort-by=.lastTimestamp | tail -15 || true
}

step_clean() {
  log "Deleting Job ${JOB_NAME}"
  "${KUBECTL[@]}" delete job "${JOB_NAME}" --ignore-not-found
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
case "${1:-all}" in
  data) step_data ;;
  image) step_image ;;
  secrets) step_secrets ;;
  pvc) step_pvc ;;
  submit) step_submit ;;
  logs) step_logs ;;
  status) step_status ;;
  clean) step_clean ;;
  all)
    [ -f "${GRPO_DIR}/data/train.jsonl" ] || step_data
    step_image
    step_secrets
    step_pvc
    step_submit
    step_logs
    ;;
  *)
    echo "usage: $0 [all|data|image|secrets|pvc|submit|logs|status|clean]" >&2
    exit 1
    ;;
esac
