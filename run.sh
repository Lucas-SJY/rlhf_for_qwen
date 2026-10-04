#!/usr/bin/env bash
# Entry point of the project: GRPO on the SFT checkpoint, as one Job on NRP.
# Same shape as ../train/run.sh:  data -> image -> secrets -> pvc -> job -> logs
#
#   ./run.sh              everything, then follow the logs
#   ./run.sh data         only rebuild GRPO/data/{train,validation}.jsonl
#   ./run.sh test         CPU tests (optional math-verify/torch extend coverage)
#   ./run.sh preflight    validate local data and settings
#   ./run.sh diagnose     scripted rewards, advantages, optional CPU gradient update
#   ./run.sh train        train in an installed GPU environment (extra Hydra args allowed)
#   ./run.sh image        only docker login + build + push (build context: GRPO/)
#   ./run.sh secrets      apply this experiment's scoped Secrets
#   ./run.sh pvc          only create the GRPO PVC if it is missing
#   ./run.sh submit       create a new Job; refuses an existing name
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
  # Plain KEY=value only; process environment wins over .env (same as entrypoint).
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in '' | '#'*) continue ;; esac
    key="${line%%=*}"
    case "$key" in '' | *[!A-Za-z0-9_]*) echo "invalid .env key" >&2; exit 1 ;; esac
    [ -n "${!key+set}" ] || export "$key=${line#*=}"
  done < .env
fi

export JOB_NAME="${JOB_NAME:-qwen-grpo-dayallen}"
export SFT_PVC_NAME="${SFT_PVC_NAME:-qwen-sft-data}"
export RL_PVC_NAME="${RL_PVC_NAME:-qwen-grpo-dayallen-data}"
export ENV_SECRET_NAME="${ENV_SECRET_NAME:-grpo-dayallen-env}"
export REGISTRY_SECRET_NAME="${REGISTRY_SECRET_NAME:-grpo-dayallen-registry}"
# Training pod size; the defaults fit the 8B run.
export POD_CPU="${POD_CPU:-8}"
export POD_MEMORY="${POD_MEMORY:-160Gi}"
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
  "${PYTHON_BIN}" "${APP_DIR}/src/cluster.py" render "$1"
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
  log "GRPO harness tests"
  "${PYTHON_BIN}" -m unittest discover -s "${APP_DIR}/tests" -v
}

step_image() {
  : "${NRP_REGISTRY_TOKEN:?set NRP_REGISTRY_TOKEN in .env}"
  if [ -z "${TRAIN_FILE:-}" ] && [ ! -f "${APP_DIR}/data/train.jsonl" ]; then step_data; fi
  step_preflight
  log "docker login ${NRP_REGISTRY} as ${NRP_REGISTRY_USER}"
  printf '%s' "${NRP_REGISTRY_TOKEN}" |
    docker login "${NRP_REGISTRY}" --username "${NRP_REGISTRY_USER}" --password-stdin

  log "Building ${IMAGE}"
  docker build --platform linux/amd64 -t "${IMAGE}" "${APP_DIR}"

  log "Pushing ${IMAGE}"
  docker push "${IMAGE}"
}

step_secrets() {
  "${PYTHON_BIN}" "${APP_DIR}/src/cluster.py" secrets
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

ensure_new_job() {
  local existing
  existing="$("${KUBECTL[@]}" get job "$1" --ignore-not-found -o name)"
  if [ -n "$existing" ]; then
    echo "Job $1 already exists. Inspect './run.sh status'; clean it explicitly or choose a new JOB_NAME." >&2
    exit 1
  fi
}

step_submit() {
  [ "${N_GPUS:-1}" = "1" ] || { echo "this Kubernetes manifest supports N_GPUS=1 only" >&2; exit 1; }
  ensure_new_job "${JOB_NAME}"
  log "Submitting Job ${JOB_NAME}"
  render "${APP_DIR}/k8s/job.yaml" | "${KUBECTL[@]}" create -f -
}

follow_job() {
  "${PYTHON_BIN}" "${APP_DIR}/src/cluster.py" logs "$1"
}

step_preflight() {
  # Image paths map into the local Docker build context. Arbitrary absolute remote
  # paths must be supplied explicitly using LOCAL_TRAIN_FILE / LOCAL_VAL_FILE.
  local train_file="${LOCAL_TRAIN_FILE:-${TRAIN_FILE:-GRPO/data/train.jsonl}}"
  local val_file="${LOCAL_VAL_FILE:-${VAL_FILE:-GRPO/data/validation.jsonl}}"
  case "$train_file" in /workspace/data/*) train_file="GRPO/data/${train_file#/workspace/data/}" ;; esac
  case "$val_file" in /workspace/data/*) val_file="GRPO/data/${val_file#/workspace/data/}" ;; esac
  "${PYTHON_BIN}" "${APP_DIR}/src/preflight.py" --train-file "$train_file" --val-file "$val_file"
}

step_status() {
  "${KUBECTL[@]}" get job "${JOB_NAME}" "${JOB_NAME}-export" 2>/dev/null || true
  "${KUBECTL[@]}" get pods -l "job-name in (${JOB_NAME},${JOB_NAME}-export)" || true
  "${KUBECTL[@]}" get events --sort-by=.lastTimestamp | tail -15 || true
}

step_export() {
  log "Submitting Job ${JOB_NAME}-export"
  ensure_new_job "${JOB_NAME}-export"
  render "${APP_DIR}/k8s/export-job.yaml" | "${KUBECTL[@]}" create -f -
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
  preflight) step_preflight ;;
  diagnose) "${PYTHON_BIN}" "${APP_DIR}/src/diagnose.py" ;;
  train) bash "${APP_DIR}/src/entrypoint.sh" "${@:2}" ;;
  render) cluster_config; render "${APP_DIR}/k8s/job.yaml" ;;
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
    ensure_new_job "${JOB_NAME}"
    step_image
    step_secrets
    step_pvc
    step_submit
    follow_job "${JOB_NAME}"
    ;;
  *)
    echo "usage: $0 [all|data|test|preflight|diagnose|train|render|image|secrets|pvc|submit|logs|status|export|clean]" >&2
    exit 1
    ;;
esac
