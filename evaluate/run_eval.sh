#!/usr/bin/env bash
# MATH-500 evaluation of a GRPO checkpoint on NRP, driven like ../../train/evaluate.
#
#   ./evaluate/run_eval.sh              code -> job -> logs
#   ./evaluate/run_eval.sh code         only refresh the grpo-eval-code ConfigMap
#   ./evaluate/run_eval.sh submit       only (re)submit the Job
#   ./evaluate/run_eval.sh logs         follow the current job's logs
#   ./evaluate/run_eval.sh status       job, pod, recent events
#   ./evaluate/run_eval.sh fetch        copy results out of the RL PVC to ./eval-results
#   ./evaluate/run_eval.sh clean        delete the Job (results and PVC survive)
#
# Which model: EVAL_MODEL, a HF model directory on the RL PVC, by default the export of
# RUN_NAME at the final step (../run.sh export writes it). For another run or step:
#   EVAL_MODEL=/grpo/runs/<run>/export/global_step_<N> ./evaluate/run_eval.sh
# EVAL_RUN_NAME names the result directory (default <run>-global_step_<N>), and
# EVAL_LIMIT=20 makes a smoke test of the first 20 problems.
#
# Reads ../.env for IMAGE, K8S_NAMESPACE, RUN_NAME and RL_PVC_NAME but never writes to
# it. No image rebuild: the script goes in as a ConfigMap, into the SFT training image
# that ../../train/evaluate uses, so the run matches the SFT baseline's environment.
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"

if [ ! -f "${REPO_ROOT}/.env" ]; then
  echo "error: ${REPO_ROOT}/.env not found" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "${REPO_ROOT}/.env"
set +a

: "${IMAGE:?set IMAGE in .env}"
: "${K8S_NAMESPACE:?set K8S_NAMESPACE in .env}"
JOB_NAME="qwen-grpo-eval"
CONFIGMAP="grpo-eval-code"
RL_PVC_NAME="${RL_PVC_NAME:-qwen-grpo-data}"
# The SFT image sits next to the GRPO one in the same registry project.
EVAL_IMAGE="${EVAL_IMAGE:-${IMAGE%/*}/context-comp-sft:latest}"
EVAL_MODEL="${EVAL_MODEL:-/grpo/runs/${RUN_NAME:?set RUN_NAME in .env}/export/global_step_${TOTAL_TRAINING_STEPS:-200}}"
if [ -z "${EVAL_RUN_NAME:-}" ]; then
  # /grpo/runs/<run>/export/global_step_<N> -> <run>-global_step_<N>; the script's own
  # default (the directory name) would make every run's results collide.
  EVAL_RUN_NAME="$(printf '%s' "${EVAL_MODEL%/}" | sed -E 's#^.*/runs/([^/]+)/export/(global_step_[0-9]+)$#\1-\2#; s#^.*/##')"
fi
EVAL_LIMIT="${EVAL_LIMIT:-0}"
KUBECTL=(kubectl --namespace "${K8S_NAMESPACE}")

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step_code() {
  log "Refreshing ConfigMap ${CONFIGMAP}"
  "${KUBECTL[@]}" delete configmap "${CONFIGMAP}" --ignore-not-found
  "${KUBECTL[@]}" create configmap "${CONFIGMAP}" --from-file=eval_math500.py
}

step_submit() {
  log "Submitting Job ${JOB_NAME}: ${EVAL_MODEL} -> /grpo/eval/${EVAL_RUN_NAME}"
  "${KUBECTL[@]}" delete job "${JOB_NAME}" --ignore-not-found
  sed -e "s|\${IMAGE}|${EVAL_IMAGE}|g" \
      -e "s|\${EVAL_MODEL}|${EVAL_MODEL}|g" \
      -e "s|\${EVAL_RUN_NAME}|${EVAL_RUN_NAME}|g" \
      -e "s|\${EVAL_LIMIT}|${EVAL_LIMIT}|g" \
      -e "s|\${RL_PVC_NAME}|${RL_PVC_NAME}|g" \
      k8s/eval-job.yaml | "${KUBECTL[@]}" apply -f -
}

eval_pod() {
  "${KUBECTL[@]}" get pods -l "job-name=${JOB_NAME}" \
    -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true
}

step_logs() {
  log "Waiting for the ${JOB_NAME} pod"
  pod=""; last=""
  for i in $(seq 1 120); do
    pod="$(eval_pod)"
    if [ -n "${pod}" ]; then
      phase="$("${KUBECTL[@]}" get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
      case "${phase}" in Running | Succeeded | Failed) break ;; esac
      reason="$("${KUBECTL[@]}" get pod "${pod}" \
        -o jsonpath='{.status.conditions[?(@.type=="PodScheduled")].message}' 2>/dev/null || true)"
      msg="${phase}: $(printf '%s' "${reason:-$phase}" | cut -c1-150)"
      if [ "${msg}" != "${last}" ]; then printf '\n  [%3ds] %s\n' "$((i * 5))" "${msg}"; last="${msg}"
      else printf '.'; fi
    fi
    sleep 5
  done
  echo
  [ -n "${pod}" ] || { echo "no pod created; try './run_eval.sh status'" >&2; exit 1; }
  "${KUBECTL[@]}" logs -f "${pod}"
}

step_status() {
  "${KUBECTL[@]}" get job "${JOB_NAME}" || true
  "${KUBECTL[@]}" get pods -l "job-name=${JOB_NAME}" || true
  "${KUBECTL[@]}" get events --sort-by=.lastTimestamp | tail -10 || true
}

# A throwaway busybox pod mounts the RL PVC read-only so the results can be streamed out.
step_fetch() {
  local pod="grpo-eval-fetch"
  log "Copying /grpo/eval out of PVC ${RL_PVC_NAME}"
  "${KUBECTL[@]}" delete pod "${pod}" --ignore-not-found >/dev/null
  "${KUBECTL[@]}" apply -f - >/dev/null <<POD
apiVersion: v1
kind: Pod
metadata:
  name: ${pod}
spec:
  restartPolicy: Never
  containers:
    - name: fetch
      image: busybox:1.36
      command: ["sleep", "900"]
      resources:
        requests: {cpu: 100m, memory: 128Mi}
        limits: {cpu: 100m, memory: 128Mi}
      volumeMounts:
        - {name: grpo, mountPath: /grpo, readOnly: true}
  volumes:
    - name: grpo
      persistentVolumeClaim: {claimName: ${RL_PVC_NAME}}
POD
  "${KUBECTL[@]}" wait --for=condition=Ready "pod/${pod}" --timeout=600s
  mkdir -p "${REPO_ROOT}/eval-results"
  "${KUBECTL[@]}" exec "${pod}" -- tar cf - -C /grpo/eval . | tar xf - -C "${REPO_ROOT}/eval-results"
  "${KUBECTL[@]}" delete pod "${pod}" --wait=false >/dev/null
  echo "results in ${REPO_ROOT}/eval-results"
}

step_clean() {
  log "Deleting Job ${JOB_NAME}"
  "${KUBECTL[@]}" delete job "${JOB_NAME}" --ignore-not-found
}

case "${1:-all}" in
  code) step_code ;;
  submit) step_submit ;;
  logs) step_logs ;;
  status) step_status ;;
  fetch) step_fetch ;;
  clean) step_clean ;;
  all) step_code; step_submit; step_logs ;;
  *) echo "usage: $0 [all|code|submit|logs|status|fetch|clean]" >&2; exit 1 ;;
esac
