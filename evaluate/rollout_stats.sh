#!/usr/bin/env bash
# Per-question outcome of a GRPO run's training rollouts: for every step and question,
# how many of its GROUP_SIZE rollouts were correct and how many were cut off.
#
#   ./evaluate/rollout_stats.sh                         RUN_NAME from ../.env
#   STATS_RUN_NAME=<run> ./evaluate/rollout_stats.sh    another run on the RL PVC
#   STATS_MODE=val ./evaluate/rollout_stats.sh          the validation rollouts instead
#
# rLLM logs every rollout as its own JSON file under /grpo/runs/<run>/episodes/episodes/
# on the RL PVC (25,600 files with the full responses for a 200-step V1 run), so the
# counting runs next to the data: a small python pod mounts the PVC read-only and runs
# rollout_stats.py (from a ConfigMap), and only its summary, one row per step and
# question, is copied to ../eval-results/rollout-stats/<run>-<mode>.json (gitignored).
#
# Reads ../.env for K8S_NAMESPACE, RUN_NAME and RL_PVC_NAME but never writes to it.
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

: "${K8S_NAMESPACE:?set K8S_NAMESPACE in .env}"
RL_PVC_NAME="${RL_PVC_NAME:-qwen-grpo-data}"
STATS_RUN_NAME="${STATS_RUN_NAME:-${RUN_NAME:?set RUN_NAME in .env}}"
STATS_MODE="${STATS_MODE:-train}"
POD="qwen-grpo-rollout-stats"
CONFIGMAP="grpo-rollout-stats-code"
EPISODES="/grpo/runs/${STATS_RUN_NAME}/episodes/episodes"
OUT_DIR="${REPO_ROOT}/eval-results/rollout-stats"
OUT="${OUT_DIR}/${STATS_RUN_NAME}-${STATS_MODE}.json"
KUBECTL=(kubectl --namespace "${K8S_NAMESPACE}")

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

cleanup() { "${KUBECTL[@]}" delete pod "${POD}" --ignore-not-found --wait=false >/dev/null 2>&1 || true; }
trap cleanup EXIT

log "Refreshing ConfigMap ${CONFIGMAP}"
"${KUBECTL[@]}" delete configmap "${CONFIGMAP}" --ignore-not-found >/dev/null
"${KUBECTL[@]}" create configmap "${CONFIGMAP}" --from-file=rollout_stats.py

# The RL PVC is ReadWriteOnce: while another pod (training, an answer Job) holds it, a
# second pod can only mount it on the same node, so pin the pod there.
holder_node="$("${KUBECTL[@]}" get pods --field-selector=status.phase=Running \
  -o jsonpath='{range .items[*]}{.spec.nodeName}{" "}{.spec.volumes[*].persistentVolumeClaim.claimName}{"\n"}{end}' |
  awk -v pvc="${RL_PVC_NAME}" '{for (i = 2; i <= NF; i++) if ($i == pvc) {print $1; exit}}')"
affinity=""
if [ -n "${holder_node}" ]; then
  echo "PVC ${RL_PVC_NAME} is mounted on ${holder_node}; scheduling the pod there"
  affinity="affinity: {nodeAffinity: {requiredDuringSchedulingIgnoredDuringExecution: {nodeSelectorTerms: [{matchExpressions: [{key: kubernetes.io/hostname, operator: In, values: [${holder_node}]}]}]}}}"
fi

log "Starting pod ${POD}"
"${KUBECTL[@]}" delete pod "${POD}" --ignore-not-found >/dev/null
"${KUBECTL[@]}" apply -f - >/dev/null <<POD
apiVersion: v1
kind: Pod
metadata:
  name: ${POD}
spec:
  restartPolicy: Never
  ${affinity}
  containers:
    - name: stats
      image: python:3.12-alpine
      command: ["sleep", "3600"]
      # NRP allows limits of at most 1.2 x the requests.
      resources:
        requests: {cpu: 500m, memory: 1Gi}
        limits: {cpu: 600m, memory: 1200Mi}
      volumeMounts:
        - {name: grpo, mountPath: /grpo, readOnly: true}
        - {name: code, mountPath: /code, readOnly: true}
  volumes:
    - name: grpo
      persistentVolumeClaim: {claimName: ${RL_PVC_NAME}, readOnly: true}
    - name: code
      configMap: {name: ${CONFIGMAP}}
POD
"${KUBECTL[@]}" wait --for=condition=Ready "pod/${POD}" --timeout=900s

log "Counting ${STATS_MODE} rollouts in ${EPISODES}"
"${KUBECTL[@]}" exec "${POD}" -- python3 /code/rollout_stats.py \
  --episodes "${EPISODES}" --mode "${STATS_MODE}" --run "${STATS_RUN_NAME}" --output /tmp/rollout_stats.json

mkdir -p "${OUT_DIR}"
"${KUBECTL[@]}" exec "${POD}" -- cat /tmp/rollout_stats.json > "${OUT}.tmp"
mv "${OUT}.tmp" "${OUT}"
echo "summary in ${OUT}"
