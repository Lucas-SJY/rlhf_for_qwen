#!/usr/bin/env bash
# Download every rollout trajectory a GRPO run has written so far, training and validation,
# from the RL PVC to ../eval-results/trajectories/<run>/ (gitignored).
#
#   ./evaluate/sync_trajectories.sh            RUN_NAME from ../.env
#   ./evaluate/sync_trajectories.sh <run>      another run under /grpo/runs on the RL PVC
#
# rLLM writes each step's rollouts as one JSON file per rollout (the full response, reward
# breakdown and metrics) to /grpo/runs/<run>/episodes/episodes/{train,val}_step_<N>_epoch_<E>/
# and writes batch_summary.json last. The script takes whatever is there now: every step
# directory not yet complete here is copied, including one the run is still writing (it is
# copied again on the next call, until its batch_summary.json has arrived). Complete
# directories already here are skipped, so it can be run again at any time, during or after
# training. A small busybox pod mounts the PVC read-only and streams the files out with tar;
# it is deleted at the end.
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
RUN="${1:-${RUN_NAME:?set RUN_NAME in .env or pass a run name}}"
POD="qwen-grpo-trajectory-sync"
EPISODES="/grpo/runs/${RUN}/episodes/episodes"
LOCAL="${REPO_ROOT}/eval-results/trajectories/${RUN}"
BATCH=10
KUBECTL=(kubectl --namespace "${K8S_NAMESPACE}")

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

cleanup() { "${KUBECTL[@]}" delete pod "${POD}" --ignore-not-found --wait=false >/dev/null 2>&1 || true; }
trap cleanup EXIT

# The RL PVC is ReadWriteOnce: while another pod (training, an eval Job) holds it, a second
# pod can only mount it on the same node, so pin the pod there.
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
    - name: sync
      image: busybox:1.36
      command: ["sleep", "3600"]
      # NRP allows limits of at most 1.2 x the requests.
      resources:
        requests: {cpu: 200m, memory: 256Mi}
        limits: {cpu: 240m, memory: 300Mi}
      volumeMounts:
        - {name: grpo, mountPath: /grpo, readOnly: true}
  volumes:
    - name: grpo
      persistentVolumeClaim: {claimName: ${RL_PVC_NAME}, readOnly: true}
POD
"${KUBECTL[@]}" wait --for=condition=Ready "pod/${POD}" --timeout=900s

if ! "${KUBECTL[@]}" exec "${POD}" -- test -d "${EPISODES}"; then
  echo "error: ${EPISODES} not found; runs on the PVC:" >&2
  "${KUBECTL[@]}" exec "${POD}" -- ls /grpo/runs >&2 || true
  exit 1
fi

# One line per step directory: its name, "complete" or "partial", and its rollout files.
log "Listing ${EPISODES}"
listing="$("${KUBECTL[@]}" exec "${POD}" -- sh -c "cd '${EPISODES}' && for d in *_step_*; do
  [ -d \"\$d\" ] || continue
  if [ -f \"\$d/batch_summary.json\" ]; then s=complete; else s=partial; fi
  echo \"\$d \$s \$(ls \"\$d\" | grep -c '^episode_')\"
done")"
if [ -z "${listing}" ]; then
  echo "no step directories in ${EPISODES} yet"
  exit 0
fi

mkdir -p "${LOCAL}"
todo=()
partial=0
while read -r dir state files; do
  [ -n "${dir}" ] || continue
  # Complete here already: nothing can have changed.
  [ -f "${LOCAL}/${dir}/batch_summary.json" ] && continue
  todo+=("${dir}")
  [ "${state}" = partial ] && partial=$((partial + 1))
done <<< "${listing}"

total_dirs="$(printf '%s\n' "${listing}" | grep -c .)"
remote_files="$(printf '%s\n' "${listing}" | awk '{n += $3} END {print n + 0}')"
echo "on the PVC: ${total_dirs} step directories, ${remote_files} rollout files"
if [ "${#todo[@]}" -eq 0 ]; then
  echo "everything is here already: ${LOCAL} ($(du -sh "${LOCAL}" | cut -f1))"
  exit 0
fi

log "Copying ${#todo[@]} step directories (${partial} still being written) to ${LOCAL}"
for ((i = 0; i < ${#todo[@]}; i += BATCH)); do
  part=("${todo[@]:i:BATCH}")
  "${KUBECTL[@]}" exec "${POD}" -- tar cf - -C "${EPISODES}" "${part[@]}" | tar xf - -C "${LOCAL}"
  echo "  $((i + ${#part[@]})) / ${#todo[@]}  (${part[0]} .. ${part[${#part[@]} - 1]})"
done

local_files="$(find "${LOCAL}" -name 'episode_*.json' | wc -l | tr -d ' ')"
echo "done: ${local_files} rollout files in ${LOCAL} ($(du -sh "${LOCAL}" | cut -f1))"
if [ "${partial}" -gt 0 ]; then
  echo "${partial} step director$([ "${partial}" -eq 1 ] && echo y || echo ies) were still being written; run the script again later for the rest"
fi
