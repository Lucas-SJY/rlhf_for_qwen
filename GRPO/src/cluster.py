#!/usr/bin/env python3
"""Render manifests, publish scoped secrets, and follow Kubernetes Job completion."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from labelcot.config import FIELDS

RUNTIME_KEYS = {spec[0] for spec in FIELDS.values()} | {
    "RUN_NAME", "OUTPUT_ROOT", "SMOKE_TEST", "TRAIN_FILE", "VAL_FILE", "HF_TOKEN", "HF_HOME",
    "HF_DATASETS_CACHE", "ACTOR_MODEL_DTYPE", "MAX_LABEL_RETRIES", "MAX_ZERO_SIGNAL_STEPS",
    "REWARD_W_ALIGN", "REWARD_W_CORRECT", "NORM_ADV_BY_STD", "SEED", "VAL_BEFORE_TRAIN",
    "REPORT_TO", "WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_MODE",
    "RAY_OBJECT_STORE_GB", "RAY_NUM_CPUS", "NCCL_DEBUG", "TOKENIZERS_PARALLELISM", "EXPORT_STEP",
}
PLACEHOLDERS = {"IMAGE", "JOB_NAME", "SFT_PVC_NAME", "RL_PVC_NAME", "POD_CPU", "POD_MEMORY",
                "ENV_SECRET_NAME", "REGISTRY_SECRET_NAME"}


def render(path: Path, env) -> str:
    def replace(match):
        name = match[1]
        if name not in PLACEHOLDERS or not env.get(name):
            raise ValueError(f"missing/unsupported manifest variable: {name}")
        value = env[name]
        if not re.fullmatch(r"[A-Za-z0-9_./:@+-]+", value):
            raise ValueError(f"invalid characters in {name}")
        return value
    return re.sub(r"\$\{([A-Z_]+)\}", replace, path.read_text())


def secret_list(env) -> dict:
    registry = env["IMAGE"].split("/", 1)[0]
    user = env.get("NRP_REGISTRY_USER") or env["IMAGE"].split("/")[1]
    token = env.get("NRP_REGISTRY_TOKEN", "")
    if not token:
        raise ValueError("NRP_REGISTRY_TOKEN is required")
    auth = base64.b64encode(f"{user}:{token}".encode()).decode()
    docker_config = json.dumps({"auths": {registry: {"auth": auth}}})
    return {"apiVersion": "v1", "kind": "List", "items": [
        {"apiVersion": "v1", "kind": "Secret", "type": "Opaque",
         "metadata": {"name": env["ENV_SECRET_NAME"], "namespace": env["K8S_NAMESPACE"]},
         "stringData": {key: env[key] for key in sorted(RUNTIME_KEYS) if key in env}},
        {"apiVersion": "v1", "kind": "Secret", "type": "kubernetes.io/dockerconfigjson",
         "metadata": {"name": env["REGISTRY_SECRET_NAME"], "namespace": env["K8S_NAMESPACE"]},
         "stringData": {".dockerconfigjson": docker_config}},
    ]}


def terminal_state(job: dict) -> str | None:
    for condition in job.get("status", {}).get("conditions", []):
        if condition.get("status") == "True" and condition.get("type") in ("Complete", "Failed"):
            return condition["type"]
    return None


def follow_job(kubectl: list[str], name: str, timeout: float) -> int:
    """Logs exiting 0 does not mean the Job succeeded. Wait for its terminal condition."""
    deadline = time.monotonic() + timeout
    logged = set()
    streams = []
    previous = None
    try:
        while time.monotonic() < deadline:
            job = json.loads(subprocess.check_output(kubectl + ["get", "job", name, "-o", "json"], text=True))
            uid = job["metadata"]["uid"]
            if previous is not None and uid != previous:
                raise ValueError("Job was replaced while watching; refusing to report a different run")
            previous = uid
            pods = json.loads(subprocess.check_output(
                kubectl + ["get", "pods", "-l", f"job-name={name}", "-o", "json"], text=True))
            for pod in sorted(pods["items"], key=lambda p: p["metadata"].get("creationTimestamp", "")):
                if not any(o.get("uid") == uid for o in pod["metadata"].get("ownerReferences", [])):
                    continue
                pod_name = pod["metadata"]["name"]
                if pod_name not in logged and pod.get("status", {}).get("phase") in ("Running", "Succeeded", "Failed"):
                    print(f"[logs] {pod_name}", flush=True)
                    streams.append(subprocess.Popen(kubectl + ["logs", "-f", pod_name, "--timestamps"]))
                    logged.add(pod_name)
            state = terminal_state(job)
            if state is not None:
                for stream in streams:
                    try:
                        stream.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                print(f"[job] {name}: {state}", flush=True)
                return 0 if state == "Complete" else 1
            time.sleep(5)
        print(f"[job] timeout waiting for {name}; job is still running or pending", file=sys.stderr)
        subprocess.run(kubectl + ["describe", "job", name], check=False)
        return 2
    finally:
        for stream in streams:
            if stream.poll() is None:
                stream.terminate()
                try:
                    stream.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stream.kill()
                    stream.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "secrets", "logs"))
    parser.add_argument("target", nargs="?")
    args = parser.parse_args()
    if args.action == "render":
        print(render(Path(args.target), os.environ), end="")
        return
    kubectl = ["kubectl", "--namespace", os.environ["K8S_NAMESPACE"]]
    if args.action == "secrets":
        # Keep secret payloads off argv and stdout. Server-side apply avoids a second
        # plaintext copy in the kubectl last-applied annotation.
        payload = json.dumps(secret_list(os.environ))
        subprocess.run(kubectl + ["apply", "--server-side", "--field-manager=labelcot-grpo", "-f", "-"],
                       input=payload, text=True, check=True)
    else:
        raise SystemExit(follow_job(kubectl, args.target, float(os.environ.get("JOB_TIMEOUT_SECONDS", "172800"))))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cluster command failed: {exc}") from exc
