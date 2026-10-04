import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster import follow_job, render, secret_list, terminal_state


class ClusterTest(unittest.TestCase):
    def test_secret_allowlist_and_isolation(self):
        env = {"IMAGE": "registry.example/team/image:tag", "K8S_NAMESPACE": "ns",
               "ENV_SECRET_NAME": "dayallen-env", "REGISTRY_SECRET_NAME": "dayallen-registry",
               "NRP_REGISTRY_TOKEN": "registry-password", "WANDB_API_KEY": "wandb-password",
               "UNRELATED_SECRET": "not-for-pod", "TRAIN_BATCH_SIZE": "8"}
        payload = secret_list(env)
        runtime, registry = payload["items"]
        self.assertEqual(runtime["stringData"], {"TRAIN_BATCH_SIZE": "8", "WANDB_API_KEY": "wandb-password"})
        self.assertNotIn("registry-password", json.dumps(runtime))
        self.assertNotIn("not-for-pod", json.dumps(payload))
        self.assertEqual(registry["metadata"]["name"], "dayallen-registry")

    def test_render_rejects_missing_and_injected_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.yaml"
            path.write_text("name: ${JOB_NAME}\nimage: ${IMAGE}\n")
            with self.assertRaises(ValueError):
                render(path, {"JOB_NAME": "mine"})
            with self.assertRaises(ValueError):
                render(path, {"JOB_NAME": "bad\ninjection", "IMAGE": "r/i:t"})
            self.assertIn("image: r/i:t", render(path, {"JOB_NAME": "mine", "IMAGE": "r/i:t"}))

    def test_running_pod_is_not_a_successful_job(self):
        self.assertIsNone(terminal_state({"status": {"active": 1}}))
        for state, code in (("Complete", 0), ("Failed", 1)):
            job = {"metadata": {"uid": "job-id"}, "status": {"conditions": [{"type": state, "status": "True"}]}}
            with patch("cluster.subprocess.check_output", side_effect=[json.dumps(job), '{"items": []}']):
                self.assertEqual(follow_job(["kubectl"], "test", 10), code)

    def test_watcher_rejects_replacement_job(self):
        first, second = ({"metadata": {"uid": uid}} for uid in ("old", "new"))
        with patch("cluster.subprocess.check_output", side_effect=[json.dumps(first), '{"items": []}', json.dumps(second)]), \
                patch("cluster.time.sleep"):
            with self.assertRaisesRegex(ValueError, "replaced"):
                follow_job(["kubectl"], "test", 10)


if __name__ == "__main__":
    unittest.main()
