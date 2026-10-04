"""Capture the actual launcher arguments without importing the GPU runtime."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from labelcot.config import settings_from_env

APP_ROOT = Path(__file__).resolve().parents[1]


def capture_entrypoint(root, smoke=False, extra=()):
    executable = root / "python3"
    executable.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                          "print('CAPTURE=' + json.dumps({'args': sys.argv[1:], "
                          "'zero_limit': os.environ.get('MAX_ZERO_SIGNAL_STEPS')}))\n")
    executable.chmod(0o755)
    env = {"PATH": f"{root}:{os.environ['PATH']}", "OUTPUT_ROOT": str(root / "runs"),
           "RUN_NAME": "fixture", "ENV_FILE": str(root / "absent"), "REPORT_TO": "",
           "SMOKE_TEST": "true" if smoke else "false"}
    result = subprocess.run(["bash", str(APP_ROOT / "src/entrypoint.sh"), *extra],
                            env=env, capture_output=True, text=True, check=True)
    return json.loads(next(line.removeprefix("CAPTURE=") for line in result.stdout.splitlines()
                           if line.startswith("CAPTURE=")))


class EntrypointTest(unittest.TestCase):
    def test_smoke_defaults_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            smoke = capture_entrypoint(Path(tmp), True)
            normal = capture_entrypoint(Path(tmp), False, ["rllm.rollout.n=2"])
        self.assertIn("rllm.rollout.n=4", smoke["args"])
        self.assertIn("rllm.trainer.total_batches=3", smoke["args"])
        self.assertEqual(smoke["zero_limit"], "0")
        self.assertTrue(any("fixture-smoke/checkpoints" in arg for arg in smoke["args"]))
        self.assertIn("rllm.rollout.n=8", normal["args"])
        self.assertEqual(normal["args"][-1], "rllm.rollout.n=2")
        self.assertIn("trainer.max_actor_ckpt_to_keep=2", normal["args"])
        self.assertIn("rllm.workflow.raise_on_error=true", normal["args"])


if __name__ == "__main__":
    unittest.main()
