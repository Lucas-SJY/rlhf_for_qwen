"""Exercise real preparation/preflight CLI and payload validation without a GPU."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from labelcot.rollout import validate_model_output
from test_runtime import task

APP_ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def test_preparation_to_preflight_preserves_heldout_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples, split, out = root / "samples", root / "sft", root / "data"
            samples.mkdir()
            split.mkdir()
            for i in range(10):
                sample = dict(task(i), spans=[{"label": "logical_deduction", "text": "reference reasoning"}])
                (samples / f"sample_{i:03}.json").write_text(json.dumps(sample))
            (split / "validation.jsonl").write_text('\n'.join(json.dumps({"id": f"q{i}"}) for i in (8, 9)))
            subprocess.run([sys.executable, str(APP_ROOT / "src/prepare_data.py"), "--input-dir", str(samples),
                            "--split-from", str(split), "--output-dir", str(out)], check=True, capture_output=True)
            env = {"PATH": os.environ["PATH"], "TOTAL_TRAINING_STEPS": "1"}
            result = subprocess.run([sys.executable, str(APP_ROOT / "src/preflight.py"),
                                     "--train-file", str(out / "train.jsonl"), "--val-file", str(out / "validation.jsonl")],
                                    env=env, check=True, text=True, capture_output=True)
            report = json.loads(result.stdout)
            self.assertEqual((report["train"]["count"], report["validation"]["count"]), (8, 2))
            self.assertEqual(report["warnings"], [])
            val = [json.loads(line)["id"] for line in (out / "validation.jsonl").read_text().splitlines()]
            self.assertEqual(val, ["q8", "q9"])

    def test_training_requires_matching_sampled_token_payload(self):
        good = {"prompt_ids": [1, 2], "completion_ids": [3, 4], "logprobs": [-0.1, -0.2]}
        validate_model_output(SimpleNamespace(**good), [1, 2])
        for changed in ({"prompt_ids": [1]}, {"completion_ids": []}, {"logprobs": None},
                        {"logprobs": [-0.1]}, {"logprobs": [float("nan"), -0.2]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_model_output(SimpleNamespace(**dict(good, **changed)), [1, 2])


if __name__ == "__main__":
    unittest.main()
