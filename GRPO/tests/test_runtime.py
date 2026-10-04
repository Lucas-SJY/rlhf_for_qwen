"""Failure-oriented tests for data, preflight and reproducible resume."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labelcot.config import settings_from_env
from labelcot.data import dataset_summary, load_tasks, validate_splits, validate_tasks
from labelcot.runtime import prepare_run, run_lock, validate_settings
from labelcot.reward import RewardWeights
from preflight import check_prompts
from prepare_data import build_task


def task(i):
    return {"id": f"q{i}", "question": f"What is {i} + 1?", "answer": str(i + 1),
            "data_source": "test", "ref_labels": ["logical_deduction"]}


class DataTest(unittest.TestCase):
    def test_group_ids_and_leakage(self):
        for tasks in ([task(1), task(1)], [dict(task(1), id="")], [dict(task(1), ref_labels=[])],
                      [dict(task(1), ref_labels=[{}])], [dict(task(1), answer=None)]):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                validate_tasks(tasks, "fixture")
        with self.assertRaisesRegex(ValueError, "id leakage"):
            validate_splits([task(1)], [task(1)])
        with self.assertRaisesRegex(ValueError, "question leakage"):
            validate_splits([task(1)], [dict(task(1), id="other")])

    def test_bad_json_reports_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(json.dumps(task(1)) + "\n{broken\n")
            with self.assertRaisesRegex(ValueError, ":2:"):
                load_tasks(path)

    def test_fingerprint_includes_order_and_content(self):
        original = dataset_summary([task(1), task(2)])["sha256"]
        self.assertNotEqual(original, dataset_summary([task(2), task(1)])["sha256"])
        self.assertNotEqual(original, dataset_summary([task(1), dict(task(2), answer="99")])["sha256"])

    def test_preparation_rejects_missing_id_and_preserves_zero_answer(self):
        sample = {"question": "value?", "answer": "0", "spans": [{"label": "concluding", "text": "zero"}]}
        with self.assertRaises(ValueError):
            build_task(sample, 100)
        self.assertEqual(build_task(dict(sample, id="q1"), 100)["answer"], "0")

    def test_tokenizer_preflight_rejects_long_prompts(self):
        class Tokenizer:
            chat_template = "fixture"
            def apply_chat_template(self, *args, **kwargs):
                return "formatted question"
            def encode(self, *args, **kwargs):
                return list(range(20))
        with self.assertRaisesRegex(ValueError, "prompts exceed"):
            check_prompts([task(1)], Tokenizer(), 10)
        self.assertEqual(check_prompts([task(1)], Tokenizer(), 20), 20)


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.train, self.val = [task(i) for i in range(8)], [task(10)]
        self.settings = settings_from_env()

    def test_invalid_configuration_fails_before_workers(self):
        cases = {"group_size": 1, "batch_size": 16, "mini_batch_size": 3, "token_budget": 10,
                 "lr": float("nan"), "kl_beta": -1, "save_freq": 0, "keep_checkpoints": 0}
        for key, value in cases.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_settings(dict(self.settings, **{key: value}), self.train, self.val)

    def test_reward_weight_validation(self):
        for align, correct in ((-1, 2), (0, 0), (float("nan"), 0.5), (1, 1)):
            with self.assertRaises(ValueError):
                RewardWeights(align, correct)

    def test_resume_identity_wandb_and_no_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["WANDB_API_KEY"] = "do-not-record-this"
            with run_lock(root):
                first = prepare_run(root, self.settings, self.train, self.val)
                again = prepare_run(root, self.settings, self.train, self.val)
                self.assertEqual(first["wandb_run_id"], again["wandb_run_id"])
                self.assertEqual(os.environ["WANDB_RUN_ID"], first["wandb_run_id"])
                self.assertNotIn("do-not-record-this", (root / "run_manifest.json").read_text())
                with self.assertRaisesRegex(ValueError, "recipe/data/code changed"):
                    prepare_run(root, dict(self.settings, group_size=4), self.train, self.val)
                with self.assertRaisesRegex(ValueError, "recipe/data/code changed"):
                    prepare_run(root, self.settings, list(reversed(self.train)), self.val)

    def test_concurrent_driver_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, run_lock(Path(tmp)):
            with self.assertRaisesRegex(ValueError, "another training driver"):
                with run_lock(Path(tmp)):
                    self.fail("lock should have failed")

    def test_untracked_checkpoints_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoints/global_step_2").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "no run manifest"):
                prepare_run(root, self.settings, self.train, self.val)

    def test_smoke_uses_separate_settings(self):
        with patch.dict(os.environ, {"SMOKE_TEST": "true"}):
            settings = settings_from_env()
        self.assertEqual((settings["batch_size"], settings["group_size"], settings["total_steps"]), (2, 4, 3))
        self.assertEqual(settings["max_zero_signal_steps"], 0)


if __name__ == "__main__":
    unittest.main()
