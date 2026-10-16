"""Unit tests for the scoring and summary parts of answer_tasks.py (no vLLM). Stdlib only:

    python3 -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from answer_tasks import default_models, parse_model, question_summary, score_sample, summary_markdown  # noqa: E402

TASK = {"id": "sample_000001", "question": "What is 4 + 5?", "answer": "9"}
RIGHT = "<think>\n[logical_deduction] 4 + 5 = 9.\n\n[concluding] So 9.\n</think>\n\nThe answer is $\\boxed{9}$."
WRONG_TAG = "<think>\n[thinking] 4 + 5 = 8.\n</think>\n\n$\\boxed{8}$"


class ScoreSampleTest(unittest.TestCase):
    def test_a_right_answer_with_valid_labels(self):
        s = score_sample(RIGHT, "stop", 40, TASK)
        self.assertTrue(s["answer_correct"] and s["label_valid"] and s["closed_think"])
        self.assertEqual((s["reward"], s["invalid_tags"], s["truncated"]), (1.0, [], False))
        self.assertEqual(s["final_answer"], "The answer is $\\boxed{9}$.")
        self.assertEqual(s["label_counts"], {"logical_deduction": 1, "concluding": 1})

    def test_an_invented_tag_and_a_wrong_answer(self):
        s = score_sample(WRONG_TAG, "stop", 20, TASK)
        self.assertFalse(s["answer_correct"] or s["label_valid"])
        self.assertEqual((s["reward"], s["invalid_tags"]), (0.0, ["thinking"]))

    def test_a_cut_off_answer_has_no_reward(self):
        s = score_sample("<think>\n[logical_deduction] 4 + 5 is", "length", 8192, TASK)
        self.assertTrue(s["truncated"])
        self.assertIsNone(s["reward"])


class SummaryTest(unittest.TestCase):
    def record(self, label, texts):
        return {"id": TASK["id"], "reference_answer": "9", "model": label,
                "samples": [score_sample(t, r, n, TASK) for t, r, n in texts]}

    def test_counts_and_table(self):
        grpo = question_summary(self.record("grpo", [(RIGHT, "stop", 40), (RIGHT, "stop", 60)]))
        sft = question_summary(self.record("sft", [(RIGHT, "stop", 40), ("<think>\n...", "length", 8192)]))
        self.assertEqual((grpo["correct"], grpo["truncated"], grpo["label_valid"], grpo["mean_response_tokens"]),
                         (2, 0, 2, 50))
        self.assertEqual((sft["correct"], sft["truncated"]), (1, 1))
        table = summary_markdown({"grpo": [grpo], "sft": [sft]})
        self.assertIn("| sample_000001 | `9` | 2/2 / 0 / 2 / 50 | 1/2 / 1 / 1 / 4116 |", table)
        self.assertIn("| **total** | | **2/2** / 0 / 2 | **1/2** / 1 / 1 |", table)

    def test_answer_models_picks_the_default_models(self):
        import os
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            env = {"POLICY_MODEL_PATH": tmp, "POLICY_MODEL_REVISION": ""}
            with mock.patch.dict(os.environ, env):
                self.assertEqual(default_models(Path(tmp) / "run", ["sft"]), [("sft", tmp)])
                with self.assertRaises(SystemExit):  # no checkpoint, so no export to answer with
                    default_models(Path(tmp) / "run", ["grpo"])
                with self.assertRaises(SystemExit):
                    default_models(Path(tmp) / "run", ["base"])

    def test_model_spec(self):
        self.assertEqual(parse_model("grpo=/grpo/runs/x/export/global_step_200"),
                         ("grpo", "/grpo/runs/x/export/global_step_200"))
        with self.assertRaises(SystemExit):
            parse_model("/no/label")


if __name__ == "__main__":
    unittest.main()
