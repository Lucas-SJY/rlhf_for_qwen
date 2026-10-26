"""Unit tests for the wandb sections added by labelcot.patches:

    python3 -m unittest discover -s tests -v

Stdlib only: add_metric_sections does not need rLLM.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labelcot.patches import METRIC_SECTIONS, add_metric_sections  # noqa: E402


class MetricSectionsTest(unittest.TestCase):
    def test_copies_the_batch_means_into_their_sections(self):
        metrics = {"batch/reward": 0.62, "batch/reward_label": 0.2, "batch/reward_correct": 0.42,
                   "batch/answer_correct": 0.35, "batch/answer_correct_finished": 0.53, "batch/truncated": 0.34}
        add_metric_sections(metrics)
        self.assertEqual(metrics["reward_breakdown/total"], 0.62)
        self.assertEqual(metrics["reward_breakdown/format"], 0.2)
        self.assertEqual(metrics["reward_breakdown/correctness"], 0.42)
        self.assertEqual(metrics["rollout_accuracy/all"], 0.35)
        self.assertEqual(metrics["rollout_accuracy/finished"], 0.53)
        # The batch/ keys stay.
        self.assertEqual(metrics["batch/answer_correct"], 0.35)

    def test_skips_what_the_step_did_not_log(self):
        # A step whose answers were all cut off has no reward and no finished accuracy.
        metrics = {"batch/answer_correct": 0.0}
        add_metric_sections(metrics)
        self.assertEqual(metrics, {"batch/answer_correct": 0.0, "rollout_accuracy/all": 0.0})

    def test_counts_the_cut_off_rollouts_and_copies_lengths_and_loops(self):
        metrics = {"batch/truncated": 0.6953125, "groups/num_trajs_before_filter": 128, "response_length/mean": 11489.9,
                   "response_length/max": 12288, "batch/loop": 0.0234, "batch/loop_among_cut": 0.034}
        add_metric_sections(metrics)
        self.assertEqual(metrics["rollout_length/cut_off_count"], 89)
        self.assertEqual((metrics["rollout_length/rollouts"], metrics["rollout_length/cut_off_share"]), (128, 0.6953125))
        self.assertEqual((metrics["rollout_length/mean_tokens"], metrics["rollout_length/max_tokens"]), (11489.9, 12288))
        self.assertEqual((metrics["loop/loop"], metrics["loop/among_cut_off"]), (0.0234, 0.034))

    def test_the_sections(self):
        self.assertEqual({name.split("/")[0] for name in METRIC_SECTIONS},
                         {"reward_breakdown", "rollout_accuracy", "rollout_length", "loop", "retry"})


if __name__ == "__main__":
    unittest.main()
