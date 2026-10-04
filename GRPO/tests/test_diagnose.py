import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnose import diagnostic, grouped_advantages


class DiagnosticTest(unittest.TestCase):
    def test_actual_reward_variation(self):
        report = diagnostic()
        self.assertEqual(report["rewards"], [1, 0.5, 0, 1])
        self.assertAlmostEqual(sum(report["advantages"]), 0)
        self.assertEqual(grouped_advantages([0, 0, 0]), [0, 0, 0])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch for CPU policy-gradient test")
    def test_optimizer_changes_probabilities_in_correct_direction(self):
        report = diagnostic(require_torch=True)
        self.assertEqual(report["policy_update"], "passed")
        self.assertGreater(report["gradient_norm"], 0)


if __name__ == "__main__":
    unittest.main()
