"""Unit tests for the label-following reward. Stdlib only:

    python3 -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labelcot.reward import (  # noqa: E402
    RewardWeights,
    answer_is_checkable,
    is_correct,
    score_completion,
    sequence_similarity,
)

REF = ["restating_problem", "planning_next_step", "logical_deduction", "logical_deduction", "concluding"]
TASK = {"answer": "9", "ref_labels": REF}


def completion(steps, answer="The answer is $\\boxed{9}$.", close=True):
    thought = "\n\n".join(steps)
    text = f"<think>\n{thought}\n"
    if close:
        text += f"</think>\n\n{answer}"
    return text


PERFECT_STEPS = [
    "[restating_problem] We need the number of divisors of 196.",
    "[planning_next_step] Factor it first.",
    "[logical_deduction] 196 = 2^2 * 7^2.",
    "[logical_deduction] So the count is (2+1)(2+1) = 9.",
    "[concluding] There are 9 divisors.",
]


class RewardTest(unittest.TestCase):
    def test_perfect_completion_scores_one(self):
        b = score_completion(completion(PERFECT_STEPS), TASK, truncated=False)
        self.assertTrue(b.closed and b.answer_correct)
        self.assertAlmostEqual(b.tag_rate, 1.0)
        self.assertAlmostEqual(b.label_alignment, 1.0)
        self.assertAlmostEqual(b.reward, 1.0)

    def test_truncated_or_unclosed_scores_zero(self):
        self.assertEqual(score_completion(completion(PERFECT_STEPS), TASK, truncated=True).reward, 0.0)
        self.assertEqual(score_completion(completion(PERFECT_STEPS, close=False), TASK, truncated=False).reward, 0.0)

    def test_untagged_and_invented_tags_lower_tag_rate(self):
        steps = PERFECT_STEPS[:3] + ["No tag on this step.", "[made_up_label] Invented tag."]
        b = score_completion(completion(steps), TASK, truncated=False)
        self.assertAlmostEqual(b.tag_rate, 3 / 5)
        self.assertAlmostEqual(b.invented_tag_rate, 1 / 5)
        self.assertLess(b.reward, 1.0)

    def test_label_drift_lowers_alignment(self):
        drifted = [s.replace("[logical_deduction]", "[reflecting]") for s in PERFECT_STEPS]
        b = score_completion(completion(drifted), TASK, truncated=False)
        self.assertAlmostEqual(b.tag_rate, 1.0)
        self.assertLess(b.label_alignment, 0.9)

    def test_boxed_inside_thought_does_not_count(self):
        steps = PERFECT_STEPS[:-1] + ["[concluding] So \\boxed{9}."]
        b = score_completion(completion(steps, answer="I am not sure."), TASK, truncated=False)
        self.assertFalse(b.answer_correct)

    def test_prose_answer_drops_correctness_term(self):
        task = {"answer": "\\text{P and Q cannot both be true.}", "ref_labels": REF}
        b = score_completion(completion(PERFECT_STEPS, answer="Proof done."), task, truncated=False)
        self.assertFalse(b.answer_checkable)
        self.assertAlmostEqual(b.reward, 1.0)
        self.assertNotIn("answer_correct", b.metrics())

    def test_weights_are_respected(self):
        w = RewardWeights(tag=1.0, align=0.0, correct=0.0)
        b = score_completion(completion(PERFECT_STEPS, answer="wrong"), TASK, truncated=False, weights=w)
        self.assertAlmostEqual(b.reward, 1.0)


class AnswerTest(unittest.TestCase):
    def test_multiple_choice(self):
        self.assertTrue(is_correct("\\textbf{(A)}", "A"))
        self.assertTrue(is_correct("(C) 15", "C"))
        self.assertFalse(is_correct("B", "A"))

    def test_numeric_and_latex(self):
        self.assertTrue(is_correct("\\frac{1}{2}", "0.5"))
        self.assertTrue(is_correct("1,000", "1000"))
        self.assertFalse(is_correct(None, "3"))

    def test_checkable(self):
        self.assertTrue(answer_is_checkable("42"))
        self.assertTrue(answer_is_checkable("Q(n)"))
        self.assertFalse(answer_is_checkable("\\text{For every positive integer } n"))
        self.assertFalse(answer_is_checkable(""))


class SimilarityTest(unittest.TestCase):
    def test_sequence_similarity_ignores_run_lengths(self):
        a = ["reflecting", "reflecting", "verifying"]
        b = ["reflecting", "verifying", "verifying", "verifying"]
        self.assertAlmostEqual(sequence_similarity(a, b), 1.0)


if __name__ == "__main__":
    unittest.main()
