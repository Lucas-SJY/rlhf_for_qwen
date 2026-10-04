"""Unit tests for the label-following reward:

    python3 -m unittest discover -s tests -v

MathVerifyTest needs math-verify (Python >= 3.10) and is skipped without it; the image
has it, so the full suite runs there.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labelcot.reward import (  # noqa: E402
    MATH_VERIFY_AVAILABLE,
    RewardWeights,
    answer_is_checkable,
    answer_is_correct,
    invalid_labels,
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
        wrong = completion(PERFECT_STEPS, answer="wrong")
        b = score_completion(wrong, TASK, truncated=False, weights=RewardWeights(align=1.0, correct=0.0))
        self.assertAlmostEqual(b.reward, 1.0)
        b = score_completion(wrong, TASK, truncated=False, weights=RewardWeights(align=0.0, correct=1.0))
        self.assertAlmostEqual(b.reward, 0.0)

    def test_reward_is_half_alignment_half_correctness(self):
        drifted = [s.replace("[logical_deduction]", "[reflecting]") for s in PERFECT_STEPS]
        b = score_completion(completion(drifted), TASK, truncated=False)
        self.assertAlmostEqual(b.reward, 0.5 * b.label_alignment + 0.5)

    def test_tag_rate_is_logged_but_not_rewarded(self):
        # Untagged paragraphs between tagged ones lower tag_rate but leave the reward alone.
        steps = PERFECT_STEPS[:2] + ["Some untagged thinking."] + PERFECT_STEPS[2:]
        b = score_completion(completion(steps), TASK, truncated=False)
        self.assertAlmostEqual(b.tag_rate, 5 / 6)
        self.assertAlmostEqual(b.reward, 1.0)
        self.assertIn("tag_rate", b.metrics())


class InvalidLabelTest(unittest.TestCase):
    def test_tags_outside_the_eight_labels_are_found(self):
        for bad in ("[thinking]", "[Planning_Next_Step]", "[planning next step]", "[self_check]", "[double-check]"):
            with self.subTest(tag=bad):
                text = completion(PERFECT_STEPS[:2] + [f"{bad} Some reasoning."] + PERFECT_STEPS[2:])
                self.assertEqual(invalid_labels(text), [bad[1:-1]])

    def test_valid_tags_untagged_paragraphs_and_math_are_allowed(self):
        steps = PERFECT_STEPS + ["No tag here.", "[1, 2] is the interval.", "[x for x in xs] is a list.",
                                 "[Step 1] Expand the product.", "[ab] is a segment."]
        self.assertEqual(invalid_labels(completion(steps)), [])

    def test_only_the_thought_is_checked(self):
        text = completion(PERFECT_STEPS, answer="[thinking] The answer is $\\boxed{9}$.")
        self.assertEqual(invalid_labels(text), [])

    def test_an_invalid_label_scores_zero(self):
        text = completion(PERFECT_STEPS[:-1] + ["[summary] There are 9 divisors."])
        b = score_completion(text, TASK, truncated=False)
        self.assertTrue(b.answer_correct)
        self.assertEqual(b.reward, 0.0)
        self.assertEqual(b.metrics()["invalid_label"], 1.0)


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


@unittest.skipUnless(MATH_VERIFY_AVAILABLE, "needs math-verify (Python >= 3.10)")
class MathVerifyTest(unittest.TestCase):
    def test_equivalent_forms_match(self):
        cases = [
            ("0.5", r"The answer is $\boxed{\frac{1}{2}}$."),
            (r"\frac{1}{2}", r"$\boxed{0.5}$"),
            (r"\dfrac{1}{13}", r"So $\boxed{1/13}$"),
            ("3840", r"Thus the product is $\boxed{3,840}$."),
            ("135", r"$\boxed{135^\circ}$"),
            ("0.1", r"$\boxed{0.1 \text{ m/s}}$"),
            (r"x^2+2x+1", r"$\boxed{(x+1)^2}$"),
        ]
        for gold, answer in cases:
            with self.subTest(gold=gold, answer=answer):
                self.assertTrue(answer_is_correct(answer, gold))
        self.assertFalse(answer_is_correct(r"$\boxed{\frac{1}{3}}$", "0.5"))

    def test_only_the_final_answer_counts(self):
        answer = r"A first guess was $\boxed{4}$; checking again, the result is $\boxed{5}$."
        self.assertTrue(answer_is_correct(answer, "5"))
        self.assertFalse(answer_is_correct(answer, "4"))
        unboxed = "Along the way 2 + 2 = 4, so the final answer is 7."
        self.assertTrue(answer_is_correct(unboxed, "7"))
        self.assertFalse(answer_is_correct(unboxed, "4"))

    def test_multiple_choice_keeps_the_letter_match(self):
        self.assertTrue(answer_is_correct(r"$\boxed{\textbf{(B) } 12}$", "B"))
        self.assertFalse(answer_is_correct(r"$\boxed{\textbf{(C) } 12}$", "B"))

    def test_reasoning_is_not_judged(self):
        # A wrong boxed value inside the thought does not matter, only the final answer.
        text = completion(["[logical_deduction] Maybe $\\boxed{1/3}$? No."], answer=r"$\boxed{\frac{1}{2}}$")
        b = score_completion(text, {"answer": "0.5", "ref_labels": ["logical_deduction"]}, truncated=False)
        self.assertTrue(b.answer_correct)


class SimilarityTest(unittest.TestCase):
    def test_sequence_similarity_ignores_run_lengths(self):
        a = ["reflecting", "reflecting", "verifying"]
        b = ["reflecting", "verifying", "verifying", "verifying"]
        self.assertAlmostEqual(sequence_similarity(a, b), 1.0)


if __name__ == "__main__":
    unittest.main()
