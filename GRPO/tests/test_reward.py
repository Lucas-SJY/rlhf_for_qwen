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
    LABEL_PRIOR,
    MATH_VERIFY_AVAILABLE,
    RewardWeights,
    answer_is_checkable,
    answer_is_correct,
    invalid_labels,
    is_correct,
    label_mix_similarity,
    score_completion,
)

TASK = {"answer": "9"}


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
    def test_valid_labels_and_right_answer_score_one(self):
        b = score_completion(completion(PERFECT_STEPS), TASK, truncated=False)
        self.assertTrue(b.closed and b.answer_correct and b.label_valid)
        self.assertAlmostEqual(b.reward, 1.0)
        # The label mix is only logged.
        self.assertTrue(0.0 < b.label_mix < 1.0)

    def test_valid_labels_and_wrong_answer_score_the_label_weight(self):
        b = score_completion(completion(PERFECT_STEPS, answer="$\\boxed{8}$"), TASK, truncated=False)
        self.assertTrue(b.label_valid and not b.answer_correct)
        self.assertAlmostEqual(b.reward, 0.2)

    def test_no_reference_trace_is_used(self):
        # The reward reads only the answer; reference labels, if a task has them, change nothing.
        with_ref = dict(TASK, ref_labels=["verifying"] * 5)
        a = score_completion(completion(PERFECT_STEPS), TASK, truncated=False)
        b = score_completion(completion(PERFECT_STEPS), with_ref, truncated=False)
        self.assertEqual(a.reward, b.reward)

    def test_unclosed_scores_zero(self):
        self.assertEqual(score_completion(completion(PERFECT_STEPS, close=False), TASK, truncated=False).reward, 0.0)

    def test_truncated_is_left_out_of_the_logged_reward(self):
        # The trainer drops a cut-off completion from the batch; its reward is a placeholder.
        b = score_completion(completion(PERFECT_STEPS), TASK, truncated=True)
        self.assertNotIn("reward", b.metrics())
        self.assertEqual(b.metrics()["truncated"], 1.0)
        self.assertIn("reward", score_completion(completion(PERFECT_STEPS), TASK, truncated=False).metrics())

    def test_untagged_paragraphs_are_allowed(self):
        # tag_rate is logged; untagged paragraphs between tagged ones keep the label term.
        steps = PERFECT_STEPS[:2] + ["Some untagged thinking."] + PERFECT_STEPS[2:]
        b = score_completion(completion(steps), TASK, truncated=False)
        ref = score_completion(completion(PERFECT_STEPS), TASK, truncated=False)
        self.assertAlmostEqual(b.tag_rate, 5 / 6)
        self.assertAlmostEqual(b.label_mix, ref.label_mix)
        self.assertAlmostEqual(b.reward, ref.reward)
        self.assertIn("tag_rate", b.metrics())

    def test_no_labels_scores_only_the_correctness_weight(self):
        b = score_completion(completion(["No tags anywhere.", "Still none."]), TASK, truncated=False)
        self.assertFalse(b.label_valid)
        self.assertAlmostEqual(b.reward, 0.8)

    def test_boxed_inside_thought_does_not_count(self):
        steps = PERFECT_STEPS[:-1] + ["[concluding] So \\boxed{9}."]
        b = score_completion(completion(steps, answer="I am not sure."), TASK, truncated=False)
        self.assertFalse(b.answer_correct)

    def test_uncheckable_answer_scores_the_label_term_alone(self):
        task = {"answer": "\\text{P and Q cannot both be true.}"}
        b = score_completion(completion(PERFECT_STEPS, answer="Proof done."), task, truncated=False)
        self.assertFalse(b.answer_checkable)
        self.assertAlmostEqual(b.reward, 1.0)
        b = score_completion(completion(["No tags."], answer="Proof done."), task, truncated=False)
        self.assertAlmostEqual(b.reward, 0.0)
        self.assertNotIn("answer_correct", b.metrics())

    def test_weights_are_respected(self):
        wrong = completion(PERFECT_STEPS, answer="wrong")
        b = score_completion(wrong, TASK, truncated=False, weights=RewardWeights(label=1.0, correct=0.0))
        self.assertAlmostEqual(b.reward, 1.0)
        b = score_completion(completion(PERFECT_STEPS), TASK, truncated=False, weights=RewardWeights(label=0.0, correct=1.0))
        self.assertAlmostEqual(b.reward, 1.0)


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

    def test_an_invalid_label_loses_the_label_weight(self):
        text = completion(PERFECT_STEPS[:-1] + ["[summary] There are 9 divisors."])
        b = score_completion(text, TASK, truncated=False)
        self.assertTrue(b.answer_correct)
        self.assertFalse(b.label_valid)
        self.assertAlmostEqual(b.reward, 0.8)
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


class LabelMixTest(unittest.TestCase):
    def test_prior_is_a_distribution_over_the_eight_labels(self):
        self.assertEqual(len(LABEL_PRIOR), 8)
        self.assertAlmostEqual(sum(LABEL_PRIOR.values()), 1.0)

    def test_matching_the_target_scores_one(self):
        labels = ["reflecting", "verifying", "verifying", "concluding"]
        target = {"reflecting": 0.25, "verifying": 0.5, "concluding": 0.25}
        self.assertAlmostEqual(label_mix_similarity(labels, target), 1.0)

    def test_a_natural_mix_beats_a_single_label(self):
        natural = [label for label, share in LABEL_PRIOR.items() for _ in range(round(share * 100))]
        one_label = ["logical_deduction"] * len(natural)
        self.assertGreater(label_mix_similarity(natural), 0.95)
        # All one label: the similarity is just that label's target share.
        self.assertAlmostEqual(label_mix_similarity(one_label), LABEL_PRIOR["logical_deduction"])
        self.assertEqual(label_mix_similarity([]), 0.0)


if __name__ == "__main__":
    unittest.main()
