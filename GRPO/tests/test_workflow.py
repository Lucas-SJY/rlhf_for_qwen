"""Unit tests for LabeledCoTWorkflow's regenerate-on-invalid-label loop.

    python3 -m unittest discover -s tests -v

Needs rLLM (the image has it) and is skipped without it. The rollout engine is a fake
that replays scripted completions, so no GPU or vLLM is involved.
"""

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if importlib.util.find_spec("rllm") is not None:
    from rllm.engine.rollout.rollout_engine import ModelOutput

    from labelcot.workflow import LabeledCoTWorkflow
else:  # missing dependency only; incompatible installed rLLM must fail the tests
    LabeledCoTWorkflow = None

TASK = {"id": "q1", "question": "What is 4 + 5?", "answer": "9", "ref_labels": ["logical_deduction", "concluding"]}
VALID = "<think>\n[logical_deduction] 4 + 5 = 9.\n\n[concluding] So it is 9.\n</think>\n\n$\\boxed{9}$"
INVALID = "<think>\n[thinking] 4 + 5 = 9.\n\n[concluding] So it is 9.\n</think>\n\n$\\boxed{9}$"


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


class FakeEngine:
    """Replays scripted completions in order (the last one repeats) and counts the calls."""

    def __init__(self, completions):
        self.completions = list(completions)
        self.calls = 0
        self.samplings = []
        self.tokenizer = FakeTokenizer()
        self.train_sampling_params = {"temperature": 1.0}
        self.val_sampling_params = {"temperature": 0.6}
        self.is_validation = False
        self.weight_version = 7

    async def get_token_output_from_token_input(self, token_input, application_id, **sampling):
        self.samplings.append(sampling)
        text = self.completions[min(self.calls, len(self.completions) - 1)]
        self.calls += 1
        return text

    def assemble_model_output(self, token_input, token_output, prompt_ids):
        thought, _, answer = token_output.partition("</think>")
        return ModelOutput(text=token_output, content=answer.strip(), reasoning=thought, prompt_ids=prompt_ids,
                           completion_ids=[4, 5, 6], logprobs=[-0.1, -0.2, -0.3], finish_reason="stop")


def run_workflow(completions, max_label_retries=3):
    engine = FakeEngine(completions)
    workflow = LabeledCoTWorkflow(engine, executor=None, max_label_retries=max_label_retries)
    episode = asyncio.run(workflow.run_with_termination_handling(TASK, "q1:0"))
    step = episode.trajectories[0].steps[0]
    return engine, workflow, step


@unittest.skipUnless(LabeledCoTWorkflow is not None, "needs rLLM")
class LabelRetryTest(unittest.TestCase):
    def test_validation_sampling_and_exact_tokens_survive(self):
        engine = FakeEngine([VALID])
        engine.is_validation = True
        workflow = LabeledCoTWorkflow(engine, executor=None)
        episode = asyncio.run(workflow.run_with_termination_handling(TASK, "val:0"))
        output = episode.trajectories[0].steps[0].model_output
        self.assertEqual(engine.samplings, [{"temperature": 0.6}])
        self.assertEqual(output.completion_ids, [4, 5, 6])
        self.assertEqual(output.logprobs, [-0.1, -0.2, -0.3])
        self.assertEqual(output.weight_version, 7)
        self.assertTrue(episode.is_correct)

    def test_valid_completion_is_kept_without_retry(self):
        engine, workflow, step = run_workflow([VALID])
        self.assertEqual(engine.calls, 1)
        self.assertEqual(step.model_response, VALID)
        self.assertAlmostEqual(step.reward, 1.0)
        self.assertEqual(workflow._breakdown.label_retries, 0)

    def test_invalid_label_is_regenerated_and_only_the_kept_sample_is_trained(self):
        engine, workflow, step = run_workflow([INVALID, INVALID, VALID])
        self.assertEqual(engine.calls, 3)
        self.assertEqual(step.model_response, VALID)
        self.assertAlmostEqual(step.reward, 1.0)
        metrics = workflow._breakdown.metrics()
        self.assertEqual((metrics["label_retries"], metrics["invalid_label"]), (2.0, 0.0))

    def test_still_invalid_after_the_last_retry_scores_zero(self):
        engine, workflow, step = run_workflow([INVALID], max_label_retries=2)
        self.assertEqual(engine.calls, 3)
        self.assertEqual(step.reward, 0.0)
        self.assertEqual(workflow._breakdown.metrics()["invalid_label"], 1.0)


if __name__ == "__main__":
    unittest.main()
