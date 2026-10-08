"""Unit tests for LabeledCoTWorkflow's regenerate-on-invalid-label loop.

    python3 -m unittest discover -s tests -v

Needs rLLM (the image has it) and is skipped without it. The rollout engine is a fake
that replays scripted completions, so no GPU or vLLM is involved.
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from rllm.engine.rollout.rollout_engine import ModelOutput

    from rllm.workflows.workflow import TerminationReason

    from labelcot.workflow import LabeledCoTWorkflow
except ImportError:  # no rLLM on this interpreter
    LabeledCoTWorkflow = None

TASK = {"id": "q1", "question": "What is 4 + 5?", "answer": "9",
        "prompt": [{"role": "user", "content": "What is 4 + 5?"}]}
VALID = "<think>\n[logical_deduction] 4 + 5 = 9.\n\n[concluding] So it is 9.\n</think>\n\n$\\boxed{9}$"
INVALID = "<think>\n[thinking] 4 + 5 = 9.\n\n[concluding] So it is 9.\n</think>\n\n$\\boxed{9}$"


class FakeTokenizer:
    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.seen.append(messages)
        return f"<|im_start|>user\n{messages[-1]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


class FakeEngine:
    """Replays scripted completions in order (the last one repeats) and counts the calls."""

    def __init__(self, completions, finish_reason="stop"):
        self.completions = list(completions)
        self.finish_reason = finish_reason
        self.calls = 0
        self.tokenizer = FakeTokenizer()
        self.train_sampling_params = {"temperature": 1.0}
        self.val_sampling_params = {"temperature": 0.6}
        self.is_validation = False
        self.weight_version = 7

    async def get_token_output_from_token_input(self, token_input, application_id, **sampling):
        text = self.completions[min(self.calls, len(self.completions) - 1)]
        self.calls += 1
        return text

    def assemble_model_output(self, token_input, token_output, prompt_ids):
        thought, _, answer = token_output.partition("</think>")
        return ModelOutput(text=token_output, content=answer.strip(), reasoning=thought, prompt_ids=prompt_ids,
                           completion_ids=[4, 5, 6], logprobs=[-0.1, -0.2, -0.3], finish_reason=self.finish_reason)


def run_episode(completions, max_label_retries=3, task=TASK, finish_reason="stop"):
    engine = FakeEngine(completions, finish_reason)
    workflow = LabeledCoTWorkflow(engine, executor=None, max_label_retries=max_label_retries)
    episode = asyncio.run(workflow.run_with_termination_handling(task, "q1:0"))
    return engine, workflow, episode


def run_workflow(completions, max_label_retries=3, task=TASK):
    engine, workflow, episode = run_episode(completions, max_label_retries, task)
    return engine, workflow, episode.trajectories[0].steps[0]


@unittest.skipUnless(LabeledCoTWorkflow is not None, "needs rLLM")
class LabelRetryTest(unittest.TestCase):
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

    def test_still_invalid_after_the_last_retry_loses_the_label_weight(self):
        engine, workflow, step = run_workflow([INVALID], max_label_retries=2)
        self.assertEqual(engine.calls, 3)
        # The answer is right, the labels are not: only the correctness weight.
        self.assertAlmostEqual(step.reward, 0.8)
        self.assertEqual(workflow._breakdown.metrics()["invalid_label"], 1.0)

    def test_a_cut_off_answer_ends_the_episode_as_overlong(self):
        # The trainer's compact filtering drops episodes by this termination reason.
        _, _, episode = run_episode([VALID], finish_reason="length")
        self.assertEqual(episode.termination_reason, TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)
        _, _, episode = run_episode([VALID])
        self.assertEqual(episode.termination_reason, TerminationReason.ENV_DONE)

    def test_the_task_prompt_is_what_the_model_sees(self):
        engine, _, _ = run_workflow([VALID])
        self.assertEqual(engine.tokenizer.seen[-1], TASK["prompt"])
        # Tasks without a prompt (bespoke-v2) fall back to the bare question.
        bare = {k: v for k, v in TASK.items() if k != "prompt"}
        engine, _, _ = run_workflow([VALID], task=bare)
        self.assertEqual(engine.tokenizer.seen[-1], [{"role": "user", "content": TASK["question"]}])


if __name__ == "__main__":
    unittest.main()
