"""Unit tests for LabeledCoTWorkflow's regenerate-on-invalid-label loop.

    python3 -m unittest discover -s tests -v

Needs rLLM (the image has it) and is skipped without it. The rollout engine is a fake
that replays scripted completions, so no GPU or vLLM is involved.
"""

import asyncio
import sys
import unittest
from types import SimpleNamespace
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
        self.sampling_seen = []

    async def get_token_output_from_token_input(self, token_input, application_id, **sampling):
        self.sampling_seen.append(sampling)
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

    def test_extra_sampling_params_reach_the_engine(self):
        extra = {"train": {"presence_penalty": 1.0, "min_p": 0.0}, "val": {"presence_penalty": 0.5}}
        for validation, expected in ((False, {"temperature": 1.0, "presence_penalty": 1.0, "min_p": 0.0}),
                                     (True, {"temperature": 0.6, "presence_penalty": 0.5})):
            engine = FakeEngine([VALID])
            engine.is_validation = validation
            workflow = LabeledCoTWorkflow(engine, executor=None, extra_sampling=extra)
            asyncio.run(workflow.run_with_termination_handling(TASK, "q1:0"))
            self.assertEqual(engine.sampling_seen, [expected])


class CharTokenizer:
    """One token per character; token 0 is the end of the completion."""

    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "P"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i != 0)


class TokenEngine:
    """Token-level engine: each attempt replays the next scripted completion, max_tokens at a
    time from where the token input leaves off, and records every call's input and limit."""

    def __init__(self, completions, max_response_length=10_000):
        self.completions = list(completions)
        self.tokenizer = CharTokenizer()
        self.train_sampling_params = {"temperature": 0.6}
        self.val_sampling_params = {"temperature": 0.6}
        self.is_validation = False
        self.weight_version = 3
        self.max_response_length = max_response_length
        self.calls = []
        self.attempt = -1

    async def get_token_output_from_token_input(self, token_input, application_id, **sampling):
        prompt_len = 1
        done = len(token_input) - prompt_len
        if done == 0:
            self.attempt += 1
        ids = [ord(c) for c in self.completions[min(self.attempt, len(self.completions) - 1)]] + [0]
        limit = sampling.get("max_tokens", self.max_response_length)
        self.calls.append({"input_len": len(token_input), "max_tokens": limit, "attempt": self.attempt,
                           "enforce": sampling.get("enforce_max_prompt_length", True)})
        new = ids[done:done + limit]
        return SimpleNamespace(token_ids=new, log_probs=[-0.5] * len(new),
                               stop_reason="length" if len(new) >= limit else "stop")

    def assemble_model_output(self, token_input, token_output, prompt_ids):
        text = self.tokenizer.decode(token_output.token_ids)
        thought, _, answer = text.partition("</think>")
        return ModelOutput(text=text, content=answer.strip(), reasoning=thought, prompt_ids=prompt_ids,
                           completion_ids=list(token_output.token_ids), logprobs=token_output.log_probs,
                           finish_reason=token_output.stop_reason)


def stuck(label="recalling_knowledge", paragraphs=6):
    """A thought that repeats one label, each paragraph 100 characters (= tokens) long."""
    body = "\n\n".join([f"[{label}] " + "x" * (99 - len(label) - 3)] * paragraphs)
    return f"<think>\n[restating_problem] We need 4 + 5.\n\n{body}\n</think>\n\n$\\boxed{{9}}$"


def run_chunked(completions, retries=2, chunk=64, validation=False, mask_final=False, episode_too=False):
    engine = TokenEngine(completions)
    engine.is_validation = validation
    workflow = LabeledCoTWorkflow(engine, executor=None, repeat_label_retries=retries, repeat_min_paragraphs=3,
                                  repeat_min_tokens=200, gen_chunk_tokens=chunk, repeat_mask_final=mask_final)
    episode = asyncio.run(workflow.run_with_termination_handling(TASK, "q1:0"))
    if episode_too:
        return engine, workflow, episode
    return engine, workflow, episode.trajectories[0].steps[0]


@unittest.skipUnless(LabeledCoTWorkflow is not None, "needs rLLM")
class RepeatedLabelRetryTest(unittest.TestCase):
    def test_a_clean_completion_is_reassembled_from_its_chunks(self):
        engine, workflow, step = run_chunked([VALID])
        self.assertEqual(step.model_response, VALID)
        self.assertEqual(len(step.model_output.completion_ids), len(VALID) + 1)
        self.assertEqual(len(step.model_output.logprobs), len(VALID) + 1)
        # Each chunk continues from the prompt plus the tokens so far; only the first is the prompt.
        self.assertEqual([c["input_len"] for c in engine.calls], [1 + 64 * i for i in range(len(engine.calls))])
        self.assertEqual([c["enforce"] for c in engine.calls], [True] + [False] * (len(engine.calls) - 1))
        self.assertAlmostEqual(step.reward, 1.0)
        self.assertEqual(workflow._breakdown.metrics()["repeat_retries"], 0.0)

    def test_a_repeated_label_stops_the_completion_and_it_is_sampled_again(self):
        engine, workflow, step = run_chunked([stuck(), VALID])
        self.assertEqual(step.model_response, VALID)
        first = [c for c in engine.calls if c["attempt"] == 0]
        # Stopped soon after the third repeated paragraph, well before its 600-token end.
        self.assertLess(64 * len(first), len(stuck()))
        m = workflow._breakdown.metrics()
        self.assertEqual((m["repeat_retries"], m["repeat_in_final"]), (1.0, 0.0))
        self.assertEqual(m["repeat_discarded_tokens"], 64 * len(first))

    def test_the_last_attempt_runs_to_the_end_and_is_kept(self):
        engine, workflow, step = run_chunked([stuck()], retries=2)
        self.assertEqual(step.model_response, stuck())
        # Two stopped attempts, then one call for the whole third one.
        self.assertEqual(sorted({c["attempt"] for c in engine.calls}), [0, 1, 2])
        self.assertEqual([c for c in engine.calls if c["attempt"] == 2][0]["max_tokens"], engine.max_response_length)
        m = workflow._breakdown.metrics()
        self.assertEqual((m["repeat_retries"], m["repeat_in_final"]), (2.0, 1.0))
        self.assertAlmostEqual(step.reward, 1.0)

    def test_a_last_attempt_that_still_repeats_is_dropped_from_training(self):
        # It ends as max_turns_exceeded, which entrypoint.sh has compact filtering mask.
        _, workflow, episode = run_chunked([stuck()], mask_final=True, episode_too=True)
        self.assertEqual(episode.termination_reason, TerminationReason.MAX_TURNS_EXCEEDED)
        self.assertEqual(workflow._breakdown.metrics()["repeat_in_final"], 1.0)
        # A clean retry, the switch off, and validation all end normally.
        for kwargs in ({"completions": [stuck(), VALID], "mask_final": True},
                       {"completions": [stuck()], "mask_final": False},
                       {"completions": [stuck()], "mask_final": True, "validation": True}):
            _, _, episode = run_chunked(episode_too=True, **kwargs)
            self.assertEqual(episode.termination_reason, TerminationReason.ENV_DONE, kwargs)

    def test_validation_is_never_stopped(self):
        engine, workflow, step = run_chunked([stuck(), VALID], validation=True)
        self.assertEqual(step.model_response, stuck())
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(workflow._breakdown.metrics()["repeat_in_final"], 1.0)

    def test_zero_retries_turns_the_check_off(self):
        engine, workflow, step = run_chunked([stuck(), VALID], retries=0)
        self.assertEqual(step.model_response, stuck())
        self.assertEqual(len(engine.calls), 1)

    def test_a_run_formed_in_the_last_chunk_still_counts(self):
        # One chunk holds the whole completion: it finishes and is still caught.
        engine, workflow, step = run_chunked([stuck(), VALID], chunk=5000)
        self.assertEqual(step.model_response, VALID)
        self.assertEqual(workflow._breakdown.metrics()["repeat_retries"], 1.0)

    def test_a_cut_off_completion_stops_at_the_budget(self):
        engine = TokenEngine(["<think>\n[logical_deduction] " + "y" * 500], max_response_length=200)
        workflow = LabeledCoTWorkflow(engine, executor=None, repeat_label_retries=2, gen_chunk_tokens=64)
        episode = asyncio.run(workflow.run_with_termination_handling(TASK, "q1:0"))
        self.assertEqual(episode.termination_reason, TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)
        self.assertEqual(len(engine.calls), 4)
        self.assertEqual([c["max_tokens"] for c in engine.calls], [64, 64, 64, 8])


@unittest.skipUnless(LabeledCoTWorkflow is not None, "needs rLLM")
class ExtraSamplingParamsTest(unittest.TestCase):
    def test_keys_the_verl_engine_drops_are_passed_on(self):
        from omegaconf import OmegaConf

        from train_grpo import extra_sampling_params

        config = OmegaConf.create({"rllm": {"rollout": {
            "train": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 8192, "min_p": 0.0, "presence_penalty": 1.0},
            "val": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
        }}})
        self.assertEqual(extra_sampling_params(config),
                         {"train": {"min_p": 0.0, "presence_penalty": 1.0}, "val": {}})


if __name__ == "__main__":
    unittest.main()
