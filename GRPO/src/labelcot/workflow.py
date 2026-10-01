"""rLLM workflow: sample one answer per call and score it with the label reward.

GRPO calls this workflow ``group_size`` times per question (rLLM repeats each task and
groups the episodes by the task's ``id``). One call = one rollout = one scored episode.

The deliberate difference from rLLM's stock ``SimpleWorkflow`` is the prompt: it is
rendered with the checkpoint's own chat template (``tokenizer.apply_chat_template``),
i.e. exactly the way the SFT data was tokenised. rLLM's ``QwenChatTemplateParser`` would
prepend a default "You are Qwen, created by Alibaba Cloud..." system turn that the SFT
model never saw. Generation still goes through rLLM's token-in/token-out path, so the
trainer gets the exact sampled token ids and their logprobs.
"""

from __future__ import annotations

from rllm.types import Action, Step, Trajectory
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

from labelcot.patches import apply_patches
from labelcot.reward import RewardBreakdown, RewardWeights, score_completion

# Runs in every process that imports this module, including rLLM's Ray TaskRunner actor
# (it imports the module when it receives the workflow class). See patches.py.
apply_patches()


class LabeledCoTWorkflow(Workflow):
    def __init__(self, rollout_engine, executor, reward_weights: dict | None = None, **kwargs):
        super().__init__(rollout_engine, executor, **kwargs)
        self.reward_weights = RewardWeights(**(reward_weights or {}))
        self._breakdown: RewardBreakdown | None = None

    def reset(self, task: dict | None = None, uid: str | None = None) -> None:
        super().reset(task, uid)
        self._breakdown = None

    async def run(self, task: dict, uid: str, **kwargs):
        self.reset(task, uid)
        engine = self.rollout_engine
        tokenizer = engine.tokenizer

        # Bare question, default enable_thinking: the prompt stops at
        # "<|im_start|>assistant\n" and the model opens its own <think> block.
        messages = [{"role": "user", "content": task["question"]}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # The unified trainer sets engine.is_validation; the older workflow trainer set
        # engine.validate. Honour both so validation always uses its own sampling params.
        is_validation = bool(getattr(engine, "is_validation", False) or getattr(engine, "validate", False))
        sampling = dict(engine.val_sampling_params if is_validation else engine.train_sampling_params)

        # Raises TerminationEvent(MAX_PROMPT_LENGTH_EXCEEDED) for an overlong prompt; the
        # base class turns that into an empty episode that the trainer drops.
        token_output = await engine.get_token_output_from_token_input(token_input=prompt_ids, application_id=uid, **sampling)
        output = engine.assemble_model_output(token_input=prompt_ids, token_output=token_output, prompt_ids=prompt_ids)
        output.weight_version = engine.weight_version

        truncated = output.finish_reason == "length"
        breakdown = score_completion(output.text or "", task, truncated=truncated, weights=self.reward_weights)
        self._breakdown = breakdown

        step = Step(
            chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
            thought=output.reasoning or "",
            action=Action(action=output.content),
            model_response=output.text or "",
            model_output=output,
            reward=breakdown.reward,
            done=True,
            metadata={"reward_breakdown": breakdown.to_dict()},
        )
        self.commit(trajectory=Trajectory(name="policy", task=task, steps=[step]))

        reason = TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED if truncated else TerminationReason.ENV_DONE
        raise TerminationEvent(reason)

    def assign_episode_correctness(self, episode) -> None:
        # Answer correctness, not "reward > 0": the reward is almost always positive, and
        # rLLM uses is_correct for pass@k in validation.
        episode.is_correct = bool(self._breakdown is not None and self._breakdown.answer_correct)

    def collect_metrics(self, episode) -> None:
        episode.metrics = self._breakdown.metrics() if self._breakdown is not None else {}
