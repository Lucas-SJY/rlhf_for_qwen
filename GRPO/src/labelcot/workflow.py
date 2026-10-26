"""rLLM workflow: sample one answer per call and score it with the label reward.

GRPO calls this workflow ``group_size`` times per question (rLLM repeats each task and
groups the episodes by the task's ``id``). One call = one rollout = one scored episode.

The deliberate difference from rLLM's stock ``SimpleWorkflow`` is the prompt: it is
rendered with the checkpoint's own chat template (``tokenizer.apply_chat_template``),
i.e. exactly the way the SFT data was tokenised. rLLM's ``QwenChatTemplateParser`` would
prepend a default "You are Qwen, created by Alibaba Cloud..." system turn that the SFT
model never saw. Generation still goes through rLLM's token-in/token-out path, so the
trainer gets the exact sampled token ids and their logprobs.

The tag format is a rule, not a reward term: a completion whose thought has a tag
outside the eight labels is thrown away and sampled again, up to ``max_label_retries``
times. Only the kept completion becomes the trajectory; one that still has an invalid
tag after the last retry gets no label credit in the reward.

Sampling parameters are rLLM's: ``rllm.rollout.train`` / ``rllm.rollout.val``. Its verl
engine (``VerlEngine``) builds its sampling params from temperature, top_p and top_k
only; any other key there (presence_penalty, min_p, ...) arrives as ``extra_sampling``
(train_grpo.py) and is passed to ``get_token_output_from_token_input``, which merges its
keyword arguments into vLLM's ``SamplingParams``.

A thought stuck on one step type is stopped and sampled again: during training (not
validation), while regenerations are left (``repeat_label_retries``), the completion is
generated in chunks of ``gen_chunk_tokens`` tokens, each chunk continuing from the prompt
plus everything generated so far, and after every chunk the thought is checked for a run
of ``repeat_min_paragraphs`` consecutive paragraphs with the same label longer than
``repeat_min_tokens`` tokens (labelcot.reward.repeated_label_run). A run stops the
completion; it is discarded and sampled again. The last attempt is generated in one call;
if it still has such a run (``repeat_mask_final``), its episode ends as
MAX_TURNS_EXCEEDED, a reason this single-turn workflow has no other use for, which the
trainer's compact filtering drops like a cut-off answer (entrypoint.sh). Chunking changes one thing about the sampling: vLLM applies
the presence penalty to the tokens of the current call only, so tokens from earlier chunks
are no longer penalised.

A completion cut off by the length limit ends the episode with
MAX_RESPONSE_LENGTH_EXCEEDED; the trainer's compact filtering drops those episodes from
the batch (entrypoint.sh, MASK_TRUNCATED), so they are neither rewarded nor penalised.
"""

from __future__ import annotations

from rllm.types import Action, Step, Trajectory
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

from labelcot.patches import apply_patches
from labelcot.reward import RewardBreakdown, RewardWeights, invalid_labels, repeated_label_run, score_completion, split_completion

# Runs in every process that imports this module, including rLLM's Ray TaskRunner actor
# (it imports the module when it receives the workflow class). See patches.py.
apply_patches()


class LabeledCoTWorkflow(Workflow):
    def __init__(self, rollout_engine, executor, reward_weights: dict | None = None,
                 max_label_retries: int = 3, extra_sampling: dict | None = None,
                 repeat_label_retries: int = 0, repeat_min_paragraphs: int = 3, repeat_min_tokens: int = 200,
                 gen_chunk_tokens: int = 1024, repeat_mask_final: bool = False, **kwargs):
        super().__init__(rollout_engine, executor, **kwargs)
        self.reward_weights = RewardWeights(**(reward_weights or {}))
        self.max_label_retries = max(0, int(max_label_retries))
        # Repeated-label runs: 0 retries turns the check off (one call per completion).
        self.repeat_label_retries = max(0, int(repeat_label_retries))
        self.repeat_min_paragraphs = max(1, int(repeat_min_paragraphs))
        self.repeat_min_tokens = int(repeat_min_tokens)
        self.gen_chunk_tokens = max(0, int(gen_chunk_tokens))
        # A last attempt that still has a run is dropped from training (see the docstring).
        self.repeat_mask_final = bool(repeat_mask_final)
        # {"train": {...}, "val": {...}}: rllm.rollout.* keys the verl engine leaves out.
        self.extra_sampling = {mode: dict((extra_sampling or {}).get(mode) or {}) for mode in ("train", "val")}
        self._breakdown: RewardBreakdown | None = None

    def reset(self, task: dict | None = None, uid: str | None = None) -> None:
        super().reset(task, uid)
        self._breakdown = None

    async def run(self, task: dict, uid: str, **kwargs):
        self.reset(task, uid)
        engine = self.rollout_engine
        tokenizer = engine.tokenizer

        # The task's chat-format prompt if it has one (grpo_try), else the bare question as
        # one user turn (bespoke-v2). Default enable_thinking: the prompt stops at
        # "<|im_start|>assistant\n" and the model opens its own <think> block.
        messages = list(task.get("prompt") or [{"role": "user", "content": task["question"]}])
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # The unified trainer sets engine.is_validation; the older workflow trainer set
        # engine.validate. Honour both so validation always uses its own sampling params.
        is_validation = bool(getattr(engine, "is_validation", False) or getattr(engine, "validate", False))
        sampling = dict(engine.val_sampling_params if is_validation else engine.train_sampling_params)
        sampling.update(self.extra_sampling["val" if is_validation else "train"])

        # Sample until the thought has no repeated-label run (at most repeat_label_retries
        # extra times, training only) and no tag outside the eight labels (at most
        # max_label_retries extra times). Rejected samples are discarded, not trained on.
        repeat_budget = 0 if is_validation else self.repeat_label_retries
        label_retries = repeat_retries = discarded = 0
        while True:
            # Raises TerminationEvent(MAX_PROMPT_LENGTH_EXCEEDED) for an overlong prompt; the
            # base class turns that into an empty episode that the trainer drops.
            token_output, stopped = await self._generate(engine, prompt_ids, uid, sampling, interruptible=repeat_retries < repeat_budget)
            if stopped:
                repeat_retries += 1
                discarded += len(token_output.token_ids)
                continue
            output = engine.assemble_model_output(token_input=prompt_ids, token_output=token_output, prompt_ids=prompt_ids)
            if invalid_labels(output.text or "") and label_retries < self.max_label_retries:
                label_retries += 1
                continue
            break
        output.weight_version = engine.weight_version

        truncated = output.finish_reason == "length"
        breakdown = score_completion(output.text or "", task, truncated=truncated, weights=self.reward_weights)
        breakdown.label_retries = label_retries
        breakdown.repeat_retries = repeat_retries
        breakdown.repeat_discarded_tokens = discarded
        breakdown.repeat_in_final = self._repeat_run(engine, output.text or "") is not None
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

        if truncated:
            reason = TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
        elif repeat_budget and self.repeat_mask_final and breakdown.repeat_in_final:
            reason = TerminationReason.MAX_TURNS_EXCEEDED
        else:
            reason = TerminationReason.ENV_DONE
        raise TerminationEvent(reason)

    def _repeat_run(self, engine, text: str):
        thought, _, _ = split_completion(text)
        tokenizer = engine.tokenizer
        return repeated_label_run(thought, lambda paragraph: len(tokenizer.encode(paragraph, add_special_tokens=False)),
                                  self.repeat_min_paragraphs, self.repeat_min_tokens)

    async def _generate(self, engine, prompt_ids, uid, sampling, interruptible):
        """One completion as (token output, stopped). Interruptible ones are generated in chunks
        and stopped (stopped=True, the tokens so far in the output) at a repeated-label run."""
        if not interruptible or self.gen_chunk_tokens <= 0:
            return await engine.get_token_output_from_token_input(token_input=prompt_ids, application_id=uid, **sampling), False
        sampling = dict(sampling)
        budget = int(sampling.pop("max_tokens", 0) or engine.max_response_length)
        tokenizer = engine.tokenizer
        stop_ids = {getattr(tokenizer, "eos_token_id", None)}
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            stop_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        token_ids, log_probs = [], []
        while True:
            size = min(self.gen_chunk_tokens, budget - len(token_ids))
            # The same application_id keeps every chunk on one vLLM server, whose prefix cache
            # then holds the prompt and the chunks before. Only the first call is the prompt.
            chunk = await engine.get_token_output_from_token_input(
                token_input=prompt_ids + token_ids, application_id=uid, max_tokens=size,
                enforce_max_prompt_length=not token_ids, **sampling)
            new = list(chunk.token_ids)
            token_ids += new
            log_probs += list(chunk.log_probs or [])
            # Checked after the last chunk too, so whether a run counts does not depend on
            # where the chunk boundaries fall.
            if self._repeat_run(engine, tokenizer.decode(token_ids, skip_special_tokens=True)) is not None:
                chunk.token_ids = token_ids
                return chunk, True
            if len(new) < size or (new and new[-1] in stop_ids) or len(token_ids) >= budget:
                break
        chunk.token_ids = token_ids
        chunk.log_probs = log_probs or None
        chunk.stop_reason = "length" if len(token_ids) >= budget else "stop"
        return chunk, False

    def assign_episode_correctness(self, episode) -> None:
        # Answer correctness, not "reward > 0": the reward is almost always positive, and
        # rLLM uses is_correct for pass@k in validation.
        episode.is_correct = bool(self._breakdown is not None and self._breakdown.answer_correct)

    def collect_metrics(self, episode) -> None:
        episode.metrics = self._breakdown.metrics() if self._breakdown is not None else {}
