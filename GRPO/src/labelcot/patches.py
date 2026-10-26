"""Runtime changes to rLLM's unified trainer (commit 3b40c37) on the verl backend.

Final checkpoint. ``VerlBackend`` saves only when ``global_step % save_freq == 0`` and its
``on_train_end`` does nothing, so up to ``save_freq - 1`` steps of training are lost when
a run ends. The patch records the last trained and last saved step in ``on_batch_end`` and
saves once more after a run that finished normally, when the two differ. It saves nothing
after a crash.

wandb sections. rLLM logs every per-answer metric as ``batch/<name>`` (the mean over the
step's answers that report it), so the reward terms, the accuracies, the cut-off and loop
shares and the retry counts sit among some 180 other keys. At the end of each training step,
once verl has added its length metrics, the patch copies them into sections of their own
(reward_breakdown/, rollout_accuracy/, rollout_length/, loop/, retry/), one line chart per
key over the training steps (METRIC_SECTIONS), and adds the number of cut-off rollouts
(rollout_length/cut_off_count). The source keys stay as they are.

The trainer is built inside a Ray actor (rLLM's ``VerlTaskRunner``), so a patch applied
in the driver would not reach it. ``workflow.py`` calls ``apply_patches()`` when it is
imported. The actor imports that module when it receives the workflow class, before
training starts.
"""

from __future__ import annotations

# wandb key -> the batch/ key it copies (labelcot.reward.RewardBreakdown.metrics).
METRIC_SECTIONS = {
    # mean reward of the answers that get one (not cut off); total = format + correctness
    "reward_breakdown/total": "batch/reward",
    "reward_breakdown/format": "batch/reward_label",
    "reward_breakdown/correctness": "batch/reward_correct",
    # every rollout of the step, a cut-off one counting as wrong
    "rollout_accuracy/all": "batch/answer_correct",
    # the rollouts that finished within the length limit
    "rollout_accuracy/finished": "batch/answer_correct_finished",
    # rollouts generated, the share cut off at the response limit, and their length in tokens
    # (verl's, over every rollout of the step, a cut-off one counting at the limit)
    "rollout_length/rollouts": "groups/num_trajs_before_filter",
    "rollout_length/cut_off_share": "batch/truncated",
    "rollout_length/mean_tokens": "response_length/mean",
    "rollout_length/max_tokens": "response_length/max",
    "rollout_length/min_tokens": "response_length/min",
    # answers ending in a repetition loop, partly repeating, and loops among the cut-off ones
    # (labelcot.reward.loop_label)
    "loop/loop": "batch/loop",
    "loop/partial": "batch/loop_partial",
    "loop/among_cut_off": "batch/loop_among_cut",
    # regenerations per rollout: for a repeated-label run, and for a tag outside the labels
    "retry/repeat_retries": "batch/repeat_retries",
    "retry/repeat_discarded_tokens": "batch/repeat_discarded_tokens",
    "retry/repeat_in_final": "batch/repeat_in_final",
    "retry/label_retries": "batch/label_retries",
    # share of rollouts dropped because the last attempt still had a repeated-label run
    "retry/masked": "batch/termination_reason/max_turns_exceeded",
}


def add_metric_sections(metrics: dict) -> None:
    for name, source in METRIC_SECTIONS.items():
        if source in metrics:
            metrics[name] = metrics[source]
    # The cut-off share is a mean over the step's rollouts; times their number it is a count.
    if "batch/truncated" in metrics and "groups/num_trajs_before_filter" in metrics:
        metrics["rollout_length/cut_off_count"] = round(float(metrics["batch/truncated"]) * float(metrics["groups/num_trajs_before_filter"]))


def apply_patches() -> None:
    from rllm.trainer.unified_trainer import UnifiedTrainer
    from rllm.trainer.verl.utils import save_checkpoint
    from rllm.trainer.verl.verl_backend import VerlBackend

    if getattr(VerlBackend, "_labelcot_patched", False):
        return

    original_on_batch_end = VerlBackend.on_batch_end
    original_fit_on_policy = UnifiedTrainer._fit_on_policy

    async def on_batch_end(self, trainer_state):
        step = trainer_state.global_step
        trained = trainer_state.has_backend_batch
        await original_on_batch_end(self, trainer_state)
        # verl adds response_length/* here; the trainer logs trainer_state.metrics right after.
        add_metric_sections(trainer_state.metrics)
        if trained:
            self._labelcot_last_trained_step = step
        save_freq = self.config.trainer.save_freq
        if save_freq > 0 and step % save_freq == 0:
            self._labelcot_last_saved_step = step

    async def _fit_on_policy(self, trainer_state):
        await original_fit_on_policy(self, trainer_state)
        backend = self.backend
        if not isinstance(backend, VerlBackend):
            return
        last_trained = getattr(backend, "_labelcot_last_trained_step", None)
        if last_trained is not None and last_trained != getattr(backend, "_labelcot_last_saved_step", None):
            print(f"[labelcot] saving final checkpoint for step {last_trained}")
            save_checkpoint(backend.config, last_trained, backend.actor_rollout_wg, train_dataloader=trainer_state.train_dataloader)
            backend._labelcot_last_saved_step = last_trained

    VerlBackend.on_batch_end = on_batch_end
    UnifiedTrainer._fit_on_policy = _fit_on_policy
    VerlBackend._labelcot_patched = True
