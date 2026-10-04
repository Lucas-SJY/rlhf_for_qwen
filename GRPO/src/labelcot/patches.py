"""Runtime fix for rLLM's unified trainer (commit 3b40c37) on the verl backend.

Final checkpoint. ``VerlBackend`` saves only when ``global_step % save_freq == 0`` and its
``on_train_end`` does nothing, so up to ``save_freq - 1`` steps of training are lost when
a run ends. The patch records the last trained and last saved step in ``on_batch_end`` and
saves once more after a run that finished normally, when the two differ. It saves nothing
after a crash.

The trainer is built inside a Ray actor (rLLM's ``VerlTaskRunner``), so a patch applied
in the driver would not reach it. ``workflow.py`` calls ``apply_patches()`` when it is
imported. The actor imports that module when it receives the workflow class, before
training starts.
"""

from __future__ import annotations

import math
import os


def check_training_health(metrics: dict, zero_streak: int, limit: int) -> int:
    """Detect non-finite optimization and prolonged zero GRPO signal."""
    for name, value in metrics.items():
        if (name in ("actor/pg_loss", "actor/grad_norm", "actor/kl_loss")
                or (name.startswith("reward/") and name.endswith("/mean"))):
            if not math.isfinite(float(value)):
                raise RuntimeError(f"non-finite training metric {name}: {value}")
    effective = [float(value) for name, value in metrics.items()
                 if name.startswith("batch/") and name.endswith("/fractions/effective")]
    if not effective:
        return zero_streak
    zero_streak = zero_streak + 1 if all(value == 0 for value in effective) else 0
    metrics["health/zero_signal_steps"] = zero_streak
    if limit > 0 and zero_streak >= limit:
        raise RuntimeError(f"{zero_streak} consecutive batches have no GRPO signal; inspect truncation, "
                           "reward variance and episode logs. Increase MAX_ZERO_SIGNAL_STEPS or set 0 to disable.")
    return zero_streak


def apply_patches() -> None:
    from rllm.trainer.unified_trainer import UnifiedTrainer
    from rllm.trainer.verl.utils import save_checkpoint
    from rllm.trainer.verl.verl_backend import VerlBackend

    if getattr(VerlBackend, "_labelcot_patched", False):
        return

    for owner, method in ((VerlBackend, "on_batch_end"), (UnifiedTrainer, "_fit_on_policy")):
        if not callable(getattr(owner, method, None)):
            raise RuntimeError(f"rLLM API changed: missing {owner.__name__}.{method}; use the pinned image")

    original_on_batch_end = VerlBackend.on_batch_end
    original_fit_on_policy = UnifiedTrainer._fit_on_policy

    async def on_batch_end(self, trainer_state):
        step = trainer_state.global_step
        trained = trainer_state.has_backend_batch
        await original_on_batch_end(self, trainer_state)
        if trained:
            self._labelcot_last_trained_step = step
        save_freq = self.config.trainer.save_freq
        if save_freq > 0 and step % save_freq == 0:
            self._labelcot_last_saved_step = step
        if trained:
            self._labelcot_zero_streak = check_training_health(
                trainer_state.metrics, getattr(self, "_labelcot_zero_streak", 0),
                int(os.environ.get("MAX_ZERO_SIGNAL_STEPS", "5")))

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
