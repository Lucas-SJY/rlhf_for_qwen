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
