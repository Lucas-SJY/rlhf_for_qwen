import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_policy import default_run_dir, validate_export
from labelcot.checkpoints import resolve_step, validate_checkpoint
from labelcot.patches import apply_patches, check_training_health


class CheckpointTest(unittest.TestCase):
    def test_resume_requires_all_ranks_optimizer_rng_and_dataloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actor = root / "global_step_2/actor"
            (actor / "huggingface").mkdir(parents=True)
            (actor / "fsdp_config.json").write_text(json.dumps({"world_size": 2}))
            (actor / "huggingface/config.json").write_text("{}")
            for kind in ("model", "optim", "extra_state"):
                for rank in range(2):
                    (actor / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"fixture")
            self.assertEqual(validate_checkpoint(root, 2), actor)
            with self.assertRaisesRegex(ValueError, "data.pt"):
                validate_checkpoint(root, 2, for_resume=True)
            (actor.parent / "data.pt").write_bytes(b"fixture")
            self.assertEqual(validate_checkpoint(root, 2, for_resume=True), actor)
            (actor / "model_world_size_2_rank_1.pt").unlink()
            with self.assertRaisesRegex(ValueError, "rank_1"):
                validate_checkpoint(root, 2)

    def test_tracker_and_smoke_export_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                resolve_step(root)
            (root / "latest_checkpointed_iteration.txt").write_text("2")
            self.assertEqual(resolve_step(root), 2)
            self.assertEqual(resolve_step(root, 3), 3)
        with patch.dict(os.environ, {"OUTPUT_ROOT": "/tmp/runs", "RUN_NAME": "test", "SMOKE_TEST": "true"}):
            self.assertEqual(default_run_dir(), Path("/tmp/runs/test-smoke"))

    def test_export_rejects_incomplete_shard_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for file in ("config.json", "tokenizer_config.json"):
                (root / file).write_text("{}")
            (root / "model-00001-of-00002.safetensors").write_bytes(b"fixture")
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "missing weight shards"):
                validate_export(root)


class PatchTest(unittest.TestCase):
    def run_fixture(self, count, fail=False):
        saves = Mock()

        class Backend:
            def __init__(self):
                self.config = types.SimpleNamespace(trainer=types.SimpleNamespace(save_freq=2))
                self.actor_rollout_wg = object()
            async def on_batch_end(self, state):
                if state.global_step % 2 == 0:
                    saves(self.config, state.global_step, self.actor_rollout_wg, train_dataloader=state.train_dataloader)

        class Trainer:
            def __init__(self):
                self.backend = Backend()
            async def _fit_on_policy(self, state):
                for step in range(1, count + 1):
                    state.global_step = step
                    await self.backend.on_batch_end(state)
                if fail:
                    raise RuntimeError("simulated training failure")
                # Pinned trainer increments at epoch exhaustion; patch must save the
                # last trained step, not this next-loop counter.
                state.global_step += 1

        modules = {}
        for name, members in {
            "rllm.trainer.unified_trainer": {"UnifiedTrainer": Trainer},
            "rllm.trainer.verl.verl_backend": {"VerlBackend": Backend},
            "rllm.trainer.verl.utils": {"save_checkpoint": saves},
        }.items():
            module = types.ModuleType(name)
            module.__dict__.update(members)
            modules[name] = module
        state = types.SimpleNamespace(global_step=0, has_backend_batch=True, train_dataloader="loader", metrics={})
        with patch.dict(sys.modules, modules):
            apply_patches()
            first = Trainer._fit_on_policy
            apply_patches()
            self.assertIs(first, Trainer._fit_on_policy)
            if fail:
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    asyncio.run(Trainer()._fit_on_policy(state))
            else:
                asyncio.run(Trainer()._fit_on_policy(state))
        return [call.args[1] for call in saves.call_args_list]

    def test_final_save_idempotence_and_failure(self):
        self.assertEqual(self.run_fixture(3), [2, 3])
        self.assertEqual(self.run_fixture(4), [2, 4])
        self.assertEqual(self.run_fixture(3, fail=True), [2])

    def test_health_guard(self):
        metrics = {"batch/policy/fractions/effective": 0}
        streak = check_training_health(metrics, 0, 2)
        self.assertEqual(streak, 1)
        with self.assertRaisesRegex(RuntimeError, "no GRPO signal"):
            check_training_health(metrics, streak, 2)
        self.assertEqual(check_training_health({"batch/policy/fractions/effective": 0.5}, 4, 5), 0)
        self.assertEqual(check_training_health(metrics, 10, 0), 11)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            check_training_health({"actor/grad_norm": float("nan")}, 0, 0)


if __name__ == "__main__":
    unittest.main()
