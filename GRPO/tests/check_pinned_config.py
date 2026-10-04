#!/usr/bin/env python3
"""Optional offline contract check using unpacked upstream source trees.

Requires hydra-core/PyYAML, but does not import Ray, vLLM or GPU workers.
python GRPO/tests/check_pinned_config.py --rllm-source /path/to/rllm --verl-source /path/to/verl
This checks actual launch overrides and schema composition; it is not a GPU test.
"""

import argparse
import ast
import logging
import os
import tempfile
import warnings
from typing import Any, Callable
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from test_entrypoint import capture_entrypoint
from test_runtime import task
from labelcot.config import recipe_identity, settings_from_config
from labelcot.runtime import validate_settings


def load_sync_config(source):
    # Run the pinned pure-OmegaConf sync function with its actual mappings, avoiding
    # imports of torch/Ray worker infrastructure elsewhere in the same source module.
    tree = ast.parse(source.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "sync_config":
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id.startswith("_") and t.id.isupper() for t in targets):
                selected.append(node)
    helpers = ast.parse((source.parent.parent / "algorithms/config.py").read_text())
    selected = [node for node in helpers.body if isinstance(node, ast.FunctionDef)
                and node.name in ("_explicit_override_keys", "_plain", "sync_shared_keys")] + selected
    namespace = {"Any": Any, "Callable": Callable, "OmegaConf": OmegaConf, "DictConfig": DictConfig, "open_dict": open_dict,
                 "logger": logging.getLogger("config-check"), "warnings": warnings}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["sync_config"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rllm-source", type=Path, required=True)
    parser.add_argument("--verl-source", type=Path, required=True)
    args = parser.parse_args()
    config_dir = args.rllm_source.resolve() / "rllm/trainer/config"
    verl_config = args.verl_source.resolve() / "verl/trainer/config"
    sync_config = load_sync_config(args.rllm_source / "rllm/trainer/verl/utils.py")
    for smoke in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            captured = capture_entrypoint(Path(tmp), smoke)
        overrides = captured["args"][1:]
        overrides.append(f"hydra.searchpath=[file://{verl_config}]")
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            config = compose(config_name="unified", overrides=overrides)
        sync_config(config, hydra_overrides=overrides)
        with patch.dict(os.environ, {"MAX_ZERO_SIGNAL_STEPS": captured["zero_limit"]}):
            settings = settings_from_config(config)
        validate_settings(settings, [task(i) for i in range(16)], [task(100)])
        recipe_identity(config)
        assert config.algorithm.adv_estimator == "grpo"
        assert config.actor_rollout_ref.actor.use_kl_loss
        assert config.actor_rollout_ref.actor.kl_loss_coef == 0.001
        assert config.actor_rollout_ref.rollout.n == (4 if smoke else 8)
        assert config.actor_rollout_ref.actor.fsdp_config.model_dtype == "bf16"
        assert config.actor_rollout_ref.actor.checkpoint.save_contents == ["model", "optimizer", "extra"]
        assert not config.rllm.async_training.enable
        print(f"{'smoke' if smoke else 'normal'}: {len(overrides)} overrides composed and synced against pinned sources")


if __name__ == "__main__":
    main()
