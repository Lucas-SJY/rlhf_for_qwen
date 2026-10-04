#!/usr/bin/env python3
"""Export a validated verl FSDP checkpoint to a merged HF model on CPU.

Exports stage in a unique sibling directory and publish only after validation. Existing
exports are never overwritten. Incomplete stages are retained for diagnosis on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from labelcot.checkpoints import resolve_step, validate_checkpoint
from labelcot.runtime import atomic_json, run_lock

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                   "generation_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")


def default_run_dir() -> Path:
    name = os.environ.get("RUN_NAME", "qwen3-8b-grpo-dayallen-v1")
    if os.environ.get("SMOKE_TEST", "false") == "true":
        name += "-smoke"
    return Path(os.environ.get("OUTPUT_ROOT", "/grpo/runs")) / name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=str(default_run_dir() / "checkpoints"))
    parser.add_argument("--step", type=int, default=int(os.environ.get("EXPORT_STEP", "0")))
    parser.add_argument("--base-model", default="", help="must match the recorded training base")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def validate_export(path: Path) -> None:
    if not (path / "config.json").is_file() or not (path / "tokenizer_config.json").is_file():
        raise ValueError("export missing model config or tokenizer config")
    weights = list(path.glob("model*.safetensors"))
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError("export has no complete safetensors weights")
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(index.read_text())["weight_map"].values())
        if any(not (path / name).is_file() for name in names):
            raise ValueError("export index references missing weight shards")


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_dir}")
    # Prevent checkpoint retention in a concurrently running trainer from deleting
    # shards midway through the CPU conversion.
    with run_lock(checkpoint_dir.parent):
        export_checkpoint(args, checkpoint_dir)


def export_checkpoint(args, checkpoint_dir):
    step = resolve_step(checkpoint_dir, args.step)
    actor = validate_checkpoint(checkpoint_dir, step)
    manifest_file = checkpoint_dir.parent / "run_manifest.json"
    if not manifest_file.is_file():
        raise ValueError("run_manifest.json missing; this exporter requires the recorded model/LoRA recipe")
    manifest = json.loads(manifest_file.read_text())
    settings = manifest["identity"]["settings"]
    base = args.base_model or settings["model"]
    if base != settings["model"]:
        raise ValueError("--base-model differs from the training model in run_manifest.json")
    out = Path(args.output_dir).resolve() if args.output_dir else checkpoint_dir.parent / "export" / f"global_step_{step}"
    if out.exists():
        raise ValueError(f"export exists: {out}; choose another --output-dir")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".export_{step}_", dir=out.parent))
    converted, staged = work / "converted", work / "model"
    try:
        subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                        "--local_dir", str(actor), "--target_dir", str(converted)], check=True)
        adapter = converted / "lora_adapter"
        if settings["lora_rank"] > 0:
            if not (adapter / "adapter_config.json").is_file():
                raise ValueError("LoRA run produced no adapter; refusing to export unchanged base weights")
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu",
                                                       trust_remote_code=False)
            model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload(safe_merge=True)
            model.save_pretrained(staged, safe_serialization=True, max_shard_size="5GB")
            shutil.copytree(adapter, staged / "lora_adapter")
        else:
            if adapter.exists():
                raise ValueError("unexpected adapter in a full-parameter checkpoint")
            converted.rename(staged)
        from transformers import AutoConfig, AutoTokenizer

        AutoTokenizer.from_pretrained(base, trust_remote_code=False).save_pretrained(staged)
        if Path(base).is_dir():
            for name in TOKENIZER_FILES:
                if (Path(base) / name).is_file():
                    shutil.copy2(Path(base) / name, staged / name)
        validate_export(staged)
        AutoConfig.from_pretrained(staged, local_files_only=True, trust_remote_code=False)
        AutoTokenizer.from_pretrained(staged, local_files_only=True, trust_remote_code=False)
        atomic_json(staged / "export_manifest.json", {"step": step, "base_model": base,
                    "checkpoint": str(actor), "training_run": manifest})
        if out.exists():
            raise ValueError(f"another export created {out}")
        staged.rename(out)
    except BaseException:
        print(f"[export] failed; diagnostic staging retained at {work}", file=sys.stderr)
        raise
    else:
        shutil.rmtree(work)
        print(f"[export] done: {out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"export failed: {exc}") from exc
