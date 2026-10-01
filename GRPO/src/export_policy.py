#!/usr/bin/env python3
"""Turn a GRPO actor checkpoint into a plain Hugging Face model directory.

verl stores the actor as FSDP shards of the LoRA-wrapped model. This script:

    1. runs verl's own converter (``python -m verl.model_merger merge``), which writes the
       base weights plus the adapter as ``lora_adapter/`` in PEFT format;
    2. merges the adapter into the SFT checkpoint with PEFT and saves the result, with the
       SFT checkpoint's tokenizer and chat template, as a normal model directory that
       ../train/evaluate can load.

Output: <OUTPUT_ROOT>/<RUN_NAME>/export/global_step_<N>/{model files, lora_adapter/}.
CPU only; needs roughly 3x the model size in RAM.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Files that define how prompts are rendered; copied verbatim from the SFT checkpoint so
# the exported model is prompted exactly like the one GRPO started from.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                   "special_tokens_map.json", "vocab.json", "merges.txt")


def parse_args() -> argparse.Namespace:
    run_dir = Path(os.environ.get("OUTPUT_ROOT", "/grpo/runs")) / os.environ.get("RUN_NAME", "qwen3-8b-grpo-labels-v1")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", default=str(run_dir / "checkpoints"))
    p.add_argument("--step", type=int, default=int(os.environ.get("EXPORT_STEP", "0")),
                   help="global step to export, 0 means the latest")
    p.add_argument("--base-model", default=os.environ.get("POLICY_MODEL_PATH", "/data/runs/qwen3-8b-sft-v3"),
                   help="the SFT checkpoint GRPO started from")
    p.add_argument("--output-dir", default="", help=f"defaults to {run_dir}/export/global_step_<N>")
    return p.parse_args()


def resolve_step(checkpoint_dir: Path, step: int) -> int:
    if step:
        return step
    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise SystemExit(f"no {tracker}; has the run saved a checkpoint yet?")
    return int(tracker.read_text().strip())


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    step = resolve_step(checkpoint_dir, args.step)
    actor_dir = checkpoint_dir / f"global_step_{step}" / "actor"
    if not actor_dir.is_dir():
        raise SystemExit(f"{actor_dir} does not exist")

    out_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir.parent / "export" / f"global_step_{step}"
    staging = out_dir.parent / f".staging_global_step_{step}"
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[export] actor checkpoint : {actor_dir}")
    print(f"[export] output           : {out_dir}", flush=True)

    subprocess.run(
        [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
         "--local_dir", str(actor_dir), "--target_dir", str(staging)],
        check=True,
    )

    import torch
    from transformers import AutoModelForCausalLM

    adapter_dir = staging / "lora_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    if adapter_dir.is_dir():
        from peft import PeftModel

        print(f"[export] merging {adapter_dir} into {args.base_model}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
        model.save_pretrained(out_dir)
        shutil.copytree(adapter_dir, out_dir / "lora_adapter", dirs_exist_ok=True)
    else:
        # Full-parameter run (LORA_RANK=0): the converter already wrote the final weights.
        print("[export] no LoRA adapter found, using the converted weights as they are", flush=True)
        for item in staging.iterdir():
            shutil.move(str(item), out_dir / item.name)

    if Path(args.base_model).is_dir():
        for name in TOKENIZER_FILES:
            src = Path(args.base_model) / name
            if src.is_file():
                shutil.copy2(src, out_dir / name)
    else:
        # A Hub id such as Qwen/Qwen3-0.6B: save its tokenizer and chat template from the cache.
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(args.base_model).save_pretrained(out_dir)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[export] done: {out_dir}")


if __name__ == "__main__":
    main()
