#!/usr/bin/env python3
"""Validate tasks/settings on a laptop; optionally check the actual model tokenizer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from labelcot.config import settings_from_env
from labelcot.data import dataset_summary, load_tasks
from labelcot.runtime import validate_settings


def check_prompts(tasks, tokenizer, max_length):
    failures = []
    maximum = 0
    if not tokenizer.chat_template:
        raise ValueError("checkpoint tokenizer has no chat template")
    for task in tasks:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["question"]}], tokenize=False, add_generation_prompt=True)
        length = len(tokenizer.encode(rendered, add_special_tokens=False))
        maximum = max(maximum, length)
        if length > max_length:
            failures.append(task["id"])
    if failures:
        raise ValueError(f"{len(failures)} prompts exceed {max_length} tokens: {failures[:5]}; "
                         "filter data or raise MAX_PROMPT_LENGTH before training")
    return maximum


def main():
    root = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default=os.environ.get("TRAIN_FILE", str(root / "train.jsonl")))
    parser.add_argument("--val-file", default=os.environ.get("VAL_FILE", str(root / "validation.jsonl")))
    parser.add_argument("--check-model", action="store_true", help="requires transformers and accessible SFT model")
    args = parser.parse_args()
    try:
        train, val = load_tasks(args.train_file), load_tasks(args.val_file)
        settings = settings_from_env()
        warnings = validate_settings(settings, train, val)
        report = {"train": dataset_summary(train), "validation": dataset_summary(val),
                  "settings": settings, "warnings": warnings}
        if args.check_model:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(settings["model"], trust_remote_code=False)
            report["longest_prompt_tokens"] = check_prompts(train + val, tokenizer, settings["max_prompt_length"])
        print(json.dumps(report, indent=2))
    except (ValueError, OSError) as exc:
        raise SystemExit(f"preflight failed: {exc}") from exc


if __name__ == "__main__":
    main()
