#!/usr/bin/env python3
"""Evaluate a fine-tuned model on MATH-500.

Reports two numbers side by side, which together are the point of the context
compression experiment:

    accuracy            -- did the model still get the answer right
    completion tokens   -- how much reasoning it needed to get there

Everything is read-only: the model comes from the PVC, results are written to a new
directory under it. Nothing in the training pipeline is touched.

The scorer is dependency-free on purpose. sympy.parsing.latex needs antlr, which the
training image does not have, so answers are compared with LaTeX normalisation plus a
numeric fallback instead.

Other datasets work when their rows have `problem` and `answer` (EVAL_DATASET, with
EVAL_DATASET_CONFIG for one that has configs, e.g. ByteDance-Seed/BeyondAIME or
RUC-AIBOX/OlymMATH en-hard). MATH-500's `unique_id`, `subject` and `level` are optional:
a row without `unique_id` is named `<split>/<index>`, and an answer that is a number is
compared as its text.

EVAL_SUITE evaluates several datasets in one run, loading the model once: entries
`dataset|config|split` separated by `;` (config may be empty). Each goes to
<output-dir>/<dataset name>[-<config>]/<run-name>/, as a single run of it would with
that run name; a dataset that fails is reported and the rest still run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from fractions import Fraction
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# The training prompts all started with this sentence, so evaluation has to use it too;
# a different instruction would measure a different distribution than the one trained on.
INSTRUCTION = "Return your final response within \\boxed{}."


def env(key: str, default):
    raw = os.environ.get(key, "")
    if raw == "":
        return default
    return type(default)(raw) if default is not None else raw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=env("EVAL_MODEL", "/data/runs/qwen3-8b-sft"),
                   help="local path or HF repo id; point it at the base model to get a baseline")
    p.add_argument("--dataset", default=env("EVAL_DATASET", "HuggingFaceH4/MATH-500"))
    p.add_argument("--split", default=env("EVAL_SPLIT", "test"))
    p.add_argument("--dataset-config", default=env("EVAL_DATASET_CONFIG", ""),
                   help="the dataset's config (subset) name, for datasets that have several")
    p.add_argument("--suite", default=env("EVAL_SUITE", ""),
                   help="several datasets as 'dataset|config|split;...'; overrides --dataset")
    p.add_argument("--output-dir", default=env("EVAL_OUTPUT_DIR", "/data/eval"))
    p.add_argument("--run-name", default=env("EVAL_RUN_NAME", ""), help="defaults to the model directory name")
    p.add_argument("--limit", type=int, default=env("EVAL_LIMIT", 0), help="only the first N problems, 0 means all")
    p.add_argument("--batch-size", type=int, default=env("EVAL_BATCH_SIZE", 16))
    p.add_argument("--max-new-tokens", type=int, default=env("EVAL_MAX_NEW_TOKENS", 8192),
                   help="a thinking model emits the whole trace before the answer, so this "
                        "has to cover both or every run truncates mid-thought")
    p.add_argument("--top-p", type=float, default=env("EVAL_TOP_P", 1.0),
                   help="nucleus sampling, used only when temperature > 0 (1.0 = off)")
    p.add_argument("--top-k", type=int, default=env("EVAL_TOP_K", 0),
                   help="top-k sampling, used only when temperature > 0 (0 = off); Qwen3's "
                        "thinking-mode recommendation is temperature 0.6, top-p 0.95, top-k 20")
    p.add_argument("--temperature", type=float, default=env("EVAL_TEMPERATURE", 0.0),
                   help="0 means greedy, which keeps the run reproducible")
    p.add_argument("--enable-thinking", default=env("EVAL_ENABLE_THINKING", "true"),
                   choices=["true", "false"],
                   help="true leaves the template at its default, so the prompt ends at "
                        "'<|im_start|>assistant' and the model opens its own <think> block -- "
                        "what a model trained on thinking-turn data expects. false prefills an "
                        "empty <think></think>, which is right only for models trained that way")
    p.add_argument("--seed", type=int, default=env("SEED", 42))
    return p.parse_args()


# ---------------------------------------------------------------------------
# answer extraction and comparison
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...}, matching braces properly."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        # \boxed 5 style, take the rest of the token
        rest = text[idx + len("\\boxed"):].strip()
        return rest.split()[0] if rest else None
    depth, out = 0, []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None


_STRIP = [
    (r"\\left", ""), (r"\\right", ""), (r"\\!", ""), (r"\\,", ""), (r"\\;", ""), (r"\\ ", " "),
    (r"\\dfrac", r"\\frac"), (r"\\tfrac", r"\\frac"), (r"\\cdot", "*"), (r"\\times", "*"),
    (r"\^\{\\circ\}", ""), (r"\^\\circ", ""), (r"\\%", ""), (r"%", ""),
    (r"\\\$", ""), (r"\$", ""), (r"\\text\{([^}]*)\}", r"\1"), (r"\\mbox\{([^}]*)\}", r"\1"),
]


def normalize(ans: str) -> str:
    if ans is None:
        return ""
    s = ans.strip()
    for pat, rep in _STRIP:
        s = re.sub(pat, rep, s)
    s = s.replace(" ", "").replace("\n", "")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac(\d)(\d)", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.rstrip(".")
    s = re.sub(r"^\{(.*)\}$", r"\1", s)          # a stray outer brace
    s = re.sub(r"(\d),(\d\d\d)", r"\1\2", s)     # thousands separators
    return s.lower()


def to_number(s: str):
    """Best-effort numeric value of a normalised answer, or None."""
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    m = re.fullmatch(r"\(?(-?[\d.]+)\)?/\(?(-?[\d.]+)\)?", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def split_answer(text: str) -> tuple[str, str, bool]:
    """Split a completion into (thought, answer, closed).

    Scoring must look only at what the model states after it closes its thought. The
    thought itself routinely contains a \\boxed{} -- the trace's own working -- and
    counting that would credit a run that never actually committed to an answer.
    """
    if "</think>" in text:
        thought, _, answer = text.partition("</think>")
        return thought, answer, True
    return text, "", False


def row_id(row: dict, index: int, split: str) -> str:
    """MATH-500's unique_id, else <split>/<index> (judge_answers.py names rows the same way)."""
    return row.get("unique_id") or f"{split}/{index}"


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    pn, gn = to_number(p), to_number(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    return False


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def dataset_dir_name(dataset: str, config: str) -> str:
    """BeyondAIME, OlymMATH-en-hard: the directory a suite entry's results go under."""
    return dataset.rstrip("/").split("/")[-1] + (f"-{config}" if config else "")


def suite_entries(args, run_name: str) -> list[tuple[str, str, str, Path]]:
    """(dataset, config, split, output directory) for each dataset to evaluate."""
    if not args.suite.strip():
        return [(args.dataset, args.dataset_config, args.split, Path(args.output_dir) / run_name)]
    entries = []
    for item in args.suite.split(";"):
        if not item.strip():
            continue
        parts = [x.strip() for x in item.split("|")] + ["", ""]
        dataset, config, split = parts[0], parts[1], parts[2] or "test"
        entries.append((dataset, config, split, Path(args.output_dir) / dataset_dir_name(dataset, config) / run_name))
    return entries


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    run_name = args.run_name or Path(args.model.rstrip("/")).name
    entries = suite_entries(args, run_name)

    print(f"[eval] model      : {args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only batched generation needs left padding

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = True

    failed = []
    for dataset, config, split, out_dir in entries:
        try:
            evaluate(args, tokenizer, model, dataset, config, split, out_dir)
        except Exception as exc:  # one broken dataset must not cost the others' hours
            if len(entries) == 1:
                raise
            failed.append(dataset)
            print(f"[eval] FAILED {dataset} [{config or '-'}|{split}]: {type(exc).__name__}: {exc}", flush=True)
    if failed:
        raise SystemExit(f"[eval] {len(failed)} of {len(entries)} datasets failed: {', '.join(failed)}")


def evaluate(args, tokenizer, model, dataset: str, config: str, split: str, out_dir: Path) -> None:
    """Generate and string-score one dataset; writes summary.json and predictions.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] dataset    : {dataset}{f' ({config})' if config else ''} [{split}]")
    print(f"[eval] output     : {out_dir}", flush=True)

    data = load_dataset(dataset, config or None, split=split)
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))
    print(f"[eval] problems   : {len(data)}", flush=True)

    enable_thinking = args.enable_thinking == "true"
    template_kwargs = {} if enable_thinking else {"enable_thinking": False}
    print(f"[eval] thinking   : {enable_thinking}", flush=True)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{INSTRUCTION} {row['problem']}"}],
            tokenize=False, add_generation_prompt=True, **template_kwargs,
        )
        for row in data
    ]

    records, n_correct, started = [], 0, time.time()
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start:start + args.batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                # Explicit, so the checkpoint's generation_config defaults never apply.
                **({"top_p": args.top_p, "top_k": args.top_k} if args.temperature > 0 else {}),
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = out[:, enc["input_ids"].shape[1]:]

        for j, seq in enumerate(generated):
            row = data[start + j]
            keep = seq[seq != tokenizer.pad_token_id]
            text = tokenizer.decode(keep, skip_special_tokens=True)

            # A thinking model writes the answer after </think>; the thought itself often
            # contains a \boxed{} too. Score the stated answer, not the scratch work, so a
            # run that never closes its thought is counted as the format failure it is.
            thought, answer_part, closed = split_answer(text)
            n_think = (len(tokenizer(thought, add_special_tokens=False)["input_ids"])
                       if closed else int(keep.numel()))
            pred = extract_boxed(answer_part) if answer_part.strip() else None

            gold = str(row["answer"])
            ok = is_correct(pred, gold)
            n_correct += ok
            records.append({
                "unique_id": row_id(row, start + j, split),
                "subject": row.get("subject", ""), "level": row.get("level", ""),
                "gold": gold, "pred": pred, "correct": bool(ok),
                "completion_tokens": int(keep.numel()),
                "thinking_tokens": n_think,
                "answer_tokens": int(keep.numel()) - n_think,
                "closed_think": closed,
                "completion": text,
            })

        done = start + len(batch_prompts)
        print(f"[eval] {done}/{len(prompts)}  acc={n_correct / done:.3f}  "
              f"{(time.time() - started) / 60:.1f} min", flush=True)

    # -----------------------------------------------------------------------
    # summary
    # -----------------------------------------------------------------------
    toks = sorted(r["completion_tokens"] for r in records)
    by_level, by_subject = {}, {}
    for r in records:
        if r["level"] != "":
            by_level.setdefault(r["level"], []).append(r["correct"])
        if r["subject"]:
            by_subject.setdefault(r["subject"], []).append(r["correct"])

    summary = {
        "model": args.model,
        "dataset": f"{dataset}[{split}]" if not config else f"{dataset}({config})[{split}]",
        "dataset_name": dataset,
        "dataset_config": config,
        "dataset_split": split,
        "n": len(records),
        "accuracy": n_correct / len(records) if records else 0.0,
        "completion_tokens_mean": sum(toks) / len(toks) if toks else 0,
        "completion_tokens_median": toks[len(toks) // 2] if toks else 0,
        "completion_tokens_max": toks[-1] if toks else 0,
        "truncated": sum(r["completion_tokens"] >= args.max_new_tokens for r in records),
        "no_boxed_answer": sum(r["pred"] is None for r in records),
        "never_closed_think": sum(not r["closed_think"] for r in records),
        "thinking_tokens_mean": sum(r["thinking_tokens"] for r in records) / len(records) if records else 0,
        "answer_tokens_mean": sum(r["answer_tokens"] for r in records) / len(records) if records else 0,
        "accuracy_by_level": {k: sum(v) / len(v) for k, v in sorted(by_level.items())},
        "accuracy_by_subject": {k: sum(v) / len(v) for k, v in sorted(by_subject.items())},
        "minutes": (time.time() - started) / 60,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        **({"top_p": args.top_p, "top_k": args.top_k} if args.temperature > 0 else {}),
        "enable_thinking": enable_thinking,
    }

    with (out_dir / "predictions.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[eval] ==== summary ====")
    for k in ("n", "accuracy", "completion_tokens_mean", "completion_tokens_median",
              "thinking_tokens_mean", "answer_tokens_mean",
              "truncated", "no_boxed_answer", "never_closed_think", "minutes"):
        print(f"  {k:26s} {summary[k]}")
    print(f"  accuracy_by_level          {summary['accuracy_by_level']}")
    print(f"[eval] wrote {out_dir}/summary.json and predictions.jsonl")


if __name__ == "__main__":
    main()
