#!/usr/bin/env python3
"""Score an eval run's answers with Hugging Face Math-Verify, next to the string scorer.

eval_math500.py's string scorer (LaTeX normalisation plus a numeric fallback) stays the
headline number, unchanged, so results compare with ../../train/evaluate. This adds a
second deterministic score: math_verify decides mathematical equivalence, so it accepts
0.5 = \\frac{1}{2}, (x+1)^2 = x^2+2x+1, intervals, sets and dropped units that exact
string matching rejects.

The check is the one the GRPO reward uses (GRPO/src/labelcot/reward.py,
_math_verify_match): the reference answer is parsed as \\boxed{gold}; the prediction is
everything the model wrote after </think> (its last 4,000 characters), from which
math_verify takes the last \\boxed{} or else the final stated answer. A run that never
closed its thought has no answer and is wrong. Unlike the reward, this is math_verify
alone, not math_verify or the string scorer, so the two scores can be told apart.

Adds a `math_verify` field to every row of predictions.jsonl (nothing else in it
changes) and to summary.json:

    accuracy_math_verify           share of rows math_verify accepts
    math_verify_flipped_to_correct rows the string scorer rejected and math_verify accepts
    math_verify_flipped_to_wrong   rows the string scorer accepted and math_verify rejects

Needs math-verify 0.9.0 (Python >= 3.10); ./run_eval.sh score runs it in the GRPO image,
which has it.

    python3 math_verify_answers.py --run-dir results/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from math_verify import parse, verify

TIMEOUT_S = 5            # parse and verify each; signal.alarm, so main thread only
MAX_ANSWER_CHARS = 4000  # as in the reward
MAX_PARSED_CHARS = 400   # a longer parsed expression is not an answer, and can stall sympy


def answer_part(completion: str) -> str | None:
    """What the model wrote after </think>, or None when it never closed its thought."""
    if "</think>" not in completion:
        return None
    return completion.partition("</think>")[2]


def math_verify_correct(completion: str, gold: str) -> bool:
    answer = answer_part(completion)
    if not answer or not answer.strip():
        return False
    try:
        gold_parsed = parse("\\boxed{" + gold + "}", parsing_timeout=TIMEOUT_S)
        pred_parsed = parse(answer[-MAX_ANSWER_CHARS:], parsing_timeout=TIMEOUT_S)
        if not gold_parsed or not pred_parsed:
            return False
        if any(len(str(x)) > MAX_PARSED_CHARS for x in pred_parsed):
            return False
        return bool(verify(gold_parsed, pred_parsed, timeout_seconds=TIMEOUT_S))
    except Exception:  # timeouts and sympy errors on odd input count as "not matched"
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="eval output directory holding predictions.jsonl")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.is_file():
        sys.exit(f"no predictions.jsonl under {run_dir}")
    records = [json.loads(line) for line in pred_path.open()]

    for i, r in enumerate(records, 1):
        r["math_verify"] = math_verify_correct(r.get("completion", ""), r["gold"])
        if i % 100 == 0:
            print(f"[math_verify] {i}/{len(records)}", flush=True)

    tmp = pred_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(pred_path)

    to_correct = [r for r in records if r["math_verify"] and not r["correct"]]
    to_wrong = [r for r in records if r["correct"] and not r["math_verify"]]
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    summary.update({
        "accuracy_string": sum(r["correct"] for r in records) / len(records),
        "accuracy_math_verify": sum(r["math_verify"] for r in records) / len(records),
        "math_verify_flipped_to_correct": len(to_correct),
        "math_verify_flipped_to_wrong": len(to_wrong),
    })
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n[math_verify] ==== summary ====")
    print(f"  accuracy (string scorer) {summary['accuracy_string']:.3f}")
    print(f"  accuracy (math_verify)   {summary['accuracy_math_verify']:.3f}")
    print(f"  string wrong -> right    {len(to_correct)}")
    print(f"  string right -> wrong    {len(to_wrong)}")
    for r in (to_correct + to_wrong)[:10]:
        print(f"    {r['unique_id']}: gold={r['gold']!r} pred={r['pred']!r} string={r['correct']} math_verify={r['math_verify']}")
    print(f"[math_verify] updated {pred_path} and {summary_path}")


if __name__ == "__main__":
    main()
