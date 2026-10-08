#!/usr/bin/env python3
"""Compute the target label mix used by the reward (labelcot.reward.LABEL_PRIOR).

For every annotated trace, take the share of each of the eight labels among its spans,
then average those shares over all traces. Averaging per trace (rather than pooling all
spans) describes a typical trace, which is what one completion is compared against;
pooling would let the longest traces dominate. Stdlib only.

Usage (from the repository root):
    python3 GRPO/src/compute_label_prior.py
    python3 GRPO/src/compute_label_prior.py --input-dir ../train/bespoke-v2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from labelcot.reward import LABELS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="../train/bespoke-v2", help="directory holding sample_*.json")
    args = p.parse_args()

    totals, n_traces, n_spans = Counter(), 0, 0
    for path in sorted(Path(args.input_dir).glob("sample_*.json")):
        spans = json.loads(path.read_text()).get("spans") or []
        labels = [s.get("label") for s in spans if s.get("label") in LABELS and (s.get("text") or "").strip()]
        if not labels:
            continue
        n_traces += 1
        n_spans += len(labels)
        counts = Counter(labels)
        for label in LABELS:
            totals[label] += counts[label] / len(labels)
    if not n_traces:
        raise SystemExit(f"no annotated sample_*.json under {args.input_dir}")

    print(f"# {n_traces} traces, {n_spans} spans from {args.input_dir}")
    print("_RAW_PRIOR = {")
    for label in sorted(LABELS, key=lambda label: -totals[label]):
        print(f'    "{label}": {totals[label] / n_traces:.4f},')
    print("}")


if __name__ == "__main__":
    main()
