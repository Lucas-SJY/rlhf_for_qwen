#!/usr/bin/env python3
"""CPU diagnostic of reward -> grouped advantages -> a tiny policy update.

Scripted responses, not Qwen inference. This detects zero-signal mistakes without
pretending to exercise vLLM/FSDP; use the GPU smoke test for those integrations.
"""

from __future__ import annotations

import argparse
import json
import statistics

from labelcot.reward import score_completion


def grouped_advantages(rewards):
    if len(rewards) < 2:
        raise ValueError("at least two responses are required")
    mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
    return [(reward - mean) / (std + 1e-6) for reward in rewards]


def diagnostic(require_torch=False):
    task = {"answer": "9", "ref_labels": ["logical_deduction", "concluding"]}
    good = "<think>\n[logical_deduction] 4 + 5 = 9.\n\n[concluding] The result is 9.\n</think>\n\\boxed{9}"
    wrong = good.replace("\\boxed{9}", "\\boxed{8}")
    cases = [(good, False), (wrong, False), (good, True), (good, False)]
    rewards = [score_completion(text, task, truncated).reward for text, truncated in cases]
    assert rewards == [1.0, 0.5, 0.0, 1.0], rewards
    advantages = grouped_advantages(rewards)
    assert advantages[0] > 0 and advantages[1] < 0 and advantages[2] < advantages[1]
    assert grouped_advantages([0.0] * 4) == [0.0] * 4
    report = {"kind": "scripted CPU diagnostic (not a GPU/LLM integration test)",
              "rewards": rewards, "advantages": advantages, "uniform_group_has_zero_signal": True}
    try:
        import torch
    except ImportError:
        if require_torch:
            raise RuntimeError("install torch to run --require-torch")
        report["policy_update"] = "skipped: torch not installed"
        return report
    # Four actions stand in for four entire completions. An initial loss of zero can
    # still have a nonzero gradient; test actual probability changes, not loss alone.
    logits = torch.zeros(4, requires_grad=True)
    old_logp = torch.log_softmax(logits.detach(), dim=0)
    ref_logp = old_logp.clone()
    optimizer = torch.optim.SGD([logits], lr=0.1)
    adv = torch.tensor(advantages)
    logp = torch.log_softmax(logits, dim=0)
    ratio = torch.exp(logp - old_logp)
    pg = torch.maximum(-adv * ratio, -adv * torch.clamp(ratio, 0.8, 1.28)).mean()
    delta = ref_logp - logp
    loss = pg + 0.001 * (delta.exp() - delta - 1).mean()
    loss.backward()
    grad_norm = float(logits.grad.norm())
    assert grad_norm > 0 and torch.isfinite(logits.grad).all()
    optimizer.step()
    probs = logits.detach().softmax(0).tolist()
    assert probs[0] > 0.25 and probs[2] < 0.25
    report.update(policy_update="passed", gradient_norm=grad_norm, probabilities_after_update=probs)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(diagnostic(args.require_torch), indent=2))
