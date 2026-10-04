"""Validate the exact sampled-token payload used in the policy loss."""

import math


def validate_model_output(output, prompt_ids):
    if output.prompt_ids != prompt_ids:
        raise ValueError("rollout prompt IDs differ from the rendered SFT prompt")
    if output.completion_ids is None or len(output.completion_ids) == 0:
        raise ValueError("rollout produced no response tokens")
    if output.logprobs is None or len(output.logprobs) != len(output.completion_ids):
        raise ValueError("each sampled response token must have a rollout log probability")
    if any(not math.isfinite(float(value)) for value in output.logprobs):
        raise ValueError("rollout contains non-finite token log probabilities")
