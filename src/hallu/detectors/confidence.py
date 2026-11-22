"""Confidence detectors: higher score = more likely hallucinated."""
from __future__ import annotations

import math
from collections.abc import Sequence


def neg_mean_logprob(logprobs: Sequence[float]) -> float:
    if len(logprobs) == 0:
        return float("nan")
    return -sum(logprobs) / len(logprobs)


def mean_entropy(entropies: Sequence[float]) -> float:
    if len(entropies) == 0:
        return float("nan")
    return sum(entropies) / len(entropies)
