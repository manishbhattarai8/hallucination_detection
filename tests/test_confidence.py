import math

from hallu.detectors.confidence import mean_entropy, neg_mean_logprob


def test_confidence_scores():
    assert neg_mean_logprob([-1.0, -3.0]) == 2.0
    assert mean_entropy([0.5, 1.5]) == 1.0


def test_empty_is_nan():
    assert math.isnan(neg_mean_logprob([]))
    assert math.isnan(mean_entropy([]))
