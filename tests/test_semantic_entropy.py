import math

import pytest

from hallu.detectors.semantic_entropy import (
    NLIJudge,
    cluster_by_entailment,
    discrete_semantic_entropy,
)


def test_semantic_entropy_values():
    assert discrete_semantic_entropy([0] * 6) == 0.0
    assert discrete_semantic_entropy([0, 1, 2, 3, 4, 5]) == pytest.approx(math.log(6))
    assert discrete_semantic_entropy([0, 0, 0, 1, 1, 1]) == pytest.approx(math.log(2))


def test_clustering_requires_bidirectional_entailment():
    one_way = lambda a, b: (a, b) == ("Paris, France", "Paris")
    assert cluster_by_entailment(["Paris", "Paris, France"], one_way) == [0, 1]
    both = lambda a, b: {a, b} == {"Paris", "Paris, France"}
    assert cluster_by_entailment(["Paris", "Paris, France", "Rome"], both) == [0, 0, 1]


def test_normalized_duplicates_skip_nli():
    never = lambda a, b: pytest.fail("NLI called")
    assert cluster_by_entailment(["The Beatles", "beatles!"], never) == [0, 0]


def test_empty_answers_cluster_together():
    never = lambda a, b: pytest.fail("NLI called")
    assert cluster_by_entailment(["", " ", "."], never) == [0, 0, 0]


@pytest.mark.gpu
def test_nli_judge_smoke():
    j = NLIJudge()
    assert j.semantic_entropy("Capital of France?", ["Paris", "Paris", "Rome"])[1] == 2
