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


@pytest.mark.gpu
def test_nli_judge_fp16_tiny_model(tmp_path):
    import torch
    from transformers import AutoTokenizer, DebertaConfig, DebertaForSequenceClassification

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    cfg = DebertaConfig(
        vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, relative_attention=True, pos_att_type=["c2p", "p2c"],
        max_position_embeddings=512, max_relative_positions=-1, position_biased_input=False,
        num_labels=3, id2label={0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"},
        label2id={"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2},
    )
    DebertaForSequenceClassification(cfg).save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    j = NLIJudge(model_name=str(tmp_path), device="cuda")
    assert all(p.dtype == torch.float16 for p in j.model.parameters())
    assert isinstance(j.entails("Q?", "a", "b"), bool)
    h, n = j.semantic_entropy("Q?", ["x", "y", "z"])
    assert 1 <= n <= 3
