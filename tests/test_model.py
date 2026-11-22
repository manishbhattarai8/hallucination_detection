import math

import pytest
import torch

from hallu.model import layer_indices, token_stats, truncate_answer_ids


class _FakeTok:
    eos_token_id = 0
    pad_token_id = 1

    def decode(self, ids):
        return "".join("\n" if i == 5 else "abc"[i % 3] for i in ids)


@pytest.fixture
def fake_tokenizer():
    return _FakeTok()


def test_layer_indices():
    assert layer_indices(24, [0.25, 0.5, 0.75, 1.0]) == [6, 12, 18, 24]
    assert layer_indices(2, [0.25]) == [1]  # never the embedding layer


def test_token_stats_uniform_and_peaked():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]])
    lp, ent, bad = token_stats(logits, torch.tensor([1, 0]))
    assert lp[0] == pytest.approx(math.log(0.25)) and ent[0] == pytest.approx(math.log(4))
    assert lp[1] == pytest.approx(0.0, abs=1e-6) and ent[1] == pytest.approx(0.0, abs=1e-6)
    assert bad is False


def test_token_stats_flags_nonfinite():
    _, _, bad = token_stats(torch.tensor([[float("nan"), 0.0]]), torch.tensor([1]))
    assert bad is True


def test_truncate_answer_ids(fake_tokenizer):
    assert truncate_answer_ids([3, 4, 0, 7], fake_tokenizer) == [3, 4]
    assert truncate_answer_ids([3, 5, 4], fake_tokenizer) == [3]
    assert truncate_answer_ids([0, 3], fake_tokenizer) == []
    assert truncate_answer_ids([3, 1, 4], fake_tokenizer) == [3]
    assert truncate_answer_ids([5, 3], fake_tokenizer) == []
    assert truncate_answer_ids([3, 4], fake_tokenizer) == [3, 4]
