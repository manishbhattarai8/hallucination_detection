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


@pytest.mark.parametrize("precision", ["fp16", "int8", "nf4"])
def test_load_model_passes_fp16_dtype(monkeypatch, precision):
    import transformers

    import hallu.model as M

    seen = {}

    class _Tok:
        pad_token = "x"

    class _Model:
        def eval(self):
            return self

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: _Tok())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **k: seen.update(k) or _Model())
    M.load_model("some/model", precision)
    assert seen["dtype"] is torch.float16
    assert seen["device_map"] == "auto"
    assert ("quantization_config" in seen) == (precision != "fp16")


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["int8", "nf4"])
def test_quantized_model_runs_in_fp16(precision):
    from hallu.model import load_model

    model, tok = load_model("Qwen/Qwen2.5-0.5B-Instruct", precision)
    assert model.dtype == torch.float16
    assert model.lm_head.weight.dtype == torch.float16
    enc = tok(["Hello there"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    assert out.logits.dtype == torch.float16
    assert out.hidden_states[-1].dtype == torch.float16


def test_truncate_answer_ids_extra_stop_ids(fake_tokenizer):
    assert truncate_answer_ids([3, 4, 9, 7], fake_tokenizer) == [3, 4, 9, 7]
    assert truncate_answer_ids([3, 4, 9, 7], fake_tokenizer, stop_ids=[9]) == [3, 4]
    assert truncate_answer_ids([3, 0, 7], fake_tokenizer, stop_ids=[9]) == [3]  # eos still stops
