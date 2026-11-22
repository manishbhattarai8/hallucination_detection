import json
import math
from dataclasses import replace

import numpy as np
import pytest

from hallu.config import load_grid, run_dir
from hallu.generate import _finite_or_none, generate_run, load_acts, sample_seed_for
from hallu.io import read_jsonl


def test_sample_seed_is_stable():
    assert sample_seed_for("q1", 1) == sample_seed_for("q1", 1) != sample_seed_for("q2", 1)
    assert 0 <= sample_seed_for("q1", 10**9) < 2**31


def test_finite_or_none_maps_nonfinite_to_null():
    assert _finite_or_none([1.5, math.nan, math.inf, -math.inf]) == [1.5, None, None, None]


def test_load_acts_dedupes_keeping_last(tmp_path):
    acts_dir = tmp_path / "acts"
    acts_dir.mkdir()
    a1 = np.zeros((2, 4, 3), dtype=np.float16)
    a2 = np.ones((2, 4, 3), dtype=np.float16)
    np.savez(acts_dir / "shard_00000.npz", qids=np.array(["a", "b"]), acts=a1)
    np.savez(acts_dir / "shard_00001.npz", qids=np.array(["b", "c"]), acts=a2)
    qids, acts = load_acts(tmp_path)
    assert sorted(qids) == ["a", "b", "c"] and len(qids) == 3
    assert acts.shape == (3, 4, 3) and acts.dtype == np.float16
    assert acts[qids.index("b")].min() == 1  # from shard 2
    assert acts[qids.index("a")].max() == 0


@pytest.mark.gpu
def test_generate_resume_no_duplicates(tmp_path):
    cfg = replace(load_grid("configs/smoke.yaml")[0], results_root=str(tmp_path))
    generate_run(cfg, limit=3)
    generate_run(cfg, limit=6)
    rows = read_jsonl(run_dir(cfg) / "generations.jsonl")
    assert len(rows) == 6 and len({r["qid"] for r in rows}) == 6
    assert all(len(r["samples"]) == 5 and len(r["token_logprobs"]) >= 1 for r in rows)
    qids, acts = load_acts(run_dir(cfg))
    assert acts.shape[:2] == (6, 4) and acts.dtype == np.float16
    assert "gpu_name" in json.load(open(run_dir(cfg) / "meta.json"))["env"]


@pytest.mark.gpu
def test_generate_nf4_smoke(tmp_path):
    cfg = replace(load_grid("configs/smoke.yaml")[0], results_root=str(tmp_path),
                  precision="nf4", config_id="qwen2.5-0.5b_nf4_triviaqa")
    generate_run(cfg, limit=4)
    rows = read_jsonl(run_dir(cfg) / "generations.jsonl")
    assert len(rows) == 4 and all(len(r["samples"]) == 5 for r in rows)
    assert all(r["token_logprobs"] and None not in r["token_logprobs"] for r in rows)
    _, acts = load_acts(run_dir(cfg))
    assert acts.shape[:2] == (4, 4) and np.isfinite(acts.astype(np.float32)).all()


# --- offline harness: tiny random Llama + stub tokenizer (no download) ---
_V, _PAD, _EOS, _NL = 64, 0, 1, 2


class _StubTok:
    eos_token_id = _EOS
    pad_token_id = _PAD

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, **kw):
        return msgs[-1]["content"]

    def __call__(self, texts, return_tensors, padding=False, add_special_tokens=False):
        from transformers import BatchEncoding

        from hallu.generate import _left_pad
        ids, mask = _left_pad([[3 + ord(c) % (_V - 3) for c in t] for t in texts], _PAD, "cpu")
        return BatchEncoding({"input_ids": ids, "attention_mask": mask})

    def decode(self, ids, skip_special_tokens=False):
        return "".join("\n" if i == _NL else "" if i in (_EOS, _PAD) else chr(65 + i % 26)
                       for i in ids)


def _tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=_V, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, eos_token_id=_EOS,
        pad_token_id=_PAD)).eval()


def test_generate_run_survives_nan_logits(tmp_path, monkeypatch):
    import hallu.generate as G
    from hallu.data import Question

    model = _tiny_llama()

    def nan_hook(mod, inp, out):
        out = out.clone()
        out[..., 5] = float("nan")
        return out

    model.lm_head.register_forward_hook(nan_hook)
    qs = [Question("a", "hi", ("x",)), Question("b", "a longer question", ("y",))]
    monkeypatch.setattr(G, "load_model", lambda *a: (model, _StubTok()))
    monkeypatch.setattr(G, "load_questions", lambda cfg: qs)
    cfg = replace(load_grid("configs/smoke.yaml")[0], results_root=str(tmp_path),
                  max_new_tokens=6, batch_size=2)
    G.generate_run(cfg)
    rows = read_jsonl(run_dir(cfg) / "generations.jsonl")
    assert [r["qid"] for r in rows] == ["a", "b"]
    assert all(r["nonfinite"] and len(r["samples"]) == 5 for r in rows)
    assert G.load_acts(run_dir(cfg))[1].shape == (2, 4, 32)


def test_sample_pass_flags_nonfinite_sampling_logits(monkeypatch):
    import torch

    import hallu.generate as G
    from hallu.data import Question

    model = _tiny_llama()
    cfg = replace(load_grid("configs/smoke.yaml")[0], max_new_tokens=4)
    q = Question("a", "hi", ("x",))
    samples, bad = G._sample_flagged(model, _StubTok(), q, cfg)
    assert len(samples) == 5 and not bad
    model.lm_head.register_forward_hook(lambda m, i, o: torch.full_like(o, float("inf")))
    samples, bad = G._sample_flagged(model, _StubTok(), q, cfg)
    assert len(samples) == 5 and bad


def _harness(tmp_path, monkeypatch, n_q=2, batch_size=2):
    import hallu.generate as G
    from hallu.data import Question

    model = _tiny_llama()
    qs = [Question(f"q{i}", f"question {i}", ("x",)) for i in range(n_q)]
    monkeypatch.setattr(G, "load_model", lambda *a: (model, _StubTok()))
    monkeypatch.setattr(G, "load_questions", lambda cfg: qs)
    cfg = replace(load_grid("configs/smoke.yaml")[0], results_root=str(tmp_path),
                  max_new_tokens=4, batch_size=batch_size)
    return G, cfg


def test_sampling_oom_is_not_swallowed(tmp_path, monkeypatch):
    import torch

    G, cfg = _harness(tmp_path, monkeypatch)

    def oom(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(G, "_sample_flagged", oom)
    with pytest.raises(torch.OutOfMemoryError):
        G.generate_run(cfg)
    assert not (run_dir(cfg) / "generations.jsonl").exists()


def test_sampling_other_runtime_error_reraised(tmp_path, monkeypatch):
    G, cfg = _harness(tmp_path, monkeypatch)
    monkeypatch.setattr(G, "_sample_flagged",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("device-side assert")))
    with pytest.raises(RuntimeError, match="device-side"):
        G.generate_run(cfg)


def test_probability_tensor_error_falls_back_with_sample_failed(tmp_path, monkeypatch):
    G, cfg = _harness(tmp_path, monkeypatch, n_q=3, batch_size=1)
    real = G._sample_flagged

    def flaky(model, tok, q, c):
        if q.qid == "q1":
            raise RuntimeError("probability tensor contains either `inf`, `nan` or element < 0")
        return real(model, tok, q, c)

    monkeypatch.setattr(G, "_sample_flagged", flaky)
    G.generate_run(cfg)
    rows = {r["qid"]: r for r in read_jsonl(run_dir(cfg) / "generations.jsonl")}
    assert len(rows) == 3
    assert rows["q1"]["sample_failed"] is True and rows["q1"]["samples"] == [""] * 5
    assert rows["q1"]["nonfinite"] is False
    assert rows["q0"]["sample_failed"] is False and rows["q2"]["sample_failed"] is False


def test_three_consecutive_sampling_fallbacks_abort(tmp_path, monkeypatch):
    G, cfg = _harness(tmp_path, monkeypatch, n_q=4, batch_size=1)
    monkeypatch.setattr(G, "_sample_flagged", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("probability tensor contains either `inf`, `nan` or element < 0")))
    with pytest.raises(RuntimeError, match="consecutive"):
        G.generate_run(cfg)
    assert len(read_jsonl(run_dir(cfg) / "generations.jsonl")) == 2


_LLAMA_LIKE = ('{%- if not date_string is defined %}{%- set date_string = strftime_now("%d %b %Y")'
               ' %}{%- endif %}Today Date: {{ date_string }}\n'
               '{% for m in messages %}{{ m["content"] }}\n{% endfor %}')


def _cached_qwen_tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")


def test_prompt_date_is_pinned(monkeypatch):
    import datetime as dt

    import transformers.utils.chat_template_utils as ctu

    import hallu.generate as G
    from hallu.data import Question

    tok = _cached_qwen_tok()
    tok.chat_template = _LLAMA_LIKE
    q = Question("a", "Who?", ("x",))
    p1 = G._prompt(tok, q)
    assert G.PROMPT_DATE == "26 Jul 2024" and "Today Date: 26 Jul 2024" in p1

    class _Later(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2031, 3, 9)

    monkeypatch.setattr(ctu, "datetime", _Later)
    assert G._prompt(tok, q) == p1


def test_qwen_prompt_unchanged_by_date_string():
    import hallu.generate as G
    from hallu.data import Question, build_messages

    tok = _cached_qwen_tok()
    q = Question("a", "Who wrote Hamlet?", ("x",))
    before = tok.apply_chat_template(build_messages(q.question), add_generation_prompt=True,
                                     tokenize=False)
    assert G._prompt(tok, q) == before
