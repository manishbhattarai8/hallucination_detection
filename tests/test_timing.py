import math
import time
from dataclasses import replace

import joblib
import numpy as np
import pytest
import torch

import hallu.timing as T
from hallu.config import RunConfig, load_grid, run_dir
from hallu.data import Question
from hallu.generate import GreedyResult
from hallu.io import read_jsonl
from hallu.timing import cuda_timer, timing_run


def _cfg(tmp_path, **kw):
    base = dict(config_id="fake", model="m", size_b=0.5, family="qwen", precision="fp16",
                dataset="triviaqa", n_questions=50, subset_seed=0, split_seed=0,
                sample_seed=0, n_samples=5, temperature=1.0, max_new_tokens=32,
                batch_size=1, layer_fracs=(0.25, 0.5, 0.75, 1.0),
                results_root=str(tmp_path), timing_excluded=False)
    base.update(kw)
    return RunConfig(**base)


def test_cuda_timer_measures_cpu_time():
    with cuda_timer() as t:
        time.sleep(0.05)
    assert 0.04 < t.seconds < 0.5


def test_timing_refuses_excluded_config():
    cfg = next(c for c in load_grid("configs/qwen.yaml") if c.timing_excluded)
    with pytest.raises(RuntimeError):
        timing_run(cfg)


def test_timing_refuses_wrong_dataset_and_device_count(tmp_path, monkeypatch):
    monkeypatch.setattr(T.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError):
        timing_run(_cfg(tmp_path, dataset="nq_open"))
    monkeypatch.setattr(T.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError):
        timing_run(_cfg(tmp_path))


class _Probe:
    def predict_proba(self, x):
        assert x.shape == (1, 8) and x.dtype == np.float32
        return np.array([[0.4, 0.6]])


def _fake_env(tmp_path, monkeypatch, n_q=10, probe=True):
    cfg = _cfg(tmp_path)
    rd = run_dir(cfg)
    if probe:
        joblib.dump(_Probe(), rd / "probe.joblib")
        (rd / "probe.json").write_text('{"chosen_layer_pos": 2}')
    else:
        (rd / "probe.json").write_text('{"chosen_layer_pos": -1}')
    qs = [Question(f"q{i}", f"question {i}?", ("a",)) for i in range(n_q)]
    monkeypatch.setattr(T, "load_questions", lambda c, *a, **k: qs)
    monkeypatch.setattr(T.torch.cuda, "device_count", lambda: 1)
    loaded = []
    monkeypatch.setattr(T, "load_model", lambda m, p: loaded.append(m) or ("model", "tok"))
    calls = []

    def gen(model, tok, questions, c):
        assert len(questions) == 1
        calls.append(questions[0].qid)
        time.sleep(0.002)
        return "generated"

    def score(model, tok, questions, c, generated):
        assert generated == "generated" and len(questions) == 1
        time.sleep(0.001)
        return [GreedyResult(questions[0].qid, "a", False, False, [-0.5, -1.0], [1.0, 2.0],
                             np.zeros((4, 8), dtype=np.float16))]

    monkeypatch.setattr(T, "greedy_generate", gen)
    monkeypatch.setattr(T, "greedy_score", score)
    monkeypatch.setattr(T, "sample_pass", lambda m, t, q, c: ["a"] * 5)

    class Judge:
        def semantic_entropy(self, q, answers):
            assert len(answers) == 6
            return 0.0, 1

    monkeypatch.setattr(T, "NLIJudge", Judge)
    return cfg, calls, loaded


def test_timing_run_fake_pipeline(tmp_path, monkeypatch):
    cfg, calls, loaded = _fake_env(tmp_path, monkeypatch)
    out = timing_run(cfg, n_warmup=2, n_timed=3)
    assert out == tmp_path / "timing" / "fake.jsonl"
    assert calls == ["q3", "q4", "q0", "q1", "q2"]  # warm-up first, positions n_timed..
    assert loaded == ["m"]
    rows = read_jsonl(out)
    assert [r.get("qid") for r in rows[:3]] == ["q0", "q1", "q2"]
    for r in rows[:3]:
        for k in ("t_greedy", "t_generate", "t_score", "t_conf", "t_samples", "t_nli",
                  "t_probe", "peak_vram_gb"):
            assert math.isfinite(r[k]) and r[k] >= 0
        assert r["t_greedy"] == pytest.approx(r["t_generate"] + r["t_score"])
        assert r["t_generate"] > 0.0015 and r["t_score"] > 0.0005
    assert rows[3]["summary"] is True and "gpu_name" in rows[3] and "gpu_count" in rows[3]
    assert len(rows) == 4
    # rerun starts fresh
    timing_run(cfg, n_warmup=2, n_timed=3)
    assert len(read_jsonl(out)) == 4


def test_timing_short_subset_raises(tmp_path, monkeypatch):
    cfg, _, loaded = _fake_env(tmp_path, monkeypatch, n_q=4)
    with pytest.raises(ValueError):
        timing_run(cfg, n_warmup=2, n_timed=3)
    assert loaded == []


def test_timing_missing_or_degenerate_probe_raises(tmp_path, monkeypatch):
    cfg, _, loaded = _fake_env(tmp_path, monkeypatch, probe=False)
    with pytest.raises(RuntimeError):
        timing_run(cfg, n_warmup=1, n_timed=2)
    assert loaded == []


@pytest.mark.gpu
def test_timing_smoke(tmp_path):
    from hallu.detect import detect_run
    from hallu.generate import generate_run
    from hallu.grade import grade_run

    cfg = replace(load_grid("configs/smoke.yaml")[0], results_root=str(tmp_path))
    generate_run(cfg)
    grade_run(cfg)
    detect_run(cfg)
    # N=20 makes the real probe degenerate (single-class val split), and timing_run
    # rightly refuses that, so install a usable synthetic probe on the last saved layer.
    import json

    from hallu.detectors.probe import make_probe
    from hallu.generate import load_acts

    rd = run_dir(cfg)
    _, acts = load_acts(rd)
    probe = make_probe().fit(acts[:, -1, :].astype(np.float32), np.arange(len(acts)) % 2)
    joblib.dump(probe, rd / "probe.joblib")
    (rd / "probe.json").write_text(json.dumps({
        "layer_val_auroc": {}, "chosen_frac": 1.0, "chosen_layer_pos": acts.shape[1] - 1}))
    out = timing_run(cfg, n_warmup=1, n_timed=3)
    rows = read_jsonl(out)
    assert len(rows) == 4 and rows[-1]["summary"] is True
    assert all(math.isfinite(r["t_nli"]) and math.isfinite(r["t_probe"]) for r in rows[:3])
