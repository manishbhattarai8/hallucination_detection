import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

import hallu.detect as D
import hallu.evaluate as E
from hallu.config import RunConfig, run_dir
from hallu.data import Question
from hallu.detectors.probe import make_probe
from hallu.evaluate import evaluate_grid, metric_ci
from hallu.io import append_jsonl

N = 60
QIDS = [f"q{i:03d}" for i in range(N)]


def test_metric_ci_perfect_and_random():
    y = np.array([0, 1] * 200)
    s = y.astype(float)
    p, lo, hi = metric_ci(y, s, "auroc")
    assert p == 1.0 and lo == 1.0
    p, lo, hi = metric_ci(y, np.random.default_rng(0).random(400), "auroc")
    assert lo < 0.5 < hi


def test_metric_ci_single_class_is_nan():
    assert all(math.isnan(v) for v in metric_ci(np.array([1, 1, 1, 0]), np.arange(4.0), "auroc"))


def _cfg(tmp_path, precision, size_b=0.5, dataset="triviaqa", excluded=False):
    return RunConfig(config_id=f"m{size_b}_{precision}_{dataset}", model="m", size_b=size_b,
                     family="qwen", precision=precision, dataset=dataset, n_questions=N,
                     subset_seed=0, split_seed=3, sample_seed=0, n_samples=5, temperature=1.0,
                     max_new_tokens=32, batch_size=1, layer_fracs=(0.25, 0.5, 0.75, 1.0),
                     results_root=str(tmp_path), timing_excluded=excluded)


def _fake_run(tmp_path, monkeypatch, cfg, seed, null_row=None, degenerate=False, chosen_pos=1):
    monkeypatch.setattr(D, "load_questions",
                        lambda c, *a, **k: [Question(q, "q?", ("a",)) for q in QIDS])
    rd = run_dir(cfg)
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    y[:4] = [0, 1, 0, 1]
    acts = rng.normal(size=(N, 4, 6)).astype(np.float16)
    acts[:, chosen_pos, 0] += 2 * y
    (rd / "acts").mkdir(exist_ok=True)
    # orphan qid in acts that has no label
    np.savez(rd / "acts" / "shard_00000.npz", qids=np.array(QIDS + ["orphan"]),
             acts=np.concatenate([acts, acts[:1]]))
    append_jsonl(rd / "labels.jsonl", [{"qid": q, "label": int(v)} for q, v in zip(QIDS, y)])
    sp = D.split_index(cfg, QIDS)
    probe = None if degenerate else make_probe().fit(
        acts[:, chosen_pos, :].astype(np.float32), y)
    test_pos = {int(i) for i in sp["test"]}
    rows = []
    for i, q in enumerate(QIDS):
        sc = y[i] + rng.normal(scale=0.8, size=3)
        pr = None
        if probe is not None and i in test_pos:
            pr = float(probe.predict_proba(acts[i:i + 1, chosen_pos, :].astype(np.float32))[0, 1])
        rows.append({"qid": q, "neg_mean_logprob": None if i == null_row else float(sc[0]),
                     "mean_entropy": float(sc[1]), "semantic_entropy": float(sc[2]),
                     "n_clusters": 2, "probe": pr})
    append_jsonl(rd / "scores.jsonl", rows)
    pj = {"layer_val_auroc": {"0.25": 0.5},
          "chosen_frac": None if degenerate else 0.5,
          "chosen_layer_pos": -1 if degenerate else chosen_pos}
    (rd / "probe.json").write_text(json.dumps(pj))
    if probe is not None:
        joblib.dump(probe, rd / "probe.joblib")
    return y


@pytest.fixture
def two_runs(tmp_path, monkeypatch):
    c16, c4 = _cfg(tmp_path, "fp16"), _cfg(tmp_path, "nf4")
    _fake_run(tmp_path, monkeypatch, c16, 1, null_row=5)
    _fake_run(tmp_path, monkeypatch, c4, 2)
    return [c16, c4]


def test_evaluate_grid_on_fake_runs(tmp_path, two_runs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        evaluate_grid(two_runs, out_dir=tmp_path / "summary")
    s = tmp_path / "summary"
    m = pd.read_csv(s / "metrics.csv")
    assert set(m.detector) == {"neg_mean_logprob", "mean_entropy", "semantic_entropy", "probe"}
    assert m.n_dropped_nonfinite.max() == 1
    assert (m[m.detector == "probe"].split == "test").all()
    allrows = m[(m.split == "all") & (m.detector == "mean_entropy")]
    assert (allrows.n == N).all()
    assert m.low_confidence.all()  # < 100 correct
    nm = m[(m.config_id == "m0.5_fp16_triviaqa") & (m.detector == "neg_mean_logprob")
           & (m.split == "all")].iloc[0]
    assert nm.n == N - 1 and nm.n_dropped_nonfinite == 1
    assert len(pd.read_csv(s / "transfer.csv")) == 1
    acc = pd.read_csv(s / "accuracy.csv")
    assert len(acc) == 2 and (acc.n == N).all()
    assert not (s / "cost.csv").exists()
    assert list((s / "figures").glob("heatmap_*.pdf"))
    assert list((s / "figures").glob("auroc_vs_size_*.pdf"))


def test_transfer_skipped_when_fp16_degenerate(tmp_path, monkeypatch):
    c16, c4 = _cfg(tmp_path, "fp16"), _cfg(tmp_path, "int8")
    _fake_run(tmp_path, monkeypatch, c16, 1, degenerate=True)
    _fake_run(tmp_path, monkeypatch, c4, 2)
    evaluate_grid([c16, c4], out_dir=tmp_path / "summary")
    t = pd.read_csv(tmp_path / "summary/transfer.csv")
    assert len(t) == 0


def test_cost_csv_from_fake_timing(tmp_path, two_runs):
    c16, c4 = two_runs
    tdir = tmp_path / "timing"
    for c, gpus in ((c16, 1), (c4, 2)):
        rows = [{"qid": f"q{i}", "t_greedy": 1.0, "t_conf": 0.1 * (i + 1), "t_samples": 2.0,
                 "t_nli": 1.0, "t_probe": 0.01, "peak_vram_gb": 1.0 + i} for i in range(3)]
        rows.append({"summary": True, "gpu_name": "T4", "gpu_count": gpus})
        append_jsonl(tdir / f"{c.config_id}.jsonl", rows)
    evaluate_grid(two_runs, out_dir=tmp_path / "summary")
    c = pd.read_csv(tmp_path / "summary/cost.csv")
    assert set(c.config_id) == {c16.config_id}  # gpu_count 2 excluded
    se = c[c.detector == "semantic_entropy"].iloc[0]
    assert se.n_generations == 6 and se.extra_s_mean == pytest.approx(3.0)
    assert se.total_s_mean == pytest.approx(4.0) and se.extra_s_std == pytest.approx(0.0)
    assert se.peak_vram_gb == 3.0
    nl = c[c.detector == "neg_mean_logprob"].iloc[0]
    assert nl.n_generations == 1
    assert nl.extra_s_mean == pytest.approx(0.2)
    assert nl.extra_s_std == pytest.approx(0.1)
    assert "auroc" in c.columns and c.auroc.notna().all()


def test_cost_excludes_timing_excluded(tmp_path, monkeypatch):
    c = _cfg(tmp_path, "fp16", excluded=True)
    _fake_run(tmp_path, monkeypatch, c, 1)
    append_jsonl(tmp_path / "timing" / f"{c.config_id}.jsonl",
                 [{"qid": "q0", "t_greedy": 1, "t_conf": 0, "t_samples": 0, "t_nli": 0,
                   "t_probe": 0, "peak_vram_gb": 1},
                  {"summary": True, "gpu_name": "T4", "gpu_count": 1}])
    evaluate_grid([c], out_dir=tmp_path / "summary")
    assert not (tmp_path / "summary/cost.csv").exists()
