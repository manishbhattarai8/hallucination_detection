import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hallu.config import run_dir
from hallu.evaluate import evaluate_grid
from hallu.io import read_jsonl
from hallu.probe_cv import probe_cv_run
from tests.test_evaluate import N, QIDS, _cfg, _fake_run


def test_probe_cv_scores_every_question(tmp_path, monkeypatch):
    c = _cfg(tmp_path, "fp16")
    y = _fake_run(tmp_path, monkeypatch, c, 1)
    probe_cv_run(c)
    rows = read_jsonl(run_dir(c) / "probe_cv.jsonl")
    assert sorted(r["qid"] for r in rows) == sorted(QIDS)  # orphan acts row excluded
    assert {r["fold"] for r in rows} == set(range(5))
    s = np.array([r["probe_cv"] for r in rows], dtype=float)
    yy = np.array([y[QIDS.index(r["qid"])] for r in rows])
    assert np.isfinite(s).all() and roc_auc_score(yy, s) > 0.8
    meta = json.loads((run_dir(c) / "probe_cv.json").read_text())
    assert meta["k"] == 5 and len(meta["chosen_frac"]) == 5 and meta["n_dropped_nonfinite_acts"] == 0


def test_probe_cv_skips_incomplete_run(tmp_path, monkeypatch):
    c = _cfg(tmp_path, "fp16")
    _fake_run(tmp_path, monkeypatch, c, 1)
    lab = run_dir(c) / "labels.jsonl"
    lab.write_text("".join(lab.read_text().splitlines(keepends=True)[:-1]))
    assert probe_cv_run(c) is None
    assert not (run_dir(c) / "probe_cv.jsonl").exists()


def test_probe_cv_drops_nonfinite_rows(tmp_path, monkeypatch):
    c = _cfg(tmp_path, "fp16")
    _fake_run(tmp_path, monkeypatch, c, 1)
    shard = run_dir(c) / "acts" / "shard_00000.npz"
    with np.load(shard) as z:
        q, a = z["qids"], z["acts"].copy()
    a[3, 2, 0] = np.nan
    np.savez(shard, qids=q, acts=a)
    probe_cv_run(c)
    rows = read_jsonl(run_dir(c) / "probe_cv.jsonl")
    assert len(rows) == N - 1 and QIDS[3] not in {r["qid"] for r in rows}
    assert json.loads((run_dir(c) / "probe_cv.json").read_text())["n_dropped_nonfinite_acts"] == 1


def test_probe_cv_transfer_to_quantized_sibling(tmp_path, monkeypatch):
    c16, c4 = _cfg(tmp_path, "fp16"), _cfg(tmp_path, "nf4")
    _fake_run(tmp_path, monkeypatch, c16, 1)
    y4 = _fake_run(tmp_path, monkeypatch, c4, 2)
    probe_cv_run(c16, siblings=[c4])
    rows = read_jsonl(run_dir(c4) / "probe_transfer_cv.jsonl")
    assert sorted(r["qid"] for r in rows) == sorted(QIDS)
    s = np.array([r["probe_transfer_cv"] for r in rows], dtype=float)
    assert np.isfinite(s).all()
    assert roc_auc_score([y4[QIDS.index(r["qid"])] for r in rows], s) > 0.7


def test_evaluate_reports_probe_cv_on_all_rows(tmp_path, monkeypatch):
    c16, c4 = _cfg(tmp_path, "fp16"), _cfg(tmp_path, "nf4")
    _fake_run(tmp_path, monkeypatch, c16, 1)
    _fake_run(tmp_path, monkeypatch, c4, 2)
    probe_cv_run(c16, siblings=[c4])
    probe_cv_run(c4)
    evaluate_grid([c16, c4], out_dir=tmp_path / "summary")
    m = pd.read_csv(tmp_path / "summary/metrics.csv").set_index(["config_id", "detector", "split"])
    for c in (c16, c4):
        assert m.loc[(c.config_id, "probe_cv", "all"), "n"] == N
        assert (c.config_id, "probe_cv", "all_common") in m.index
    t = pd.read_csv(tmp_path / "summary/transfer.csv")
    assert set(t.method) == {"split", "cv"}
    cv = t[t.method == "cv"].iloc[0]
    assert cv.n == N and np.isfinite(cv.auroc)
