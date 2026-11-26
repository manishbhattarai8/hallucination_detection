import math

import numpy as np

from hallu.detectors.probe import ProbeResult, make_probe, train_select_probe


def _split(N):
    idx = np.arange(N)
    return {"train": idx[: int(N * 0.6)], "val": idx[int(N * 0.6): int(N * 0.8)],
            "test": idx[int(N * 0.8):]}


def test_probe_picks_informative_layer():
    rng = np.random.default_rng(0)
    N, H = 600, 16
    y = rng.integers(0, 2, N)
    acts = rng.normal(size=(N, 4, H)).astype(np.float16)
    acts[:, 2, 0] += 3 * y
    r = train_select_probe(acts, y, _split(N), [0.25, 0.5, 0.75, 1.0])
    assert r.chosen_layer_pos == 2 and r.chosen_frac == 0.75
    assert r.layer_val_auroc[0.75] > 0.9 and r.test_scores.shape == (120,)
    assert r.model is not None


def test_single_class_train_returns_nan_result():
    rng = np.random.default_rng(0)
    N = 100
    y = np.zeros(N, dtype=int)
    acts = rng.normal(size=(N, 2, 4)).astype(np.float16)
    r = train_select_probe(acts, y, _split(N), [0.5, 1.0])
    assert isinstance(r, ProbeResult)
    assert r.model is None and r.test_scores is None
    assert all(math.isnan(v) for v in r.layer_val_auroc.values())
    assert math.isnan(r.chosen_frac) and r.chosen_layer_pos == -1


def test_single_class_val_gives_nan_result():
    rng = np.random.default_rng(1)
    N = 100
    y = rng.integers(0, 2, N)
    y[60:80] = 0
    acts = rng.normal(size=(N, 2, 4)).astype(np.float16)
    r = train_select_probe(acts, y, _split(N), [0.5, 1.0])
    assert r.model is None and r.chosen_layer_pos == -1


def test_ties_pick_deeper_layer():
    N = 100
    y = np.array([0, 1] * 50)
    acts = np.zeros((N, 3, 4), dtype=np.float16)  # constant -> identical AUROC 0.5
    r = train_select_probe(acts, y, _split(N), [0.25, 0.5, 1.0])
    assert r.chosen_layer_pos == 2 and r.chosen_frac == 1.0


def test_make_probe_structure():
    p = make_probe()
    assert [n for n, _ in p.steps] == ["standardscaler", "logisticregression"]
    assert p.steps[1][1].max_iter == 5000


# ---- cross-validated probe (every row scored by a probe that never saw it) ----
from sklearn.metrics import roc_auc_score  # noqa: E402

from hallu.detectors.probe import cv_probe_scores  # noqa: E402


def test_cv_probe_scores_every_row_and_picks_informative_layer():
    rng = np.random.default_rng(0)
    N, H = 500, 16
    y = rng.integers(0, 2, N)
    acts = rng.normal(size=(N, 4, H)).astype(np.float16)
    acts[:, 2, 0] += 3 * y
    folds = np.arange(N) % 5
    r = cv_probe_scores(acts, y, folds, [0.25, 0.5, 0.75, 1.0], seed=0)
    assert r.scores.shape == (N,) and np.isfinite(r.scores).all()
    assert roc_auc_score(y, r.scores) > 0.9
    assert r.chosen_frac == [0.75] * 5 and r.layer_pos == [2] * 5
    assert len(r.models) == 5 and all(m is not None for m in r.models)


def test_cv_probe_has_no_leakage_on_random_data():
    # more features than rows: an in-sample probe would look near-perfect
    rng = np.random.default_rng(1)
    N, H = 200, 400
    y = rng.integers(0, 2, N)
    acts = rng.normal(size=(N, 2, H)).astype(np.float16)
    r = cv_probe_scores(acts, y, np.arange(N) % 5, [0.5, 1.0], seed=0)
    in_sample = make_probe().fit(acts[:, 1, :].astype(np.float32), y)
    assert roc_auc_score(y, in_sample.predict_proba(acts[:, 1, :].astype(np.float32))[:, 1]) > 0.95
    assert abs(roc_auc_score(y, r.scores) - 0.5) < 0.15


def test_cv_probe_degenerate_fold_is_nan():
    N = 100
    y = np.zeros(N, dtype=int)
    y[:5] = 1  # positives only in fold 0 -> folds 1-4 still see them in training,
    folds = np.arange(N) % 5  # but fold 0's training rows hold a single class
    folds[:5] = 0
    acts = np.random.default_rng(2).normal(size=(N, 2, 4)).astype(np.float16)
    r = cv_probe_scores(acts, y, folds, [0.5, 1.0], seed=0)
    assert np.isnan(r.scores[folds == 0]).all()
    assert r.models[0] is None and r.layer_pos[0] == -1 and math.isnan(r.chosen_frac[0])
