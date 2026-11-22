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
