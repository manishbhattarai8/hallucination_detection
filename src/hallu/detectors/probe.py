"""Internal-state probe: per-layer logistic regression on saved hidden states."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_probe() -> Pipeline:
    return Pipeline([
        ("standardscaler", StandardScaler()),
        # l2 is the default penalty; passing penalty= warns in sklearn>=1.8
        ("logisticregression", LogisticRegression(C=1.0, max_iter=5000)),
    ])


@dataclass
class ProbeResult:
    layer_val_auroc: dict[float, float]
    chosen_frac: float
    chosen_layer_pos: int
    model: Pipeline | None
    test_scores: np.ndarray | None


def train_select_probe(acts: np.ndarray, labels: np.ndarray,
                       split_idx: dict[str, np.ndarray],
                       layer_fracs: Sequence[float]) -> ProbeResult:
    """Fit one probe per layer on train, choose the layer with best val AUROC.

    Ties go to the deeper layer (higher position); NaN AUROCs are ignored.
    Degenerate case: if train or val has a single class, that layer's val AUROC
    is NaN. If every layer is NaN (or train is single-class) the result has
    model=None, test_scores=None, chosen_frac=NaN and chosen_layer_pos=-1.
    Probe score is predict_proba[:, 1] (higher = more hallucinated).
    """
    labels = np.asarray(labels)
    tr, va, te = (np.asarray(split_idx[k]) for k in ("train", "val", "test"))
    y_tr, y_va = labels[tr], labels[va]
    aurocs = {f: math.nan for f in layer_fracs}
    models: dict[int, Pipeline] = {}
    if len(np.unique(y_tr)) >= 2 and len(np.unique(y_va)) >= 2:
        for pos, frac in enumerate(layer_fracs):
            x = np.asarray(acts[:, pos, :], dtype=np.float32)
            m = make_probe().fit(x[tr], y_tr)
            aurocs[frac] = float(roc_auc_score(y_va, m.predict_proba(x[va])[:, 1]))
            models[pos] = m
    best = -1
    for pos, frac in enumerate(layer_fracs):
        v = aurocs[frac]
        if math.isnan(v):
            continue
        if best < 0 or v >= aurocs[layer_fracs[best]]:
            best = pos
    if best < 0:
        return ProbeResult(aurocs, math.nan, -1, None, None)
    x = np.asarray(acts[:, best, :], dtype=np.float32)
    scores = models[best].predict_proba(x[te])[:, 1]
    return ProbeResult(aurocs, layer_fracs[best], best, models[best], scores)
