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


@dataclass
class CVProbeResult:
    scores: np.ndarray                 # out-of-fold P(hallucinated) per row; NaN if its fold is degenerate
    chosen_frac: list[float]           # per fold
    layer_pos: list[int]               # per fold (-1 if degenerate)
    models: list[Pipeline | None]      # per fold, fit on all of that fold's training rows


def cv_probe_scores(acts: np.ndarray, labels: np.ndarray, folds: np.ndarray,
                    layer_fracs: Sequence[float], seed: int = 0,
                    inner_val_frac: float = 0.25) -> CVProbeResult:
    """K-fold probe: for each fold, pick the layer on a random inner holdout of the other
    folds (train_select_probe), refit at that layer on all of them, score the held-out fold."""
    labels, folds = np.asarray(labels), np.asarray(folds)
    scores = np.full(len(labels), np.nan)
    chosen, pos_l, models = [], [], []
    for f in sorted(np.unique(folds)):
        held = np.flatnonzero(folds == f)
        train = np.flatnonzero(folds != f)
        perm = np.random.default_rng(seed + int(f)).permutation(train)
        n_val = int(round(len(perm) * inner_val_frac))
        split = {"train": perm[n_val:], "val": perm[:n_val], "test": held}
        sel = train_select_probe(acts, labels, split, layer_fracs)
        if sel.chosen_layer_pos < 0:
            chosen.append(math.nan)
            pos_l.append(-1)
            models.append(None)
            continue
        p = sel.chosen_layer_pos
        x = np.asarray(acts[:, p, :], dtype=np.float32)
        m = make_probe().fit(x[train], labels[train])
        scores[held] = m.predict_proba(x[held])[:, 1]
        chosen.append(sel.chosen_frac)
        pos_l.append(p)
        models.append(m)
    return CVProbeResult(scores, chosen, pos_l, models)
