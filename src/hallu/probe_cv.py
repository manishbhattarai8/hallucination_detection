"""Cross-validated probe: every question gets a probe score from a probe that never saw it.

Runs on CPU from saved activations (after detect). Writes, per run dir:
  probe_cv.jsonl           {"qid", "probe_cv", "fold"}
  probe_cv.json            per-fold chosen layer + bookkeeping
and, for an fp16 run given its int8/nf4 siblings, in each sibling's run dir:
  probe_transfer_cv.jsonl  {"qid", "probe_transfer_cv", "fold"}: the fp16 probe of that
                           question's fold, applied to the sibling's activations.
Folds come from the full question subset (data.fold_ids with split_seed), so they are
identical across configs. The single-split probe (probe.json / scores.jsonl) is untouched.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

import hallu.detect as D
from hallu.config import RunConfig, run_dir
from hallu.data import fold_ids
from hallu.detectors.probe import cv_probe_scores
from hallu.generate import load_acts
from hallu.io import env_info, read_jsonl

K_FOLDS = 5


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    tmp.replace(path)


def _complete(cfg: RunConfig):
    """(labels, acts qids, acts, full qid list) for a complete run, else None."""
    rd = run_dir(cfg)
    if not (rd / "labels.jsonl").exists():
        return None
    labels = {r["qid"]: int(r["label"]) for r in read_jsonl(rd / "labels.jsonl")}
    full = [q.qid for q in D.load_questions(cfg)]
    try:
        a_qids, acts = load_acts(rd)
    except FileNotFoundError:
        return None
    have = set(a_qids)
    if any(q not in labels or q not in have for q in full):
        return None
    return labels, a_qids, acts, full


def probe_cv_run(cfg: RunConfig, siblings: Sequence[RunConfig] = ()) -> Path | None:
    got = _complete(cfg)
    if got is None:
        print(f"[{cfg.config_id}] probe_cv skipped: run incomplete", flush=True)
        return None
    labels, a_qids, acts, full = got
    fullset = set(full)
    keep = [i for i, q in enumerate(a_qids) if q in fullset]
    ok = D.finite_act_rows(acts[keep])
    n_dropped = int((~ok).sum())
    keep = [i for i, good in zip(keep, ok) if good]
    kq = [a_qids[i] for i in keep]
    fold_of = fold_ids(full, K_FOLDS, cfg.split_seed)
    folds = np.array([fold_of[q] for q in kq])
    y = np.array([labels[q] for q in kq], dtype=int)
    res = cv_probe_scores(acts[keep], y, folds, list(cfg.layer_fracs), seed=cfg.split_seed)

    rd = run_dir(cfg)
    _write_jsonl(rd / "probe_cv.jsonl", [
        {"qid": q, "probe_cv": float(s) if math.isfinite(s) else None, "fold": int(f)}
        for q, s, f in zip(kq, res.scores, folds)])
    meta = {"k": K_FOLDS, "seed": cfg.split_seed,
            "chosen_frac": [None if math.isnan(c) else c for c in res.chosen_frac],
            "layer_pos": res.layer_pos, "n_scored": len(kq), "n_total": len(full),
            "n_dropped_nonfinite_acts": n_dropped, "git_sha": env_info()["git_sha"]}
    (rd / "probe_cv.json").write_text(json.dumps(meta, allow_nan=False), encoding="utf-8")

    if cfg.precision == "fp16":
        for sib in siblings:
            _transfer(sib, res, fold_of)
    return rd


def _transfer(sib: RunConfig, res, fold_of: dict[str, int]) -> None:
    got = _complete(sib)
    if got is None:
        print(f"[{sib.config_id}] probe_cv transfer skipped: run incomplete", flush=True)
        return
    _, s_qids, s_acts, full = got
    pos_of = {q: i for i, q in enumerate(s_qids)}
    rows = []
    for q in full:
        f = fold_of[q]
        p, m = res.layer_pos[f], res.models[f]
        if m is None:
            continue
        x = np.asarray(s_acts[pos_of[q], p, :], dtype=np.float32)[None, :]
        if not np.isfinite(x).all():  # same rule as the split transfer: chosen layer only
            continue
        rows.append({"qid": q, "probe_transfer_cv": float(m.predict_proba(x)[0, 1]), "fold": f})
    _write_jsonl(run_dir(sib) / "probe_transfer_cv.jsonl", rows)
