"""Stage 3: per-question detector scores (confidence, semantic entropy, probe)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import joblib
import numpy as np

from hallu.config import RunConfig, run_dir
from hallu.data import load_questions, split_ids
from hallu.detectors.confidence import mean_entropy, neg_mean_logprob
from hallu.detectors.probe import train_select_probe
from hallu.detectors.semantic_entropy import NLIJudge
from hallu.generate import load_acts
from hallu.io import append_jsonl, done_ids, read_jsonl


def _clean(x):
    """Non-finite floats -> None so JSON never contains bare NaN tokens."""
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def _conf(fn, xs, nonfinite: bool):
    if nonfinite or any(v is None for v in xs):
        return None
    return _clean(fn(xs))


def split_index(cfg: RunConfig, qids: list[str]) -> dict[str, np.ndarray]:
    """Positions in `qids` of each split; splits come from the full subset so they match across configs."""
    full = [q.qid for q in load_questions(cfg)]
    sp = {k: set(v) for k, v in split_ids(full, cfg.split_seed).items()}
    return {k: np.array([i for i, q in enumerate(qids) if q in s], dtype=int)
            for k, s in sp.items()}


def _dump(obj) -> str:
    return json.dumps(obj, allow_nan=False)


def _probe_step(cfg: RunConfig, rd: Path, scores_path: Path) -> None:
    probe_json = rd / "probe.json"
    if probe_json.exists():
        return
    labels = {r["qid"]: int(r["label"]) for r in read_jsonl(rd / "labels.jsonl")}
    a_qids, acts = load_acts(rd)
    keep = [i for i, q in enumerate(a_qids) if q in labels]
    kq = [a_qids[i] for i in keep]
    y = np.array([labels[q] for q in kq], dtype=int)
    x = acts[keep] if keep else acts[:0]
    split = split_index(cfg, kq)
    res = train_select_probe(x, y, split, list(cfg.layer_fracs))

    probe_by_q: dict[str, float] = {}
    if res.model is not None:
        joblib.dump(res.model, rd / "probe.joblib")
        probe_by_q = {kq[i]: float(s) for i, s in zip(split["test"], res.test_scores)}

    rows = read_jsonl(scores_path)
    for r in rows:
        r["probe"] = probe_by_q.get(r["qid"])
    tmp = scores_path.with_name(scores_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(_dump(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, scores_path)

    # written last: its existence marks the probe step as complete
    meta = {
        "layer_val_auroc": {str(k): _clean(v) for k, v in res.layer_val_auroc.items()},
        "chosen_frac": _clean(res.chosen_frac),
        "chosen_layer_pos": int(res.chosen_layer_pos),
    }
    probe_json.write_text(_dump(meta), encoding="utf-8")


def detect_run(cfg: RunConfig, nli: NLIJudge | None = None) -> Path:
    rd = run_dir(cfg)
    scores_path = rd / "scores.jsonl"
    qtext = {q.qid: q.question for q in load_questions(cfg)}
    gens = read_jsonl(rd / "generations.jsonl")
    done = done_ids(scores_path)
    todo = [g for g in gens if g["qid"] not in done]
    if todo and nli is None:
        nli = NLIJudge()
    for g in todo:
        nf = bool(g.get("nonfinite"))
        ent, ncl = nli.semantic_entropy(qtext[g["qid"]], [g["answer"]] + list(g["samples"]))
        append_jsonl(scores_path, [{
            "qid": g["qid"],
            "neg_mean_logprob": _conf(neg_mean_logprob, g["token_logprobs"], nf),
            "mean_entropy": _conf(mean_entropy, g["token_entropies"], nf),
            "semantic_entropy": _clean(ent),
            "n_clusters": int(ncl),
            "probe": None,
        }])
    if not scores_path.exists():
        scores_path.write_text("", encoding="utf-8")
    _probe_step(cfg, rd, scores_path)
    return rd
