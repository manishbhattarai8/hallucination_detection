"""Stage 4: metrics with bootstrap CIs, transfer, cost tables and figures."""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Literal

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from hallu.config import RunConfig  # noqa: E402
from hallu.detect import split_index  # noqa: E402
from hallu.generate import load_acts  # noqa: E402
from hallu.io import read_jsonl  # noqa: E402

DETECTORS = ["neg_mean_logprob", "mean_entropy", "semantic_entropy", "probe"]
PRECISIONS = ["fp16", "int8", "nf4"]
_NAN3 = (math.nan, math.nan, math.nan)
_FN = {"auroc": roc_auc_score, "auprc": average_precision_score}


def metric_ci(labels, scores, metric: Literal["auroc", "auprc"],
              n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(point, lo, hi); all NaN if either class has < 2 examples."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    if n_pos < 2 or len(y) - n_pos < 2:
        return _NAN3
    fn = _FN[metric]
    point = float(fn(y, s))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals.append(fn(yb, s[idx]))
    if not vals:
        return point, math.nan, math.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def _ci_fields(y, s) -> dict:
    out = {}
    for m in ("auroc", "auprc"):
        p, lo, hi = metric_ci(y, s, m)
        out.update({m: p, f"{m}_lo": lo, f"{m}_hi": hi})
    return out


def _load(cfg: RunConfig):
    """(labels dict, scores df joined on qid), or a reason string if the run is incomplete."""
    rd = Path(cfg.results_root) / cfg.config_id
    for name in ("labels.jsonl", "scores.jsonl"):
        if not (rd / name).exists():
            return f"{name} missing"
    labels = {r["qid"]: int(r["label"]) for r in read_jsonl(rd / "labels.jsonl")}
    sc = pd.DataFrame(read_jsonl(rd / "scores.jsonl"))
    sc = sc[sc.qid.isin(labels)].reset_index(drop=True)
    sc["label"] = sc.qid.map(labels)
    return labels, sc


def _metric_rows(cfg: RunConfig, labels: dict, sc: pd.DataFrame) -> list[dict]:
    n_correct = sum(1 for v in labels.values() if v == 0)
    test_pos = split_index(cfg, sc.qid.tolist())["test"]
    parts = {"test": sc.iloc[test_pos], "all": sc}
    rows = []
    for det in DETECTORS:
        for split, df in parts.items():
            if det == "probe" and split == "all":
                continue
            if det not in df:
                continue
            s = pd.to_numeric(df[det], errors="coerce").astype(float)
            ok = s.notna() & np.isfinite(s)
            y = df.label[ok].to_numpy()
            n_pos = int(y.sum())
            rows.append({
                "config_id": cfg.config_id, "family": cfg.family, "size_b": cfg.size_b,
                "precision": cfg.precision, "dataset": cfg.dataset, "detector": det,
                "split": split, "n": int(ok.sum()), "n_pos": n_pos,
                "n_neg": int(ok.sum()) - n_pos, "n_dropped_nonfinite": int((~ok).sum()),
                **_ci_fields(y, s[ok].to_numpy()), "low_confidence": n_correct < 100,
            })
    return rows


def _transfer_rows(cfgs: list[RunConfig], data: dict, missing: list) -> list[dict]:
    by_key = {(c.family, c.size_b, c.dataset, c.precision): c for c in cfgs}
    rows = []
    for c in cfgs:
        if c.precision == "fp16":
            continue
        base = by_key.get((c.family, c.size_b, c.dataset, "fp16"))
        why = None
        if base is None:
            why = "fp16 run missing"
        else:
            bd = Path(base.results_root) / base.config_id
            try:
                ok = (bd / "probe.joblib").exists() and                     int(json.loads((bd / "probe.json").read_text(encoding="utf-8"))
                        .get("chosen_layer_pos", -1)) >= 0
            except (OSError, ValueError):
                ok = False
            if not ok:
                why = "fp16 probe missing or degenerate"
        if why is None and c.config_id not in data:
            why = "run incomplete"
        if why:
            missing.append({"config_id": c.config_id, "table": "transfer", "reason": why})
    for (fam, size, ds, prec), base in by_key.items():
        if prec != "fp16":
            continue
        rd = Path(base.results_root) / base.config_id
        pj = rd / "probe.json"
        if not pj.exists() or not (rd / "probe.joblib").exists():
            continue
        meta = json.loads(pj.read_text(encoding="utf-8"))
        pos = int(meta.get("chosen_layer_pos", -1))
        if pos < 0:
            continue
        probe = joblib.load(rd / "probe.joblib")
        for tp in ("int8", "nf4"):
            other = by_key.get((fam, size, ds, tp))
            if other is None or other.config_id not in data:
                continue
            try:
                qids, acts = load_acts(Path(other.results_root) / other.config_id)
            except FileNotFoundError:
                continue
            labels = data[other.config_id][0]
            keep = [i for i, q in enumerate(qids) if q in labels]
            kq = [qids[i] for i in keep]
            test = split_index(other, kq)["test"]
            if len(test) == 0:
                continue
            x = np.asarray(acts[keep][test][:, pos, :], dtype=np.float32)
            y = np.array([labels[kq[i]] for i in test])
            s = probe.predict_proba(x)[:, 1]
            f = _ci_fields(y, s)
            rows.append({"family": fam, "size_b": size, "dataset": ds, "test_precision": tp,
                         "chosen_frac": meta.get("chosen_frac"),
                         "auroc": f["auroc"], "auroc_lo": f["auroc_lo"], "auroc_hi": f["auroc_hi"]})
    return rows


_EXTRA = {"neg_mean_logprob": ["t_conf"], "mean_entropy": ["t_conf"],
          "semantic_entropy": ["t_samples", "t_nli"], "probe": ["t_probe"]}


def _cost_rows(cfgs: list[RunConfig], metrics: pd.DataFrame, missing: list) -> list[dict]:
    rows = []
    for cfg in cfgs:
        path = Path(cfg.results_root) / "timing" / f"{cfg.config_id}.jsonl"

        def skip(why, cid=cfg.config_id):
            missing.append({"config_id": cid, "table": "cost", "reason": why})
        if cfg.timing_excluded:
            if path.parent.exists():
                skip("timing_excluded")
            continue
        if not path.exists():
            if path.parent.exists():
                skip("timing file missing")
            continue
        recs = read_jsonl(path)
        summ = [r for r in recs if r.get("summary")]
        timed = pd.DataFrame([r for r in recs if not r.get("summary")])
        if not summ:
            skip("timing summary row missing")
            continue
        if summ[-1].get("gpu_count") != 1:
            skip(f"gpu_count={summ[-1].get('gpu_count')}")
            continue
        if timed.empty:
            skip("no timed rows")
            continue
        for det, cols in _EXTRA.items():
            extra = timed[cols].sum(axis=1)
            total = timed["t_greedy"] + extra
            hit = metrics[(metrics.config_id == cfg.config_id) & (metrics.detector == det)
                          & (metrics.split == "test")]
            rows.append({
                "config_id": cfg.config_id, "detector": det,
                "extra_s_mean": extra.mean(), "extra_s_std": extra.std(ddof=1),
                "total_s_mean": total.mean(), "total_s_std": total.std(ddof=1),
                "n_generations": 6 if det == "semantic_entropy" else 1,
                "peak_vram_gb": timed["peak_vram_gb"].max(),
                "auroc": hit.auroc.iloc[0] if len(hit) else math.nan,
            })
    return rows


def _heatmap(df: pd.DataFrame, det: str, ds: str, fam: str, path: Path) -> None:
    d = df[(df.detector == det) & (df.dataset == ds) & (df.family == fam) & (df.split == "test")]
    sizes = sorted(d.size_b.unique())
    precs = [p for p in PRECISIONS if p in set(d.precision)]
    grid = np.full((len(sizes), len(precs)), np.nan)
    for r in d.itertuples():
        grid[sizes.index(r.size_b), precs.index(r.precision)] = r.auroc
    fig, ax = plt.subplots(figsize=(1.4 * len(precs) + 2, 0.7 * len(sizes) + 1.5))
    im = ax.imshow(np.ma.masked_invalid(grid), vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(precs)), precs)
    ax.set_yticks(range(len(sizes)), [f"{s:g}B" for s in sizes])
    for i in range(len(sizes)):
        for j in range(len(precs)):
            v = grid[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.3f}", ha="center", va="center",
                    color="black" if np.isnan(v) or v >= 0.75 else "white", fontsize=8)
    ax.set_title(f"{det} AUROC ({ds}, {fam})", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _line_fig(df: pd.DataFrame, ds: str, fam: str, path: Path) -> None:
    d = df[(df.dataset == ds) & (df.family == fam) & (df.split == "test")]
    precs = [p for p in PRECISIONS if p in set(d.precision)]
    fig, axes = plt.subplots(1, len(precs), figsize=(3.6 * len(precs), 3.2),
                             sharey=True, squeeze=False)
    for ax, prec in zip(axes[0], precs):
        plotted: list[np.ndarray] = []
        for det in DETECTORS:
            g = d[(d.precision == prec) & (d.detector == det)].sort_values("size_b")
            g = g[np.isfinite(g.auroc.astype(float))]
            if g.empty:
                continue
            ax.plot(g.size_b, g.auroc, marker="o", label=det)
            ci = g[np.isfinite(g.auroc_lo.astype(float)) & np.isfinite(g.auroc_hi.astype(float))]
            if not ci.empty:
                ax.fill_between(ci.size_b, ci.auroc_lo, ci.auroc_hi, alpha=0.2)
            plotted.append(g.size_b.to_numpy(dtype=float))
        if plotted and (np.concatenate(plotted) > 0).any():
            ax.set_xscale("log")
        if not plotted:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("size (B params)")
        ax.set_title(prec, fontsize=9)
    axes[0][0].set_ylabel("AUROC")
    if axes[0][0].get_legend_handles_labels()[0]:
        axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{ds}, {fam}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate_grid(cfgs: list[RunConfig], out_dir="results/summary") -> None:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    data, mrows, arows, missing = {}, [], [], []
    for cfg in cfgs:
        loaded = _load(cfg)
        if isinstance(loaded, str):
            missing.append({"config_id": cfg.config_id, "table": "metrics", "reason": loaded})
            warnings.warn(f"{cfg.config_id}: skipped in metrics ({loaded})", stacklevel=2)
            continue
        data[cfg.config_id] = loaded
        labels, sc = loaded
        if len(sc) < cfg.n_questions:
            why = f"run incomplete ({len(sc)}/{cfg.n_questions})"
            missing.append({"config_id": cfg.config_id, "table": "metrics", "reason": why})
            warnings.warn(f"{cfg.config_id}: evaluated on a partial run, {why}", stacklevel=2)
        mrows += _metric_rows(cfg, labels, sc)
        nc = sum(1 for v in labels.values() if v == 0)
        arows.append({"config_id": cfg.config_id, "family": cfg.family, "size_b": cfg.size_b,
                      "precision": cfg.precision, "dataset": cfg.dataset, "n": len(labels),
                      "accuracy": nc / len(labels) if labels else math.nan, "n_correct": nc})
    mcols = ["config_id", "family", "size_b", "precision", "dataset", "detector", "split", "n",
             "n_pos", "n_neg", "n_dropped_nonfinite", "auroc", "auroc_lo", "auroc_hi", "auprc",
             "auprc_lo", "auprc_hi", "low_confidence"]
    metrics = pd.DataFrame(mrows, columns=mcols)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(arows, columns=["config_id", "family", "size_b", "precision", "dataset", "n",
                                 "accuracy", "n_correct"]).to_csv(out / "accuracy.csv", index=False)
    pd.DataFrame(_transfer_rows(cfgs, data, missing),
                 columns=["family", "size_b", "dataset", "test_precision", "chosen_frac", "auroc",
                          "auroc_lo", "auroc_hi"]).to_csv(out / "transfer.csv", index=False)
    cost = _cost_rows(cfgs, metrics, missing)
    if cost:
        pd.DataFrame(cost).to_csv(out / "cost.csv", index=False)
    pd.DataFrame(missing, columns=["config_id", "table", "reason"]).to_csv(
        out / "missing.csv", index=False)
    for ds, fam in sorted({(c.dataset, c.family) for c in cfgs if c.config_id in data}):
        for det in DETECTORS:
            _heatmap(metrics, det, ds, fam, out / "figures" / f"heatmap_{det}_{ds}_{fam}.pdf")
        _line_fig(metrics, ds, fam, out / "figures" / f"auroc_vs_size_{ds}_{fam}.pdf")
