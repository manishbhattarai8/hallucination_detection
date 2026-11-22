"""Label audit: sample auto-graded answers for human checking and score grader agreement."""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from hallu.data import load_questions
from hallu.io import read_jsonl

COLUMNS = ["config_id", "qid", "question", "answer", "aliases", "auto_label", "human_label"]


def stratified_sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Round-robin over sorted (config_id, label) strata, random order within each, until n rows."""
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["config_id"], r["label"])].append(r)
    queues = []
    for key in sorted(strata):
        q = list(strata[key])
        rng.shuffle(q)
        queues.append(q)
    out: list[dict] = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop())
    return out


def make_audit_sheet(cfgs, out_csv, n: int = 200, seed: int = 0) -> None:
    candidates = []
    for cfg in cfgs:
        d = Path(cfg.results_root) / cfg.config_id
        lab_path, gen_path = d / "labels.jsonl", d / "generations.jsonl"
        if not lab_path.exists() or not gen_path.exists():
            print(f"skipping {cfg.config_id}: no labels.jsonl", file=sys.stderr)
            continue
        labels = {r["qid"]: r["label"] for r in read_jsonl(lab_path)}
        answers = {r["qid"]: r["answer"] for r in read_jsonl(gen_path)}
        for q in load_questions(cfg):
            if q.qid in labels and q.qid in answers:
                candidates.append({"config_id": cfg.config_id, "qid": q.qid, "label": labels[q.qid],
                                   "answer": answers[q.qid], "question": q.question, "aliases": q.aliases})
    sample = stratified_sample(candidates, n, seed)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in sample:
            w.writerow({"config_id": r["config_id"], "qid": r["qid"], "question": r["question"],
                        "answer": r["answer"], "aliases": " | ".join(r["aliases"]),
                        "auto_label": r["label"], "human_label": ""})


def score_audit(csv_path) -> dict:
    n = agree = fh = fc = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            h = (row.get("human_label") or "").strip()
            if h == "":
                continue
            if h not in ("0", "1"):
                raise ValueError(f"row {i} ({row.get('config_id')}/{row.get('qid')}): "
                                 f"human_label must be 0 or 1, got {h!r}")
            auto = int(str(row["auto_label"]).strip())
            n += 1
            if auto == int(h):
                agree += 1
            elif auto == 1:
                fh += 1
            else:
                fc += 1
    agreement = round(agree / n, 4) if n else float("nan")
    res = {"n": n, "agreement": agreement, "false_hallucination": fh, "false_correct": fc}
    print(f"audit: n={n} agreement={agreement} false_hallucination={fh} false_correct={fc}")
    if n and agreement < 0.90:
        print(f"WARNING: grader agreement {agreement:.4f} is below the 0.90 requirement", file=sys.stderr)
    return res
