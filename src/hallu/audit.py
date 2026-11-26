"""Label audit: sample auto-graded answers for human checking and score grader agreement."""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from hallu.data import load_questions
from hallu.io import read_jsonl

# The sheet is blind (no auto_label); the grader's labels live in a separate key file.
SHEET_COLUMNS = ["config_id", "qid", "question", "answer", "aliases", "human_label"]
KEY_COLUMNS = ["config_id", "qid", "auto_label", "stratum_size"]


def key_path(sheet) -> Path:
    sheet = Path(sheet)
    return sheet.with_name(f"{sheet.stem}_key.csv")


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
    sizes: dict[tuple, int] = defaultdict(int)
    for r in candidates:
        sizes[(r["config_id"], r["label"])] += 1
    sample = stratified_sample(candidates, n, seed)
    random.Random(seed).shuffle(sample)  # hide the config/label round-robin order
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for r in sample:
            w.writerow({"config_id": r["config_id"], "qid": r["qid"], "question": r["question"],
                        "answer": r["answer"], "aliases": " | ".join(r["aliases"]),
                        "human_label": ""})
    with open(key_path(out_csv), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLUMNS)
        w.writeheader()
        for r in sample:
            w.writerow({"config_id": r["config_id"], "qid": r["qid"], "auto_label": r["label"],
                        "stratum_size": sizes[(r["config_id"], r["label"])]})


def score_audit(csv_path, key=None) -> dict:
    """Agreement of human labels (sheet) with auto labels (key; default <sheet stem>_key.csv).

    weighted_agreement weights each (config_id, auto_label) stratum's agreement by its
    stratum_size, undoing the equal-per-stratum sampling.
    """
    key = Path(key) if key is not None else key_path(csv_path)
    with open(key, newline="", encoding="utf-8-sig") as f:
        auto_of = {(r["config_id"], r["qid"]): r for r in csv.DictReader(f)}
    n = agree = fh = fc = 0
    strata: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])  # [n, agree, size]
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            h = (row.get("human_label") or "").strip()
            if h == "":
                continue
            if h not in ("0", "1"):
                raise ValueError(f"row {i} ({row.get('config_id')}/{row.get('qid')}): "
                                 f"human_label must be 0 or 1, got {h!r}")
            k = auto_of.get((row["config_id"], row["qid"]))
            if k is None:
                raise ValueError(f"row {i} ({row['config_id']}/{row['qid']}) not in key {key}")
            auto = int(str(k["auto_label"]).strip())
            st = strata[(row["config_id"], auto)]
            st[2] = int(k["stratum_size"])
            n += 1
            st[0] += 1
            if auto == int(h):
                agree += 1
                st[1] += 1
            elif auto == 1:
                fh += 1
            else:
                fc += 1
    agreement = round(agree / n, 4) if n else float("nan")
    w_total = sum(size for _, _, size in strata.values())
    weighted = (round(sum(size * a / m for m, a, size in strata.values()) / w_total, 4)
                if w_total else float("nan"))
    res = {"n": n, "agreement": agreement, "weighted_agreement": weighted,
           "false_hallucination": fh, "false_correct": fc}
    print(f"audit: n={n} agreement={agreement} weighted_agreement={weighted} "
          f"false_hallucination={fh} false_correct={fc}")
    if n and agreement < 0.90:
        print(f"WARNING: grader agreement {agreement:.4f} is below the 0.90 requirement",
              file=sys.stderr)
    return res
