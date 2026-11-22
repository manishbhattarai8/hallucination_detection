import csv
import json
from collections import Counter
from dataclasses import replace
from itertools import product

import pytest

import hallu.audit as audit
from hallu.audit import make_audit_sheet, score_audit, stratified_sample
from hallu.config import RunConfig
from hallu.data import Question


def test_stratified_sample_balances_strata():
    rows = [{"config_id": c, "label": l, "qid": f"{c}{l}{i}"} for c in "ABCD" for l in (0, 1) for i in range(100)]
    s = stratified_sample(rows, 200, seed=0)
    assert len(s) == 200
    assert Counter((r["config_id"], r["label"]) for r in s) == {k: 25 for k in product("ABCD", (0, 1))}


def test_stratified_sample_small_strata_exhausted_first():
    rows = [{"config_id": "A", "label": 0, "qid": f"a{i}"} for i in range(3)]
    rows += [{"config_id": "B", "label": 1, "qid": f"b{i}"} for i in range(50)]
    s = stratified_sample(rows, 20, seed=1)
    assert len(s) == 20
    assert sum(r["config_id"] == "A" for r in s) == 3
    assert len(stratified_sample(rows, 1000)) == 53


def _write_scored(tmp_path, human):
    p = tmp_path / "a.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["config_id", "qid", "question", "answer", "aliases", "auto_label", "human_label"])
        w.writeheader()
        for i, h in enumerate(human):
            auto = 1 if i == 0 else 0
            w.writerow({"config_id": "c", "qid": f"q{i}", "question": "q", "answer": "a", "aliases": "x",
                        "auto_label": auto, "human_label": h})
    return p


def test_score_audit(tmp_path):
    p = _write_scored(tmp_path, ["0"] + ["0"] * 9)
    assert score_audit(p) == {"n": 10, "agreement": 0.9, "false_hallucination": 1, "false_correct": 0}


def test_score_audit_blank_ignored_false_correct_and_warning(tmp_path, capsys):
    p = _write_scored(tmp_path, ["1", "", "1", " 0 "])
    r = score_audit(p)
    assert r == {"n": 3, "agreement": round(2 / 3, 4), "false_hallucination": 0, "false_correct": 1}
    assert "WARNING" in capsys.readouterr().err


def test_score_audit_bad_value(tmp_path):
    p = _write_scored(tmp_path, ["1", "yes"])
    with pytest.raises(ValueError, match="q1"):
        score_audit(p)


def _cfg(cid, root):
    return RunConfig(config_id=cid, model="m", size_b=1, family="f", precision="fp16", dataset="triviaqa",
                     n_questions=3, subset_seed=0, split_seed=0, sample_seed=0, n_samples=5, temperature=1.0,
                     max_new_tokens=32, batch_size=1, layer_fracs=(0.5,), results_root=str(root),
                     timing_excluded=False)


def test_make_audit_sheet(tmp_path, monkeypatch, capsys):
    root = tmp_path / "res"
    good, bad = _cfg("good", root), _cfg("bad", root)
    d = root / "good"
    d.mkdir(parents=True)
    qs = [Question(f"q{i}", f"Qué {i}?", ("A", "B")) for i in range(3)]
    (d / "generations.jsonl").write_text("".join(json.dumps({"qid": q.qid, "answer": f"ans{i}"}) + "\n" for i, q in enumerate(qs)))
    (d / "labels.jsonl").write_text("".join(json.dumps({"qid": q.qid, "label": i % 2}) + "\n" for i, q in enumerate(qs)))
    monkeypatch.setattr(audit, "load_questions", lambda cfg: qs)
    out = tmp_path / "sheet.csv"
    make_audit_sheet([good, bad], out, n=2, seed=0)
    assert "bad" in capsys.readouterr().err
    with open(out, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        assert rd.fieldnames == ["config_id", "qid", "question", "answer", "aliases", "auto_label", "human_label"]
        rows = list(rd)
    assert len(rows) == 2
    assert all(r["human_label"] == "" and r["aliases"] == "A | B" and r["config_id"] == "good" for r in rows)
    assert {r["auto_label"] for r in rows} == {"0", "1"}
