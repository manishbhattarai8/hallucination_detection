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


def _write_scored(tmp_path, human, sizes=(300, 100)):
    """Blind sheet + key; row 0 auto=1 (stratum size sizes[0]), others auto=0 (sizes[1])."""
    p = tmp_path / "a.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f, \
            open(tmp_path / "a_key.csv", "w", newline="", encoding="utf-8-sig") as k:
        w = csv.DictWriter(f, fieldnames=audit.SHEET_COLUMNS)
        kw = csv.DictWriter(k, fieldnames=audit.KEY_COLUMNS)
        w.writeheader()
        kw.writeheader()
        for i, h in enumerate(human):
            auto = 1 if i == 0 else 0
            w.writerow({"config_id": "c", "qid": f"q{i}", "question": "q", "answer": "a",
                        "aliases": "x", "human_label": h})
            kw.writerow({"config_id": "c", "qid": f"q{i}", "auto_label": auto,
                         "stratum_size": sizes[0] if auto else sizes[1]})
    return p


def test_score_audit(tmp_path):
    p = _write_scored(tmp_path, ["0"] + ["0"] * 9)
    assert score_audit(p) == {"n": 10, "agreement": 0.9, "weighted_agreement": 0.25,
                              "false_hallucination": 1, "false_correct": 0}


def test_score_audit_explicit_key_path(tmp_path):
    p = _write_scored(tmp_path, ["1", "0"])
    key = tmp_path / "elsewhere.csv"
    (tmp_path / "a_key.csv").rename(key)
    with pytest.raises(FileNotFoundError):
        score_audit(p)
    assert score_audit(p, key=key)["agreement"] == 1.0


def test_score_audit_blank_ignored_false_correct_and_warning(tmp_path, capsys):
    p = _write_scored(tmp_path, ["1", "", "1", " 0 "])
    r = score_audit(p)
    assert r == {"n": 3, "agreement": round(2 / 3, 4), "weighted_agreement": round(
        (300 * 1 + 100 * 0.5) / 400, 4), "false_hallucination": 0, "false_correct": 1}
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
        assert rd.fieldnames == ["config_id", "qid", "question", "answer", "aliases", "human_label"]
        rows = list(rd)
    assert len(rows) == 2
    assert all(r["human_label"] == "" and r["aliases"] == "A | B" and r["config_id"] == "good" for r in rows)
    with open(tmp_path / "sheet_key.csv", newline="", encoding="utf-8-sig") as f:
        kr = csv.DictReader(f)
        assert kr.fieldnames == ["config_id", "qid", "auto_label", "stratum_size"]
        key = {r["qid"]: r for r in kr}
    assert set(key) == {r["qid"] for r in rows}
    assert {r["auto_label"] for r in key.values()} == {"0", "1"}
    # q0, q2 have label 0 (stratum size 2), q1 has label 1 (size 1)
    assert all(r["stratum_size"] == ("1" if r["auto_label"] == "1" else "2") for r in key.values())


def test_audit_sheet_rows_shuffled_with_seed(tmp_path, monkeypatch):
    root = tmp_path / "res"
    cfgs = [_cfg(f"c{j}", root) for j in range(4)]
    qs = [Question(f"q{i}", f"Q{i}?", ("A",)) for i in range(3)]
    for c in cfgs:
        d = root / c.config_id
        d.mkdir(parents=True)
        (d / "generations.jsonl").write_text(
            "".join(json.dumps({"qid": q.qid, "answer": "a"}) + "\n" for q in qs))
        (d / "labels.jsonl").write_text(
            "".join(json.dumps({"qid": q.qid, "label": i % 2}) + "\n" for i, q in enumerate(qs)))
    monkeypatch.setattr(audit, "load_questions", lambda cfg: qs)

    def order(name, seed):
        make_audit_sheet(cfgs, tmp_path / name, n=12, seed=seed)
        with open(tmp_path / name, newline="", encoding="utf-8-sig") as f:
            return [(r["config_id"], r["qid"]) for r in csv.DictReader(f)]

    a, b, c = order("a.csv", 0), order("b.csv", 0), order("c.csv", 1)
    assert a == b and sorted(a) == sorted(c) and a != c
    # stratified_sample alone yields configs in round-robin order regardless of seed
    round_robin = ["c0", "c0", "c1", "c1", "c2", "c2", "c3", "c3", "c0", "c1", "c2", "c3"]
    assert [x for x, _ in a] != round_robin and [x for x, _ in c] != round_robin


def test_audit_cli_score_with_key(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts" / "audit.py"
    spec = importlib.util.spec_from_file_location("audit_cli", p)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    sheet = _write_scored(tmp_path, ["1", "0"])
    key = tmp_path / "k.csv"
    (tmp_path / "a_key.csv").rename(key)
    cli.main(["score", str(sheet), "--key", str(key)])
    assert "weighted_agreement=1.0" in capsys.readouterr().out
