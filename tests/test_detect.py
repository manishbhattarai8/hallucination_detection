import json
import math
from dataclasses import replace

import numpy as np

import hallu.detect as D
from hallu.config import RunConfig, run_dir
from hallu.data import Question, split_ids
from hallu.io import append_jsonl, read_jsonl


class StubJudge:
    def __init__(self):
        self.calls = []

    def semantic_entropy(self, question, answers):
        self.calls.append((question, list(answers)))
        return 0.0, 1


def _cfg(tmp_path):
    return RunConfig(config_id="fake", model="m", size_b=0.5, family="qwen", precision="fp16",
                     dataset="triviaqa", n_questions=50, subset_seed=0, split_seed=3,
                     sample_seed=0, n_samples=5, temperature=1.0, max_new_tokens=32,
                     batch_size=1, layer_fracs=(0.25, 0.5, 0.75, 1.0),
                     results_root=str(tmp_path), timing_excluded=False)


def _build(tmp_path, monkeypatch, n=50):
    cfg = _cfg(tmp_path)
    rd = run_dir(cfg)
    qids = [f"q{i:03d}" for i in range(n)]
    qs = [Question(q, f"question {q}?", ("a",)) for q in qids]
    monkeypatch.setattr(D, "load_questions", lambda c, *a, **k: qs)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    y[:2] = [0, 1]
    rows = []
    for i, q in enumerate(qids):
        rows.append({"qid": q, "answer": f"ans{i}", "empty_answer": False,
                     "nonfinite": i == 7,
                     "token_logprobs": [None, -1.0] if i == 7 else [-0.5, -1.5],
                     "token_entropies": [None, 1.0] if i == 7 else [1.0, 3.0],
                     "samples": [f"s{j}" for j in range(5)],
                     "t_greedy_s": 0.1, "t_samples_s": 0.2})
    append_jsonl(rd / "generations.jsonl", rows)
    append_jsonl(rd / "labels.jsonl", [{"qid": q, "label": int(l)} for q, l in zip(qids, y)])
    acts = rng.normal(size=(n, 4, 8)).astype(np.float16)
    acts[:, 2, 0] += 3 * y
    (rd / "acts").mkdir()
    np.savez(rd / "acts" / "shard_00000.npz", qids=np.array(qids), acts=acts)
    return cfg, rd, qids


def test_detect_run_on_fake_results(tmp_path, monkeypatch):
    cfg, rd, qids = _build(tmp_path, monkeypatch)
    judge = StubJudge()
    D.detect_run(cfg, nli=judge)
    rows = {r["qid"]: r for r in read_jsonl(rd / "scores.jsonl")}
    assert len(rows) == 50
    assert rows["q007"]["neg_mean_logprob"] is None and rows["q007"]["mean_entropy"] is None
    assert rows["q007"]["semantic_entropy"] == 0.0
    assert rows["q000"]["neg_mean_logprob"] == 1.0 and rows["q000"]["mean_entropy"] == 2.0
    test_q = set(split_ids(qids, cfg.split_seed)["test"])
    assert all((rows[q]["probe"] is not None) == (q in test_q) for q in rows)
    assert (rd / "probe.joblib").exists()
    pj = json.loads((rd / "probe.json").read_text())
    assert pj["chosen_layer_pos"] == 2 and "0.25" in pj["layer_val_auroc"]
    assert judge.calls[0][1] == ["ans0"] + [f"s{j}" for j in range(5)]
    assert "NaN" not in (rd / "scores.jsonl").read_text()


def test_resume_skips_done_and_probe_json(tmp_path, monkeypatch):
    cfg, rd, qids = _build(tmp_path, monkeypatch)
    D.detect_run(cfg, nli=StubJudge())
    before = (rd / "scores.jsonl").read_text()
    j2 = StubJudge()
    D.detect_run(cfg, nli=j2)
    assert j2.calls == [] and (rd / "scores.jsonl").read_text() == before


def test_lazy_judge_not_built_when_nothing_to_do(tmp_path, monkeypatch):
    cfg, rd, qids = _build(tmp_path, monkeypatch)
    D.detect_run(cfg, nli=StubJudge())
    monkeypatch.setattr(D, "NLIJudge", lambda: (_ for _ in ()).throw(AssertionError("built")))
    D.detect_run(cfg)


def test_degenerate_probe(tmp_path, monkeypatch):
    cfg, rd, qids = _build(tmp_path, monkeypatch)
    (rd / "labels.jsonl").write_text(
        "".join(json.dumps({"qid": q, "label": 0}) + "\n" for q in qids))
    D.detect_run(cfg, nli=StubJudge())
    rows = read_jsonl(rd / "scores.jsonl")
    assert len(rows) == 50 and all(r["probe"] is None for r in rows)
    assert not (rd / "probe.joblib").exists()
    pj = json.loads((rd / "probe.json").read_text())
    assert pj["chosen_frac"] is None and pj["chosen_layer_pos"] == -1
    assert all(v is None for v in pj["layer_val_auroc"].values())


def test_orphan_acts_and_missing_labels(tmp_path, monkeypatch):
    cfg, rd, qids = _build(tmp_path, monkeypatch)
    np.savez(rd / "acts" / "shard_00001.npz", qids=np.array(["orphan"]),
             acts=np.zeros((1, 4, 8), dtype=np.float16))
    D.detect_run(cfg, nli=StubJudge())
    rows = read_jsonl(rd / "scores.jsonl")
    assert len(rows) == 50 and "orphan" not in {r["qid"] for r in rows}


def test_split_index_positions(tmp_path, monkeypatch):
    cfg, rd, qids = _build(tmp_path, monkeypatch)
    sub = list(reversed(qids[:30]))
    si = D.split_index(cfg, sub)
    sp = split_ids(qids, cfg.split_seed)
    for k in ("train", "val", "test"):
        assert [sub[i] for i in si[k]] == [q for q in sub if q in set(sp[k])]


def test_sample_failed_rows_have_null_semantic_entropy(tmp_path, monkeypatch):
    cfg, rd, qids = _build(tmp_path, monkeypatch)
    rows = read_jsonl(rd / "generations.jsonl")
    for r in rows:
        r["sample_failed"] = r["qid"] == "q003"
    (rd / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    judge = StubJudge()
    D.detect_run(cfg, nli=judge)
    sc = {r["qid"]: r for r in read_jsonl(rd / "scores.jsonl")}
    assert sc["q003"]["semantic_entropy"] is None and sc["q003"]["n_clusters"] is None
    assert sc["q003"]["neg_mean_logprob"] == 1.0
    assert sc["q004"]["semantic_entropy"] == 0.0 and len(judge.calls) == 49
