import dataclasses
import json

import pytest

from hallu.config import load_grid
import hallu.data as hd
from hallu.data import (FEW_SHOT, SYSTEM_PROMPT, build_messages, load_questions,
                        select_subset, split_ids)


def test_select_subset_deterministic_and_order_independent():
    ids = [f"q{i}" for i in range(5000)]
    a = select_subset(ids, 1500, seed=0)
    b = select_subset(list(reversed(ids)), 1500, seed=0)
    assert a == b and len(set(a)) == 1500


def test_split_is_60_20_20_disjoint_and_stable():
    ids = [f"q{i}" for i in range(1500)]
    s = split_ids(ids, seed=0)
    assert [len(s[k]) for k in ("train", "val", "test")] == [900, 300, 300]
    assert set(s["train"]).isdisjoint(s["test"]) and s == split_ids(ids, seed=0)


def test_messages_contain_system_fewshot_and_question():
    m = build_messages("Who painted the Mona Lisa?")
    assert m[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert len(m) == 1 + 2 * len(FEW_SHOT) + 1
    assert m[-1] == {"role": "user", "content": "Question: Who painted the Mona Lisa?"}


@pytest.mark.network
def test_load_triviaqa_subset_saved_and_reused(tmp_path):
    cfg = dataclasses.replace(load_grid("configs/smoke.yaml")[0], n_questions=10)
    qs = load_questions(cfg, subset_dir=tmp_path)
    assert len(qs) == 10 and all(q.aliases for q in qs)
    f = tmp_path / "triviaqa_n10_seed0.json"
    assert f.exists() and json.loads(f.read_text(encoding="utf-8")) == [q.qid for q in qs]
    qs2 = load_questions(cfg, subset_dir=tmp_path)
    assert [q.qid for q in qs] == [q.qid for q in qs2]


@pytest.mark.parametrize("ids", [["a", "b"], ["a", "b", "zzz"], ["a", "a", "b"]])
def test_stale_subset_file_raises_value_error(tmp_path, monkeypatch, ids):
    fake = {k: hd.Question(k, f"q{k}", ("x",)) for k in "abcd"}
    monkeypatch.setattr(hd, "_load_all", lambda dataset: fake)
    cfg = dataclasses.replace(load_grid("configs/smoke.yaml")[0], n_questions=3)
    f = tmp_path / "triviaqa_n3_seed0.json"
    f.write_text(json.dumps(ids), encoding="utf-8")
    with pytest.raises(ValueError, match="triviaqa_n3_seed0.json"):
        hd.load_questions(cfg, subset_dir=tmp_path)


def test_dataset_revisions_are_pinned(monkeypatch):
    import datasets

    seen = []

    def fake_load(name, *args, **kw):
        seen.append((name, kw.get("revision")))
        return []

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    hd._load_all("triviaqa")
    hd._load_all("nq_open")
    assert seen == [("mandarjoshi/trivia_qa", hd.TRIVIAQA_REVISION),
                    ("google-research-datasets/nq_open", hd.NQ_OPEN_REVISION)]
    assert all(len(r) == 40 for _, r in seen)


def test_fold_ids_deterministic_balanced_order_independent():
    from hallu.data import fold_ids
    qids = [f"q{i}" for i in range(103)]
    a = fold_ids(qids, k=5, seed=0)
    assert set(a) == set(qids) and set(a.values()) == set(range(5))
    sizes = sorted(list(a.values()).count(f) for f in range(5))
    assert sizes[-1] - sizes[0] <= 1
    assert fold_ids(list(reversed(qids)), k=5, seed=0) == a  # same folds whatever the order
    assert fold_ids(qids, k=5, seed=1) != a
