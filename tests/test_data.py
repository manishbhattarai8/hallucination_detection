import dataclasses

import pytest

from hallu.config import load_grid
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
    assert (tmp_path / "triviaqa_n10_seed0.json").exists() or list(tmp_path.glob("*.json"))
    qs2 = load_questions(cfg, subset_dir=tmp_path)
    assert [q.qid for q in qs] == [q.qid for q in qs2]
