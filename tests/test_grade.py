from dataclasses import replace

import pytest

import hallu.grade as hg
from hallu.config import load_grid, run_dir
from hallu.data import Question
from hallu.grade import grade_run, is_correct, normalize
from hallu.io import append_jsonl, read_jsonl


def test_normalize():
    assert normalize("  The Beatles!! ") == "beatles"
    assert normalize("A  Tale of Two Cities.") == "tale of two cities"


@pytest.mark.parametrize("ans,aliases,ok", [
    ("Paris", ["paris"], True),
    ("It is Paris, France", ["Paris"], True),
    ("1999", ["1"], False),
    ("George Orwell", ["Orwell"], True),
    ("London", ["Paris"], False),
    ("", ["Paris"], False),
    ("anything", ["The", "Paris"], False),
])
def test_is_correct(ans, aliases, ok):
    assert is_correct(ans, aliases) is ok


def test_grade_run_labels_and_resumes(tmp_path, monkeypatch):
    cfg = replace(load_grid("configs/smoke.yaml")[0], results_root=str(tmp_path))
    qs = [Question("a", "q?", ("Paris",)), Question("b", "q?", ("Paris",)),
          Question("c", "q?", ("The", "Paris"))]
    monkeypatch.setattr(hg, "load_questions", lambda c: qs)
    gen = run_dir(cfg) / "generations.jsonl"
    append_jsonl(gen, [{"qid": "a", "answer": "Paris"}, {"qid": "b", "answer": ""},
                       {"qid": "c", "answer": "London"}])
    p = grade_run(cfg)
    assert [(r["qid"], r["label"]) for r in read_jsonl(p)] == [("a", 0), ("b", 1), ("c", 1)]
    append_jsonl(gen, [{"qid": "d", "answer": "x"}])
    qs.append(Question("d", "q?", ("x",)))
    grade_run(cfg)
    rows = read_jsonl(p)
    assert len(rows) == 4 and rows[-1] == {"qid": "d", "label": 0}
