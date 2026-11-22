from hallu.io import append_jsonl, read_jsonl, done_ids, env_info, write_json
import json
import pytest


def test_append_then_read_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, [{"qid": "a"}, {"qid": "b"}])
    append_jsonl(p, [{"qid": "c"}])
    assert [r["qid"] for r in read_jsonl(p)] == ["a", "b", "c"]


def test_truncated_last_line_is_ignored(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, [{"qid": "a"}, {"qid": "b"}])
    with open(p, "a") as f:
        f.write('{"qid": "c", "ans')
    assert [r["qid"] for r in read_jsonl(p)] == ["a", "b"]
    assert done_ids(p) == {"a", "b"}


def test_corrupt_middle_line_raises(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"qid": "a"}\nnot json\n{"qid": "c"}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(p)


def test_done_ids_missing_file_is_empty(tmp_path):
    assert done_ids(tmp_path / "nope.jsonl") == set()


def test_write_json(tmp_path):
    write_json(tmp_path / "sub" / "m.json", {"a": 1})
    assert json.loads((tmp_path / "sub" / "m.json").read_text()) == {"a": 1}


def test_env_info_has_versions():
    e = env_info()
    assert {"python", "torch", "transformers", "gpu_name"} <= e.keys()


def test_append_after_truncated_tail_repairs_file(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, [{"qid": "a"}, {"qid": "b"}])
    with open(p, "a") as f:
        f.write('{"qid": "c", "ans')
    append_jsonl(p, [{"qid": "c"}])
    assert [r["qid"] for r in read_jsonl(p)] == ["a", "b", "c"]
    assert done_ids(p) == {"a", "b", "c"}
    append_jsonl(p, [{"qid": "d"}])
    assert [r["qid"] for r in read_jsonl(p)] == ["a", "b", "c", "d"]


def test_append_after_truncated_only_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_bytes(b'{"qid": "a", "an')
    append_jsonl(p, [{"qid": "b"}])
    assert [r["qid"] for r in read_jsonl(p)] == ["b"]


def test_tail_cut_mid_multibyte_char_is_ignored(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, [{"qid": "a", "answer": "café"}])
    with open(p, "ab") as f:
        f.write(b'{"answer": "caf\xc3')
    assert [r["qid"] for r in read_jsonl(p)] == ["a"]
