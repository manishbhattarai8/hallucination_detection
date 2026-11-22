import json
import math
from dataclasses import replace

import numpy as np
import pytest

from hallu.config import load_grid, run_dir
from hallu.generate import _finite_or_none, generate_run, load_acts, sample_seed_for
from hallu.io import read_jsonl


def test_sample_seed_is_stable():
    assert sample_seed_for("q1", 1) == sample_seed_for("q1", 1) != sample_seed_for("q2", 1)
    assert 0 <= sample_seed_for("q1", 10**9) < 2**31


def test_finite_or_none_maps_nonfinite_to_null():
    assert _finite_or_none([1.5, math.nan, math.inf, -math.inf]) == [1.5, None, None, None]


def test_load_acts_dedupes_keeping_last(tmp_path):
    acts_dir = tmp_path / "acts"
    acts_dir.mkdir()
    a1 = np.zeros((2, 4, 3), dtype=np.float16)
    a2 = np.ones((2, 4, 3), dtype=np.float16)
    np.savez(acts_dir / "shard_00000.npz", qids=np.array(["a", "b"]), acts=a1)
    np.savez(acts_dir / "shard_00001.npz", qids=np.array(["b", "c"]), acts=a2)
    qids, acts = load_acts(tmp_path)
    assert sorted(qids) == ["a", "b", "c"] and len(qids) == 3
    assert acts.shape == (3, 4, 3) and acts.dtype == np.float16
    assert acts[qids.index("b")].min() == 1  # from shard 2
    assert acts[qids.index("a")].max() == 0


@pytest.mark.gpu
def test_generate_resume_no_duplicates(tmp_path):
    cfg = replace(load_grid("configs/smoke.yaml")[0], results_root=str(tmp_path))
    generate_run(cfg, limit=3)
    generate_run(cfg, limit=6)
    rows = read_jsonl(run_dir(cfg) / "generations.jsonl")
    assert len(rows) == 6 and len({r["qid"] for r in rows}) == 6
    assert all(len(r["samples"]) == 5 and len(r["token_logprobs"]) >= 1 for r in rows)
    qids, acts = load_acts(run_dir(cfg))
    assert acts.shape[:2] == (6, 4) and acts.dtype == np.float16
    assert "gpu_name" in json.load(open(run_dir(cfg) / "meta.json"))["env"]
