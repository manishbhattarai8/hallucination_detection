from hallu.config import load_grid, run_dir


def test_qwen_grid_expands_to_24(tmp_path):
    cfgs = load_grid("configs/qwen.yaml")
    assert len(cfgs) == 24
    ids = {c.config_id for c in cfgs}
    assert "qwen2.5-0.5b_fp16_triviaqa" in ids and "qwen2.5-7b_nf4_nq_open" in ids


def test_timing_exclusion_only_7b_fp16():
    cfgs = load_grid("configs/qwen.yaml")
    excluded = {c.config_id for c in cfgs if c.timing_excluded}
    assert excluded == {"qwen2.5-7b_fp16_triviaqa", "qwen2.5-7b_fp16_nq_open"}


def test_llama_grid_is_triviaqa_only():
    cfgs = load_grid("configs/llama.yaml")
    assert len(cfgs) == 6 and {c.dataset for c in cfgs} == {"triviaqa"}


def test_per_model_batch_size_and_defaults():
    c = next(c for c in load_grid("configs/qwen.yaml") if c.size_b == 7.0)
    assert c.batch_size == 8 and c.n_questions == 1500 and c.layer_fracs == (0.25, 0.5, 0.75, 1.0)


def test_smoke_and_run_dir(tmp_path):
    c = load_grid("configs/smoke.yaml")[0]
    assert c.n_questions == 20 and c.batch_size == 4 and c.results_root == "results_smoke"
    from dataclasses import replace
    c = replace(c, results_root=str(tmp_path))
    d = run_dir(c)
    assert d.is_dir() and d.name == c.config_id
