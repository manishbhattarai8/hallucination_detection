import importlib.util
from pathlib import Path

import pytest

_p = Path(__file__).resolve().parents[1] / "scripts" / "run_stage.py"
_spec = importlib.util.spec_from_file_location("run_stage", _p)
run_stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_stage)
main = run_stage.main

GRID = str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")


def test_unknown_config_id_exits_2(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--grid", GRID, "--stage", "grade", "--config-id", "nope"])
    assert e.value.code == 2 and "qwen2.5-0.5b_fp16_triviaqa" in capsys.readouterr().err


def test_all_runs_stages_in_order(monkeypatch, capsys):
    calls = []
    for name in ("generate_run", "grade_run", "detect_run", "timing_run"):
        monkeypatch.setattr(run_stage, name, lambda cfg, *a, _n=name, **k: calls.append((_n, k)))
    monkeypatch.setattr(run_stage, "NLIJudge", lambda: "judge")
    monkeypatch.setattr(run_stage, "evaluate_grid", lambda cfgs, out_dir: calls.append(("evaluate", str(out_dir))))
    assert main(["--grid", GRID, "--stage", "all", "--limit", "3"]) == 0
    names = [c[0] for c in calls]
    assert names == ["generate_run", "grade_run", "detect_run", "evaluate"]
    assert calls[0][1] == {"limit": 3}
    assert calls[2][1] == {"nli": "judge"}
    assert calls[3][1].replace("\\", "/").endswith("results_smoke/summary")
