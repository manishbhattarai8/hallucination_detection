"""CLI entrypoint: run pipeline stages over a config grid."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from hallu.config import load_grid
from hallu.detect import detect_run
from hallu.detectors.semantic_entropy import NLIJudge
from hallu.evaluate import evaluate_grid
from hallu.generate import generate_run
from hallu.grade import grade_run
from hallu.timing import timing_run

STAGES = ["generate", "grade", "detect", "timing", "evaluate", "all"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", action="append", required=True, help="grid yaml (repeatable)")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--config-id", nargs="+", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None, help="generate stage only")
    ap.add_argument("--deadline-hours", type=float, default=None,
                    help="wall-clock budget; stop cleanly (exit 0) when reached, rerun to resume")
    a = ap.parse_args(argv)
    deadline = None if a.deadline_hours is None else time.monotonic() + a.deadline_hours * 3600

    def out_of_time() -> bool:
        if deadline is not None and time.monotonic() >= deadline:
            print("deadline reached, stopping cleanly; rerun to resume", flush=True)
            return True
        return False

    cfgs = [c for g in a.grid for c in load_grid(g)]
    by_id = {c.config_id: c for c in cfgs}
    if a.config_id:
        ids = [i for grp in a.config_id for i in grp]
        bad = [i for i in ids if i not in by_id]
        if bad:
            ap.error(f"unknown config id(s) {bad}; valid ids: {sorted(by_id)}")
        sel = [by_id[i] for i in dict.fromkeys(ids)]
    else:
        sel = cfgs

    def log(cfg, stage):
        print(f"[{stage}] {cfg.config_id}", flush=True)

    judge = None
    for cfg in sel:
        if a.stage in ("generate", "all"):
            if out_of_time():
                return 0
            log(cfg, "generate")
            generate_run(cfg, limit=a.limit, deadline=deadline)
        if a.stage in ("grade", "all"):
            if out_of_time():
                return 0
            log(cfg, "grade")
            grade_run(cfg)
        if a.stage in ("detect", "all"):
            if out_of_time():
                return 0
            log(cfg, "detect")
            if judge is None:
                judge = NLIJudge()
            detect_run(cfg, nli=judge, deadline=deadline)
        if a.stage == "timing":
            if out_of_time():
                return 0
            log(cfg, "timing")
            timing_run(cfg)
    if a.stage in ("evaluate", "all"):
        if out_of_time():
            return 0
        out = Path(sel[0].results_root) / "summary"
        print(f"[evaluate] {len(sel)} config(s) -> {out}", flush=True)
        evaluate_grid(sel, out_dir=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
