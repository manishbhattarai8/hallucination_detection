"""CLI: build / score the human label-audit sheet."""
from __future__ import annotations

import argparse

from hallu.audit import make_audit_sheet, score_audit
from hallu.config import load_grid


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--grid", action="append", required=True, help="grid yaml (repeatable)")
    mk.add_argument("--out", required=True)
    mk.add_argument("--n", type=int, default=200)
    mk.add_argument("--seed", type=int, default=0)
    sc = sub.add_parser("score")
    sc.add_argument("csv")
    a = ap.parse_args(argv)
    if a.cmd == "make":
        cfgs = [c for g in a.grid for c in load_grid(g)]
        make_audit_sheet(cfgs, a.out, n=a.n, seed=a.seed)
    else:
        score_audit(a.csv)


if __name__ == "__main__":
    main()
