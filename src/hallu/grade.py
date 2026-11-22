from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from hallu.config import RunConfig, run_dir
from hallu.data import load_questions
from hallu.io import append_jsonl, done_ids, read_jsonl

_PUNCT = re.compile(r"[^\w\s]|_")
_ARTICLES = {"a", "an", "the"}


def _tokens(text: str) -> list[str]:
    return [t for t in _PUNCT.sub(" ", text.lower()).split() if t not in _ARTICLES]


def normalize(text: str) -> str:
    return " ".join(_tokens(text))


def is_correct(answer: str, aliases: Sequence[str]) -> bool:
    ans = _tokens(answer)
    for alias in aliases:
        al = _tokens(alias)
        if not al:
            continue
        n = len(al)
        if any(ans[i:i + n] == al for i in range(len(ans) - n + 1)):
            return True
    return False


def grade_run(cfg: RunConfig) -> Path:
    d = run_dir(cfg)
    out = d / "labels.jsonl"
    done = done_ids(out)
    aliases = {q.qid: q.aliases for q in load_questions(cfg)}
    rows = [{"qid": g["qid"], "label": 0 if is_correct(g["answer"], aliases[g["qid"]]) else 1}
            for g in read_jsonl(d / "generations.jsonl") if g["qid"] not in done]
    append_jsonl(out, rows)
    return out
