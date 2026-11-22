from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import RunConfig
from .io import write_json

# Frozen: changing the prompt invalidates every run.
SYSTEM_PROMPT = "Answer the question with a short phrase only."
FEW_SHOT: list[tuple[str, str]] = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote the novel 1984?", "George Orwell"),
    ("How many legs does a spider have?", "Eight"),
]


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    aliases: tuple[str, ...]


def select_subset(qids: Sequence[str], n: int, seed: int) -> list[str]:
    return random.Random(seed).sample(sorted(qids), n)


def split_ids(qids: Sequence[str], seed: int) -> dict[str, list[str]]:
    ids = sorted(qids)
    random.Random(seed).shuffle(ids)
    a, b = int(len(ids) * 0.6), int(len(ids) * 0.8)
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}


def build_messages(question: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in FEW_SHOT:
        msgs.append({"role": "user", "content": f"Question: {q}"})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"Question: {question}"})
    return msgs


def _dedup(items) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _load_all(dataset: str) -> dict[str, Question]:
    from datasets import load_dataset

    if dataset == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        return {
            r["question_id"]: Question(
                r["question_id"], r["question"],
                _dedup(list(r["answer"]["aliases"]) + [r["answer"]["value"]]))
            for r in ds
        }
    if dataset == "nq_open":
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        return {
            f"nq_{i}": Question(f"nq_{i}", r["question"], _dedup(r["answer"]))
            for i, r in enumerate(ds)
        }
    raise ValueError(f"unknown dataset: {dataset}")


def load_questions(cfg: RunConfig, subset_dir="data/subsets") -> list[Question]:
    path = Path(subset_dir) / f"{cfg.dataset}_n{cfg.n_questions}_seed{cfg.subset_seed}.json"
    allq = _load_all(cfg.dataset)
    if path.exists():
        ids = json.loads(path.read_text(encoding="utf-8"))
    else:
        ids = select_subset(list(allq), cfg.n_questions, cfg.subset_seed)
        write_json(path, ids)
    return [allq[i] for i in ids]
