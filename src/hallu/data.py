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

# Pinned Hub dataset commits (HfApi().dataset_info(...).sha on 2026-11-22), so the subset
# id lists keep resolving to the same questions and aliases.
TRIVIAQA_REVISION = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
NQ_OPEN_REVISION = "5dd9790a83002ad084ddeb7c420dc716852c6f28"


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


def fold_ids(qids: Sequence[str], k: int, seed: int) -> dict[str, int]:
    """Cross-validation fold (0..k-1) per qid; depends only on the qid set, so pass the
    full subset to get identical folds across configs."""
    ids = sorted(qids)
    random.Random(seed).shuffle(ids)
    return {q: i % k for i, q in enumerate(ids)}


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
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation",
                          revision=TRIVIAQA_REVISION)
        return {
            r["question_id"]: Question(
                r["question_id"], r["question"],
                _dedup(list(r["answer"]["aliases"]) + [r["answer"]["value"]]))
            for r in ds
        }
    if dataset == "nq_open":
        ds = load_dataset("google-research-datasets/nq_open", split="validation",
                          revision=NQ_OPEN_REVISION)
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
        problem = None
        if len(ids) != cfg.n_questions:
            problem = f"has {len(ids)} ids, expected {cfg.n_questions}"
        elif len(set(ids)) != len(ids):
            problem = "contains duplicate ids"
        elif any(i not in allq for i in ids):
            problem = "contains ids absent from the dataset"
        if problem:
            raise ValueError(f"stale subset file {path}: {problem}")
    else:
        ids = select_subset(list(allq), cfg.n_questions, cfg.subset_seed)
        write_json(path, ids)
    return [allq[i] for i in ids]
