"""Discrete semantic entropy with bidirectional-entailment clustering."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

from hallu.grade import normalize


def cluster_by_entailment(
    answers: Sequence[str], entails: Callable[[str, str], bool]
) -> list[int]:
    """Greedy clustering; compare against the first member of each cluster."""
    reps: list[str] = []
    ids: list[int] = []
    for a in answers:
        na = normalize(a)
        assigned = -1
        for cid, rep in enumerate(reps):
            if na == normalize(rep) or (entails(a, rep) and entails(rep, a)):
                assigned = cid
                break
        if assigned < 0:
            reps.append(a)
            assigned = len(reps) - 1
        ids.append(assigned)
    return ids


def discrete_semantic_entropy(cluster_ids: Sequence[int]) -> float:
    n = len(cluster_ids)
    if n == 0:
        return float("nan")
    h = -sum((c / n) * math.log(c / n) for c in Counter(cluster_ids).values())
    return h + 0.0  # avoid -0.0


class NLIJudge:
    def __init__(self, model_name: str = "microsoft/deberta-large-mnli", device: str = "cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # Load without a dtype kwarg, then cast explicitly: from_pretrained(dtype=fp16)
        # leaves some DeBERTa params (q_bias/v_bias) in fp32, which breaks matmul.
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(device=device, dtype=dtype)
            .eval()
        )
        id2label = self.model.config.id2label
        self.entail_id = next(i for i, l in id2label.items() if str(l).upper() == "ENTAILMENT")
        self._cache: dict[tuple[str, str, str], bool] = {}

    def _predict(self, question: str, pairs: list[tuple[str, str]]) -> list[bool]:
        import torch

        enc = self.tok(
            [f"{question} {a}" for a, _ in pairs],
            [f"{question} {b}" for _, b in pairs],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return (logits.argmax(-1) == self.entail_id).tolist()

    def entails(self, question: str, a: str, b: str) -> bool:
        key = (question, a, b)
        if key not in self._cache:
            self._cache[key] = self._predict(question, [(a, b)])[0]
        return self._cache[key]

    def semantic_entropy(self, question: str, answers: Sequence[str]) -> tuple[float, int]:
        self._cache = {}  # cache is per question call
        ids = cluster_by_entailment(answers, lambda a, b: self.entails(question, a, b))
        return discrete_semantic_entropy(ids), len(set(ids))
