"""Controlled cost measurement: batch size 1 on a single GPU, synchronized timers.

Greedy decoding (t_generate) and the scoring forward (t_score: hidden states + token
stats) are timed separately; t_greedy = t_generate + t_score. The scoring forward is
attributed to confidence/probe; in deployment it can be folded into generate via
output_scores/output_hidden_states.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import torch

from .config import RunConfig, run_dir
from .data import load_questions
from .detectors.confidence import mean_entropy, neg_mean_logprob
from .detectors.semantic_entropy import NLIJudge
from .generate import greedy_generate, greedy_score, sample_pass
from .io import append_jsonl, env_info
from .model import load_model


class _Timer:
    seconds: float = float("nan")


@contextmanager
def cuda_timer():
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    t = _Timer()
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        if cuda:
            torch.cuda.synchronize()
        t.seconds = time.perf_counter() - t0


def _measure(model, tok, judge, probe, pos: int, q, cfg: RunConfig) -> dict:
    with cuda_timer() as t_generate:
        generated = greedy_generate(model, tok, [q], cfg)
    with cuda_timer() as t_score:
        g = greedy_score(model, tok, [q], cfg, generated)[0]
    with cuda_timer() as t_conf:
        neg_mean_logprob(g.token_logprobs)
        mean_entropy(g.token_entropies)
    with cuda_timer() as t_samples:
        samples = sample_pass(model, tok, q, cfg)
    with cuda_timer() as t_nli:
        judge.semantic_entropy(q.question, [g.answer] + list(samples))
    x = np.asarray(g.acts[pos], dtype=np.float32).reshape(1, -1)
    with cuda_timer() as t_probe:
        probe.predict_proba(x)
    return {"qid": q.qid, "t_greedy": t_generate.seconds + t_score.seconds,
            "t_generate": t_generate.seconds, "t_score": t_score.seconds,
            "t_conf": t_conf.seconds,
            "t_samples": t_samples.seconds, "t_nli": t_nli.seconds,
            "t_probe": t_probe.seconds}


def timing_run(cfg: RunConfig, n_warmup: int = 5, n_timed: int = 100) -> Path:
    if cfg.timing_excluded:
        raise RuntimeError(f"{cfg.config_id} is excluded from timing")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"timing needs exactly one GPU, found {torch.cuda.device_count()}")
    if cfg.dataset != "triviaqa":
        raise RuntimeError("timing runs on the TriviaQA subset only")

    rd = run_dir(cfg)
    probe_json = rd / "probe.json"
    pos = -1
    if probe_json.exists():
        pos = int(json.loads(probe_json.read_text(encoding="utf-8")).get("chosen_layer_pos", -1))
    if pos < 0 or not (rd / "probe.joblib").exists():
        raise RuntimeError(f"no usable probe in {rd}; run the main pipeline first "
                           "(cost needs all detectors)")
    probe = joblib.load(rd / "probe.joblib")

    questions = load_questions(cfg)
    if len(questions) < n_timed + n_warmup:
        raise ValueError(f"subset has {len(questions)} questions, need {n_timed + n_warmup}")
    timed = questions[:n_timed]
    warm = questions[n_timed:n_timed + n_warmup]

    out = Path(cfg.results_root) / "timing" / f"{cfg.config_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")

    model, tok = load_model(cfg.model, cfg.precision)
    judge = NLIJudge()
    for q in warm:
        _measure(model, tok, judge, probe, pos, q, cfg)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for q in timed:
        row = _measure(model, tok, judge, probe, pos, q, cfg)
        row["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
        append_jsonl(out, [row])
    env = env_info()
    append_jsonl(out, [{"summary": True, "gpu_name": env.get("gpu_name"),
                        "gpu_count": env.get("gpu_count")}])
    return out
