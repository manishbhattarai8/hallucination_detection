"""Stage 1: greedy answer + token stats + hidden states, and sampled answers (resumable)."""
from __future__ import annotations

import math
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from .config import RunConfig, run_dir
from .data import Question, build_messages, load_questions
from .io import append_jsonl, done_ids, env_info, write_json
from .model import layer_indices, load_model, token_stats, truncate_answer_ids


@dataclass
class GreedyResult:
    qid: str
    answer: str
    empty_answer: bool
    nonfinite: bool
    token_logprobs: list[float]
    token_entropies: list[float]
    acts: np.ndarray  # [n_layers_saved, hidden], fp16


def sample_seed_for(qid: str, base: int) -> int:
    return (base * 1_000_003 + zlib.crc32(qid.encode())) % 2**31


def _finite_or_none(xs: list[float]) -> list[float | None]:
    return [x if math.isfinite(x) else None for x in xs]


def _gen_kwargs(tok, **overrides) -> dict:
    # Explicit neutral values so the checkpoint's generation_config (e.g. Qwen's
    # temperature/top_p/top_k/repetition_penalty) neither alters decoding nor warns.
    kw = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
              pad_token_id=tok.pad_token_id)
    kw.update(overrides)
    return kw


def _prompt(tok, q: Question) -> str:
    return tok.apply_chat_template(build_messages(q.question), add_generation_prompt=True,
                                   tokenize=False)


def _left_pad(seqs: list[list[int]], pad_id: int, device):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, width - len(s):] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def greedy_pass(model, tok, questions: list[Question], cfg: RunConfig) -> list[GreedyResult]:
    layers = layer_indices(model.config.num_hidden_layers, cfg.layer_fracs)
    results: list[GreedyResult] = []
    for start in range(0, len(questions), cfg.batch_size):
        batch = questions[start:start + cfg.batch_size]
        enc = tok([_prompt(tok, q) for q in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        out = model.generate(**enc, **_gen_kwargs(tok, do_sample=False,
                                                  max_new_tokens=cfg.max_new_tokens))
        width = enc["input_ids"].shape[1]

        seqs, scored, answers = [], [], []
        for i in range(len(batch)):
            prompt_ids = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
            new_ids = out[i, width:].tolist()
            ans_ids = truncate_answer_ids(new_ids, tok)
            s = ans_ids if ans_ids else new_ids[:1]
            seqs.append(prompt_ids + s)
            scored.append(s)
            answers.append(ans_ids)

        ids, mask = _left_pad(seqs, tok.pad_token_id, model.device)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        fwd = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                    output_hidden_states=True)
        acts = torch.stack([fwd.hidden_states[j][:, -1, :].float().cpu() for j in layers], dim=1)
        acts = acts.numpy().astype(np.float16)

        for i, q in enumerate(batch):
            k = len(scored[i])
            lp, ent, bad = token_stats(fwd.logits[i, -k - 1:-1, :],
                                       torch.tensor(scored[i], device=fwd.logits.device))
            results.append(GreedyResult(
                qid=q.qid,
                answer=tok.decode(answers[i], skip_special_tokens=True).strip(),
                empty_answer=not answers[i],
                nonfinite=bad,
                token_logprobs=lp,
                token_entropies=ent,
                acts=acts[i],
            ))
    return results


class _FlagAndSanitize(LogitsProcessor):
    """Records whether raw logits were non-finite and replaces NaN/inf so sampling can't crash.

    Custom processors run before the temperature warper, so this sees the model's raw logits.
    """

    def __init__(self):
        self.nonfinite = False

    def __call__(self, input_ids, scores):
        if not bool(torch.isfinite(scores).all()):
            self.nonfinite = True
            fi = torch.finfo(scores.dtype)
            scores = torch.nan_to_num(scores, nan=0.0, posinf=fi.max, neginf=fi.min)
        return scores


@torch.no_grad()
def _sample_flagged(model, tok, q: Question, cfg: RunConfig) -> tuple[list[str], bool]:
    """Sampled answers plus whether any sampling step produced non-finite logits."""
    torch.manual_seed(sample_seed_for(q.qid, cfg.sample_seed))
    enc = tok([_prompt(tok, q)], return_tensors="pt", add_special_tokens=False).to(model.device)
    flag = _FlagAndSanitize()
    out = model.generate(**enc, logits_processor=LogitsProcessorList([flag]), **_gen_kwargs(
        tok, do_sample=True, temperature=cfg.temperature, top_p=1.0, top_k=0,
        num_return_sequences=cfg.n_samples, max_new_tokens=cfg.max_new_tokens))
    width = enc["input_ids"].shape[1]
    samples = [tok.decode(truncate_answer_ids(row[width:].tolist(), tok),
                          skip_special_tokens=True).strip() for row in out]
    return samples, flag.nonfinite


def sample_pass(model, tok, q: Question, cfg: RunConfig) -> list[str]:
    return _sample_flagged(model, tok, q, cfg)[0]


def _next_shard(acts_dir: Path) -> Path:
    idx = [int(p.stem.split("_")[1]) for p in acts_dir.glob("shard_*.npz")]
    return acts_dir / f"shard_{max(idx, default=-1) + 1:05d}.npz"


def load_acts(run_dir: Path) -> tuple[list[str], np.ndarray]:
    """Concatenate all shards; for a duplicated qid the last occurrence wins."""
    shards = sorted((Path(run_dir) / "acts").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no activation shards in {run_dir}/acts")
    by_qid: dict[str, np.ndarray] = {}
    for p in shards:
        with np.load(p) as z:
            for qid, a in zip(z["qids"].tolist(), z["acts"]):
                by_qid.pop(qid, None)
                by_qid[qid] = a
    return list(by_qid), np.stack(list(by_qid.values()))


def generate_run(cfg: RunConfig, limit: int | None = None) -> Path:
    rd = run_dir(cfg)
    write_json(rd / "meta.json", {"config": asdict(cfg), "env": env_info()})
    gen_path = rd / "generations.jsonl"
    questions = load_questions(cfg)
    if limit is not None:
        questions = questions[:limit]
    done = done_ids(gen_path)
    todo = [q for q in questions if q.qid not in done]
    if not todo:
        return rd

    model, tok = load_model(cfg.model, cfg.precision)
    acts_dir = rd / "acts"
    acts_dir.mkdir(exist_ok=True)
    for start in range(0, len(todo), cfg.batch_size):
        batch = todo[start:start + cfg.batch_size]
        t0 = time.perf_counter()
        greedy = greedy_pass(model, tok, batch, cfg)
        t_greedy = (time.perf_counter() - t0) / len(batch)

        rows = []
        for q, g in zip(batch, greedy):
            t0 = time.perf_counter()
            try:
                samples, sample_bad = _sample_flagged(model, tok, q, cfg)
            except RuntimeError:  # last resort: never let one question block the run
                samples, sample_bad = [""] * cfg.n_samples, True
            rows.append({
                "qid": g.qid,
                "answer": g.answer,
                "empty_answer": g.empty_answer,
                "nonfinite": g.nonfinite or sample_bad,
                "token_logprobs": _finite_or_none(g.token_logprobs),
                "token_entropies": _finite_or_none(g.token_entropies),
                "samples": samples,
                "t_greedy_s": t_greedy,
                "t_samples_s": time.perf_counter() - t0,
            })

        # Shard first, then JSONL: a crash in between leaves an orphan shard whose
        # qids are regenerated next run and superseded by load_acts (last wins).
        np.savez(_next_shard(acts_dir), qids=np.array([g.qid for g in greedy]),
                 acts=np.stack([g.acts for g in greedy]))
        append_jsonl(gen_path, rows)
        print(f"[{cfg.config_id}] {start + len(batch)}/{len(todo)} new", flush=True)
    return rd
