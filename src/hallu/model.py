"""Model loading and per-token statistics."""
from __future__ import annotations

from typing import Sequence

import torch


def load_model(model_id: str, precision: str):
    """Load (model, tokenizer) at fp16 / int8 / nf4 with device_map='auto'."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if precision == "fp16":
        kwargs = {"dtype": torch.float16}
    elif precision == "int8":
        kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    elif precision == "nf4":
        kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        }
    else:
        raise ValueError(f"unknown precision: {precision!r}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", **kwargs)
    model.eval()
    return model, tokenizer


def layer_indices(n_layers: int, fracs: Sequence[float]) -> list[int]:
    """Indices into outputs.hidden_states (0 = embeddings); never 0."""
    return [max(1, round(f * n_layers)) for f in fracs]


def token_stats(logits: torch.Tensor, token_ids: torch.Tensor):
    """Per-token logprob and entropy (float32). Returns (logprobs, entropies, nonfinite)."""
    logits = logits.float()
    nonfinite = not bool(torch.isfinite(logits).all())
    logp = torch.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, token_ids.long().unsqueeze(-1)).squeeze(-1)
    ent = -(logp.exp() * logp).sum(-1)
    return lp.tolist(), ent.tolist(), nonfinite


def truncate_answer_ids(gen_ids: list[int], tokenizer) -> list[int]:
    """Cut at first EOS/pad token, or at first token whose text contains a newline."""
    stop = {t for t in (tokenizer.eos_token_id, tokenizer.pad_token_id) if t is not None}
    out: list[int] = []
    for i in gen_ids:
        if i in stop or "\n" in tokenizer.decode([i]):
            break
        out.append(i)
    return out
