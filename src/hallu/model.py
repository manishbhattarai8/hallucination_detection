"""Model loading and per-token statistics."""
from __future__ import annotations

import gc
from typing import Iterable, Sequence

import torch


def load_model(model_id: str, precision: str):
    """Load (model, tokenizer) at fp16 / int8 / nf4 with device_map='auto'.

    dtype=float16 is passed for every precision: without it transformers uses the
    checkpoint dtype (bf16) for the non-quantized parts (embeddings, norms, lm_head).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if precision == "fp16":
        kwargs = {"dtype": torch.float16}
    elif precision == "int8":
        kwargs = {"dtype": torch.float16,
                  "quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    elif precision == "nf4":
        kwargs = {
            "dtype": torch.float16,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        }
    else:
        raise ValueError(f"unknown precision: {precision!r}")

    # Release the previous config's model first. device_map="auto" plans from *free*
    # GPU memory; unreclaimed or allocator-cached memory would make it offload layers
    # to CPU (seen as OOM for 7B fp16 on 2x T4 when it was the second config).
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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


def truncate_answer_ids(gen_ids: list[int], tokenizer,
                        stop_ids: Iterable[int] | None = None) -> list[int]:
    """Cut at first EOS/pad/stop_ids token, or at first token whose text contains a newline."""
    stop = {t for t in (tokenizer.eos_token_id, tokenizer.pad_token_id) if t is not None}
    stop |= set(stop_ids or ())
    out: list[int] = []
    for i in gen_ids:
        if i in stop or "\n" in tokenizer.decode([i]):
            break
        out.append(i)
    return out
