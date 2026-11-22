from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunConfig:
    config_id: str
    model: str
    size_b: float
    family: str
    precision: str
    dataset: str
    n_questions: int
    subset_seed: int
    split_seed: int
    sample_seed: int
    n_samples: int
    temperature: float
    max_new_tokens: int
    batch_size: int
    layer_fracs: tuple[float, ...]
    results_root: str
    timing_excluded: bool


def _short(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace("-instruct", "")


def load_grid(path: str | Path) -> list[RunConfig]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    d = spec["defaults"]
    excl = tuple(spec.get("timing_exclude") or [])
    out: list[RunConfig] = []
    for m in spec["models"]:
        for precision in spec["precisions"]:
            for dataset in spec["datasets"]:
                cid = f"{_short(m['id'])}_{precision}_{dataset}"
                out.append(RunConfig(
                    config_id=cid,
                    model=m["id"],
                    size_b=float(m["size_b"]),
                    family=spec["family"],
                    precision=precision,
                    dataset=dataset,
                    n_questions=int(d["n_questions"]),
                    subset_seed=int(d["subset_seed"]),
                    split_seed=int(d["split_seed"]),
                    sample_seed=int(d["sample_seed"]),
                    n_samples=int(d["n_samples"]),
                    temperature=float(d["temperature"]),
                    max_new_tokens=int(d["max_new_tokens"]),
                    batch_size=int(m.get("batch_size", d.get("batch_size", 1))),
                    layer_fracs=tuple(float(x) for x in d["layer_fracs"]),
                    results_root=str(d["results_root"]),
                    timing_excluded=any(cid.startswith(p) for p in excl),
                ))
    return out


def run_dir(cfg: RunConfig) -> Path:
    p = Path(cfg.results_root) / cfg.config_id
    p.mkdir(parents=True, exist_ok=True)
    return p
