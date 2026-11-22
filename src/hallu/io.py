from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Iterable


def append_jsonl(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path) -> list[dict]:
    """Read rows; a malformed final line (crash mid-write) is skipped, others raise."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    lines = [ln for ln in lines if ln.strip()]
    rows = []
    for i, ln in enumerate(lines):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
    return rows


def done_ids(path, key: str = "qid") -> set[str]:
    if not Path(path).exists():
        return set()
    return {r[key] for r in read_jsonl(path)}


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _version(mod: str):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "bitsandbytes": _version("bitsandbytes"),
        "datasets": _version("datasets"),
        "sklearn": _version("sklearn"),
        "gpu_name": "cpu",
        "gpu_count": 0,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    return info
