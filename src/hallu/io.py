from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _truncate_partial_tail(path: Path) -> None:
    """Drop a trailing line lacking a newline (crash mid-write) so appends stay line-aligned."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        data = f.read()
        if data.endswith(b"\n"):
            return
        f.truncate(data.rfind(b"\n") + 1)


def append_jsonl(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _truncate_partial_tail(path)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path) -> list[dict]:
    """Read rows; a malformed/undecodable final line (crash mid-write) is skipped, others raise."""
    lines = [ln for ln in Path(path).read_bytes().split(b"\n") if ln.strip()]
    rows = []
    for i, ln in enumerate(lines):
        try:
            rows.append(json.loads(ln.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
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


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True,
                           timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def env_info() -> dict:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if sha else None
    info = {
        "python": platform.python_version(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "bitsandbytes": _version("bitsandbytes"),
        "datasets": _version("datasets"),
        "sklearn": _version("sklearn"),
        "git_sha": sha.strip() if sha else None,
        "git_dirty": bool(status.strip()) if status is not None else None,
        "gpu_name": "cpu",
        "gpu_count": 0,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as e:  # recorded, not hidden: a broken CUDA setup must be visible
        info["gpu_error"] = f"{type(e).__name__}: {e}"
    return info


def record_meta(run_dir, stage: str, config: dict, **extra) -> dict:
    """Write meta.json (latest invocation) and append the same record to meta_history.jsonl."""
    rec = {"stage": stage, "time_utc": datetime.now(timezone.utc).isoformat(),
           "config": config, "env": env_info(), **extra}
    write_json(Path(run_dir) / "meta.json", rec)
    append_jsonl(Path(run_dir) / "meta_history.jsonl", [rec])
    return rec
