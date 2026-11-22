# Kaggle runner. Paste each "# %%" cell into a notebook cell, or File -> Import this file.
# Plain Python (no "!" magics), so it also parses as a normal script.

# %% [1] Settings -- edit these
REPO = "https://github.com/manishbhattarai8/hallucination_detection.git"
BRANCH = "feat/pipeline"          # switch to "main" once merged
GRID = "configs/qwen.yaml"        # or configs/llama.yaml, configs/smoke.yaml
CONFIG_IDS = ["qwen2.5-0.5b_fp16_triviaqa", "qwen2.5-3b_nf4_triviaqa"]  # [] = whole grid
STAGES = ["all"]                  # any of: generate grade detect timing evaluate all
LIMIT = None                      # smoke tests only (GRID = "configs/smoke.yaml", scratch results root);
                                  # keep None for real runs: probe and evaluate need complete runs
PREV_RESULTS = None               # e.g. "/kaggle/input/hallu-results/hallucination/results" to resume
MAKE_ZIP = False                  # True only on the final session (zip doubles stored size)
TIME_BUDGET_H = 11                # stop cleanly before Kaggle's 12 h cap (counted from this cell)

import time

SESSION_START = time.monotonic()

# %% [2] GPU check
import subprocess
import sys

subprocess.run(["nvidia-smi"], check=True)

# %% [3] Clone repo and install (keeps Kaggle's preinstalled torch)
import os
import shutil

WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/hallucination"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "-b", BRANCH, "--depth", "1", REPO, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
os.chdir(REPO_DIR)

# Drop numpy== and torch lines: Kaggle's torch build is tied to its own numpy.
with open("requirements.txt", encoding="utf-8") as f:
    reqs = [ln for ln in f.read().splitlines()
            if ln.strip() and not ln.lower().startswith(("numpy==", "torch"))]
with open("/tmp/requirements_kaggle.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(reqs) + "\n")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/tmp/requirements_kaggle.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)

# %% [4] Environment report; assert T4
import bitsandbytes
import torch
import transformers

print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| bitsandbytes", bitsandbytes.__version__)
gpu = torch.cuda.get_device_name(0)
print("GPU:", gpu, "| count:", torch.cuda.device_count())
assert "T4" in gpu, f"expected a T4, got {gpu}"

# %% [5] Hugging Face login (Kaggle secret HF_TOKEN)
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

login(UserSecretsClient().get_secret("HF_TOKEN"))

# %% [6] Resume from an attached Dataset (optional), then run
import glob


def resolve_prev(given):
    """Find the results dir inside an attached Dataset; never silently continue."""
    tried = [f"{given}/hallucination/results", f"{given}/results", given]
    for c in tried:
        if os.path.isdir(c):
            return c
    globbed = sorted(glob.glob("/kaggle/input/*/hallucination/results"))
    tried += globbed or ["/kaggle/input/*/hallucination/results (no match)"]
    for c in globbed:
        return c
    raise FileNotFoundError("PREV_RESULTS not found; tried: " + ", ".join(tried))


if PREV_RESULTS:
    src = resolve_prev(PREV_RESULTS)
    shutil.copytree(src, "results", dirs_exist_ok=True)
    n_gen = sum(1 for _ in glob.glob("results/**/generations.jsonl", recursive=True))
    n_rows = 0
    for g in glob.glob("results/**/generations.jsonl", recursive=True):
        with open(g, encoding="utf-8") as f:
            n_rows += sum(1 for _ in f)
    print(f"restored from {src}: {n_gen} run dirs with generations, {n_rows} generation rows")

for stage in STAGES:
    left_h = TIME_BUDGET_H - (time.monotonic() - SESSION_START) / 3600
    if left_h <= 0:
        print("time budget used up; save this version and rerun to resume")
        break
    cmd = [sys.executable, "scripts/run_stage.py", "--grid", GRID, "--stage", stage,
           "--deadline-hours", f"{left_h:.4f}"]
    if CONFIG_IDS:
        cmd += ["--config-id", *CONFIG_IDS]
    if LIMIT:
        cmd += ["--limit", str(LIMIT)]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

# %% [7] Optional zip (off by default: it would be stored again in every saved version)
if MAKE_ZIP:
    shutil.make_archive("/kaggle/working/results", "zip", root_dir=REPO_DIR, base_dir="results")
    print("Wrote /kaggle/working/results.zip (delete it before 'Save Version' if you continue later)")
print("To use locally: download the saved output (or results.zip) and unzip so that "
      "<repo>/results/ exists (results/ is gitignored; never commit it).")
