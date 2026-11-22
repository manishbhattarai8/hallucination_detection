# Kaggle runner. Paste each "# %%" cell into a notebook cell, or File -> Import this file.
# Plain Python (no "!" magics), so it also parses as a normal script.

# %% [1] Settings -- edit these
REPO = "https://github.com/manishbhattarai8/hallucination_detection.git"
BRANCH = "feat/pipeline"          # switch to "main" once merged
GRID = "configs/qwen.yaml"        # or configs/llama.yaml, configs/smoke.yaml
CONFIG_IDS = ["qwen2.5-0.5b_fp16_triviaqa", "qwen2.5-3b_nf4_triviaqa"]  # [] = whole grid
STAGES = ["all"]                  # any of: generate grade detect timing evaluate all
LIMIT = None                      # e.g. 20 for a quick generate-only check
PREV_RESULTS = None               # e.g. "/kaggle/input/hallu-results/results" to resume

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
if PREV_RESULTS:
    shutil.copytree(PREV_RESULTS, "results", dirs_exist_ok=True)
    print("restored previous results from", PREV_RESULTS)

for stage in STAGES:
    cmd = [sys.executable, "scripts/run_stage.py", "--grid", GRID, "--stage", stage]
    if CONFIG_IDS:
        cmd += ["--config-id", *CONFIG_IDS]
    if LIMIT:
        cmd += ["--limit", str(LIMIT)]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

# %% [7] Zip results for download
shutil.make_archive(f"{WORK}/results", "zip", root_dir=REPO_DIR, base_dir="results")
print("Wrote /kaggle/working/results.zip")
print("To use locally: download results.zip from the notebook Output tab, then unzip it "
      "into your local repo root so that it creates <repo>/results/ "
      "(results/ is gitignored; never commit it).")
