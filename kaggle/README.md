# Running on Kaggle (free T4)

All reported experiments run on Kaggle. Budget: about 30 GPU-hours per week, 12 hours max per session.

## One-time setup
1. Create a notebook at kaggle.com -> Code -> New Notebook.
2. Settings -> Accelerator: **GPU T4** (single). Use **GPU T4 x2** only for `qwen2.5-7b_fp16_*` configs.
3. Settings -> Internet: **On** (phone verification is required once).
4. Add-ons -> Secrets: add `HF_TOKEN` (your Hugging Face token) and tick it for this notebook. Your Llama licenses must already be approved.
5. Load the runner: File -> Import Notebook and pick `kaggle/run_on_kaggle.py`, or paste each `# %%` cell into its own cell.
6. Edit the first cell: `GRID`, `CONFIG_IDS` (empty list = whole grid), `STAGES`. Run all cells.

`LIMIT` is for smoke testing only, into a scratch results root (e.g. `GRID = "configs/smoke.yaml"`, which writes to `results_smoke/`). Keep `LIMIT = None` for real runs: the probe is trained only once every subset question is generated and labeled (otherwise detect prints `probe skipped: n/N labeled`), and evaluate flags partial runs in `missing.csv` as `run incomplete (n/N)`.

The runner clones the repo to `/kaggle/working/hallucination`, installs `requirements.txt` without the `numpy==` and `torch` lines (Kaggle's torch build needs its own numpy; this is done automatically), asserts a T4, logs in to Hugging Face, runs `scripts/run_stage.py`, and zips results. If you run it manually and a pinned numpy conflicts: `pip install -r <(grep -v numpy requirements.txt)`.

## First run: the pilot
`GRID = "configs/qwen.yaml"`, `CONFIG_IDS = ["qwen2.5-0.5b_fp16_triviaqa", "qwen2.5-3b_nf4_triviaqa"]`, `STAGES = ["all"]`. Use its wall time x 24 configs (+6 Llama) to estimate the quota. If over about 30 GPU-h, set `n_questions: 1000` in both grids before any full run.

## Saving output and resuming
Only `/kaggle/working` is saved. Results are written to `/kaggle/working/hallucination/results`.
1. Click "Save Version" -> Save & Run All (Commit). When it finishes, open the version's Output tab -> "New Dataset" (name e.g. `hallu-results`). Later versions: "Update dataset" with the new output.
2. Next session, attach it: Add Input -> Your Datasets -> `hallu-results`.
3. Set `PREV_RESULTS = "/kaggle/input/hallu-results/hallucination/results"` (a Dataset made from Output holds the whole `/kaggle/working` tree). The runner also tries `<dir>/results` and globs `/kaggle/input/*/hallucination/results`, and raises FileNotFoundError if nothing is found. It prints how many run dirs and generation rows were restored. It is copied into `results/` before running; stages skip already-done question IDs.

Stop and save before the 12 h session cap; resume afterwards.

## Zip (optional)
`MAKE_ZIP = False` by default, because `results.zip` in `/kaggle/working` is saved again with every version and Dataset (about 20 GB limit). Set `MAKE_ZIP = True` only on the final session if you want one file to download. If you made it mid-project, delete `/kaggle/working/results.zip` before "Save Version".

## Copy results back into the repo
Download the saved output (or `results.zip`) and unzip so the local repo has `results/`. `results/` is gitignored; results are not committed to the public repo.

The fixed question subsets (`data/subsets/*_n1500_seed0.json`) are committed, so Kaggle and the laptop use identical questions.
