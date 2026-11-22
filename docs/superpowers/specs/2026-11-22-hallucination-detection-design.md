# Design: Hallucination Detection Under Model Shrinking and Quantization

Date: 2026-11-22
Target: arXiv preprint (cs.CL), no deadline

## 1. Intent

Most hallucination-detection methods were developed and evaluated on 7B+ models at full
precision. In practice, small models (0.5B–3B) are deployed precisely where resources are
tight, and almost always quantized (8-bit / 4-bit). Nobody has a clear picture of whether
detectors keep working in that regime.

**Research question:** How well do hallucination detectors hold up as language models get
smaller and more heavily quantized, and what does each detector cost to run?

**Expected contribution:** a controlled grid (model size × precision × detector) showing where
each detector family degrades, alongside its compute cost — giving practitioners a guide for
choosing a detector for a small/quantized deployment. A null result ("detectors are robust")
is still a publishable, useful finding.

## 2. Goals, non-goals, anti-goals

**Goals**
- Measure detection quality (AUROC) for three detector families across 4 model sizes × 3
  precisions, on two short-answer QA datasets.
- Measure compute cost of each detector on identical hardware.
- Replicate key findings on a second model family.
- Release code + results so the whole study is reproducible.
- Produce an arXiv-ready paper (LaTeX, ACL template).

**Non-goals**
- Proposing a new detection method.
- Long-form generation, summarization, dialogue, or RAG hallucinations.
- Quantization methods other than bitsandbytes (GPTQ/AWQ/GGUF → future work).
- Fine-tuning or training the language models themselves.

**Anti-goals (would make the study a failure even if it "runs")**
- Unfair comparison: different prompts, sample counts, seeds, or question subsets across
  configurations.
- Headline claims built on noisy correctness labels without measuring label quality.
- Results that can't be regenerated from the released code + configs.
- Cost numbers mixed across different GPUs.

## 3. Constraints

- **Compute:** Kaggle free tier — 1× NVIDIA T4 (16 GB), ~30 GPU-h/week, 12-h session cap,
  ~20 GB output storage. Always select "GPU T4" (never P100) for consistency.
- **Development machine:** Windows 11 laptop, RTX 4050 Laptop (6 GB), Python 3.11. Used for
  writing/debugging code against Qwen2.5-0.5B only; no reported numbers come from it.
- **T4 has no native bf16** → full precision means **fp16**.
- **Budget:** $0.
- Llama-3.2 models are gated on Hugging Face: requires accepting the license and an HF token
  (stored as a Kaggle secret).
- arXiv first-time submission in cs.CL/cs.LG likely needs an **endorsement** — author should
  line one up early.

## 4. Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| **A. Plain Python staged pipeline** (HF transformers + bitsandbytes, one config file, each stage writes to disk) | Detectors are hand-written (~50 lines each), but full control over hidden states and multi-sampling; resumable; same code on laptop & Kaggle | **Chosen** |
| B. Build on lm-eval-harness | Datasets/grading for free, but not built to expose hidden states or multi-sample generation — fights the core of the study | Rejected |
| C. Notebooks only | Fastest start, but 24+ configs become inconsistent and hard to reproduce | Rejected |
| llama.cpp / GGUF | Faster, but hidden states are hard to extract and its quantization isn't what "8-bit/4-bit" means in HF | Rejected |

Compute decision: **all reported experiments on Kaggle T4**; laptop for development only.
(Chosen over laptop+Kaggle hybrid so all timings come from one GPU.)

## 5. Experimental design

### 5.1 Models (Instruct variants, since the task is answering questions)
- **Main family:** Qwen2.5-Instruct — 0.5B, 1.5B, 3B, 7B.
- **Replication family:** Llama-3.2-Instruct — 1B, 3B.

### 5.2 Precisions (bitsandbytes)
- **fp16** (full precision on T4)
- **int8** — `load_in_8bit=True` (LLM.int8)
- **nf4** — `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, double quant on, compute dtype fp16

Total configurations: (4 + 2) models × 3 precisions = **18**.

**Qwen2.5-7B fp16 exception** *(builder's call)*: its fp16 weights (~15.2 GB) don't fit a
single T4 with KV cache and hidden-state capture. This one cell runs on Kaggle's **"GPU T4
x2"** (`device_map="auto"` across both GPUs). Its AUROC/accuracy are reported normally and it
serves as the fp16 baseline for 7B cross-precision probe transfer, but its timings are
**excluded from the cost comparison** (marked as different hardware in the paper). All other
17 cells run on a single T4.

### 5.3 Datasets
- **TriviaQA** (`rc.nocontext`, validation split) — primary.
- **Natural Questions Open** (validation split) — second dataset.
- Draw a fixed random subset of **N = 1,500** questions per dataset (seed fixed, saved to
  disk as a question-ID list), identical across all 18 configs.
- Stretch (only if quota allows): TruthfulQA as an adversarial set.

### 5.4 Generation
- Same prompt for every model: chat template + short instruction ("Answer with a short
  phrase only") + a few fixed few-shot examples. `max_new_tokens = 32`.
- **Main answer:** greedy decoding. This is the answer that is graded and that detectors judge.
- **Samples for semantic entropy:** 5 additional samples, temperature 1.0, fixed seed.
- Saved per question: greedy answer text, per-token log-probs and full-vocab entropies of the
  greedy answer, 5 sampled answers, hidden states (see 5.6), timing per stage.

### 5.5 Grading (labels: hallucinated = 1 / correct = 0)
- Normalize text (lowercase, strip punctuation/articles), then the answer is **correct** if it
  matches or contains any gold alias (TriviaQA aliases / NQ answer list).
- **Label audit:** manually check a stratified random sample of **200** graded answers
  (spread over sizes/precisions); report grader agreement in the paper. If agreement < 90%,
  revisit the grader before trusting results.
- Also report each config's accuracy (= 1 − hallucination rate); smaller/quantized models
  hallucinate more, which changes class balance — AUROC is robust to this, AUPRC is reported
  alongside.

### 5.6 Detectors (each produces a score; higher = more likely hallucinated)
1. **Output confidence (single pass, no training)**
   - Negative mean token log-probability of the greedy answer.
   - Mean token entropy of the greedy answer.
2. **Answer consistency — semantic entropy**
   - Cluster the 5 samples + greedy answer by meaning using bidirectional entailment with an
     NLI model (`microsoft/deberta-large-mnli`); compute entropy over cluster frequencies
     (discrete semantic entropy). NLI model runs in fp16 regardless of the generator's
     precision; its runtime counts toward this detector's cost.
3. **Internal-state probe (single pass + light training)**
   - Hidden state at the last token of the greedy answer, from layers at ~25%, 50%, 75%, and
     100% of depth, saved as fp16.
   - Logistic regression (scikit-learn, L2) per layer; split questions 60/20/20
     train/val/test (same split across configs); pick layer on val, report on test.
   - **Cross-precision transfer** (key quantization analysis): train probe on fp16
     activations, test on int8/nf4 activations of the same model.

### 5.7 Metrics
- **AUROC** (primary), **AUPRC** (secondary), with **95% bootstrap CIs** (1,000 resamples).
  For probes, metrics are on the test split; for training-free detectors, report on the same
  test split (for comparability) and on all N.
- **Cost** — measured in a **separate, controlled timing run** *(builder's call)*, not from
  the main generation logs (whose batch sizes vary per config):
  - Single T4, **batch size 1**, a fixed subset of **100 questions** (first 100 of the TriviaQA
    subset), after 5 warm-up questions, with `torch.cuda.synchronize()` around timed regions.
  - Run for every single-T4 config (17 cells; the 7B fp16 T4×2 cell is excluded).
  - Report two numbers per detector: **(a) extra cost** on top of the shared greedy pass
    (which is needed anyway to produce the answer), and **(b) total cost** including that
    pass. Confidence detectors will have near-zero extra cost; that is a real finding, not an
    artifact.
  - Also record number of generations and peak VRAM.
  - Main runs use whatever batch size fits; their timings are logged but not reported as cost.

### 5.8 Planned outputs (paper figures/tables)
- Heatmap: AUROC by (model size × precision) for each detector.
- Line plot: AUROC vs model size, one line per detector, panels per precision.
- Table: cost per detector vs AUROC (quality-per-cost).
- Table: cross-precision probe transfer.
- Table: base accuracy per config + label-audit agreement.

## 6. Software design (Approach A)

```
hallucination/
  configs/             # one YAML per experiment grid; model, precision, dataset, N, seeds
  src/hallu/
    data.py            # load datasets, fixed subset selection, prompts
    generate.py        # stage 1: greedy + samples + logprobs + hidden states + timings
    grade.py           # stage 2: normalization + alias matching -> labels
    detectors/         # stage 3: confidence.py, semantic_entropy.py, probe.py
    evaluate.py        # stage 4: AUROC/AUPRC + bootstrap CIs, cost tables, figures
  scripts/             # thin CLI entrypoints: run_stage.py --config ... --stage ...
  kaggle/              # notebook that installs package, pulls code, runs a config
  results/<config_id>/ # per-stage outputs (JSONL + .npy/.safetensors activations)
  paper/               # LaTeX, ACL template
  docs/
```

- **Resumable:** stage outputs are written incrementally (per-question JSONL rows /
  sharded arrays); on restart a stage skips question IDs already done. Required for
  Kaggle's 12-h cap and disconnects.
- **Each stage reads only the previous stage's files** → adding/changing a detector never
  requires re-generation.
- **Storage:** activations ≈ N × 4 layers × hidden × 2 bytes (7B: ~45 MB per dataset) — well
  under 20 GB total. Results copied off Kaggle into the repo (or a Kaggle Dataset) after
  each run.
- **Reproducibility:** pinned `requirements.txt`, seeds in config, record library versions
  and GPU name in every result file.
- **Tests:** unit tests for grading normalization, detector math (on toy inputs), and
  resume logic; an end-to-end smoke config (Qwen2.5-0.5B, 20 questions) that runs on the
  laptop in minutes.

## 7. Execution order

1. Build pipeline + smoke test on laptop (0.5B, 20 questions).
2. Label audit on a small pilot run; fix grader if needed.
3. Full Qwen grid on Kaggle (12 configs × 2 datasets).
4. Llama replication (6 configs) on **TriviaQA only** *(builder's call, to save quota)*;
   extend to NQ-Open if GPU quota remains after steps 3–4.
5. Controlled timing run (17 configs × 100 questions, batch size 1).
6. Analysis, figures, paper draft; release code on GitHub; arXiv submission.

## 8. Risks

- **Kaggle quota** → start with N=1,000 if the pilot shows the full grid > ~30 GPU-h; scale up after.
- **fp16 instability** in some models on T4 → detect NaN/inf in logits; note in paper if any.
- **Grader noise** → label audit (5.5); consider adding an LLM-judge check as future work.
- **Very low accuracy for 0.5B** (few correct answers → unstable AUROC) → check class
  counts; report CIs; if < ~100 correct answers, flag cell as low-confidence.

## 9. Left to the builder

- Exact prompt wording and few-shot examples (must be fixed once and shared by all configs).
- Batch sizes per model/precision.
- File formats for activations (.npy vs safetensors) and JSONL schema.
- Plotting style.
- Kaggle notebook mechanics (pulling code via GitHub or Kaggle Dataset).
