# Hallucination Detection Under Shrinking & Quantization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, staged Python pipeline that generates answers from small/quantized
LLMs, grades them, scores three hallucination-detector families, measures their cost, and
produces the paper's tables and figures.

**Architecture:** Plain Python package `hallu` (spec Approach A). Four stages —
`generate → grade → detect → evaluate` — plus a separate `timing` stage. Each stage reads only
the previous stage's files under `results/<config_id>/` and skips question IDs already done.
One grid YAML expands into per-run configs; a single CLI runs any stage for any config.

**Tech Stack:** Python 3.11, PyTorch (Kaggle's preinstalled build; CUDA build on laptop),
transformers, accelerate, bitsandbytes, datasets, scikit-learn, numpy, pandas, matplotlib,
PyYAML, joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-hallucination-detection-design.md`

## Global Constraints

- Python 3.11. Reported numbers come only from Kaggle **"GPU T4"**; laptop (RTX 4050, 6 GB) is for development with Qwen2.5-0.5B only.
- Full precision is **fp16** (T4 has no bf16). Precisions: `fp16`, `int8` (`load_in_8bit=True`), `nf4` (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`, `bnb_4bit_compute_dtype=torch.float16`).
- Models (Instruct): Qwen2.5 0.5B/1.5B/3B/7B; Llama-3.2 1B/3B. Llama replication on TriviaQA only.
- Qwen2.5-7B fp16 runs on "GPU T4 x2" with `device_map="auto"`; excluded from cost tables.
- Datasets: TriviaQA `rc.nocontext` validation; NQ-Open validation. N = 1,500 per dataset, fixed seed, subset saved as an ID list, identical across all configs.
- Generation: same prompt for all models; `max_new_tokens = 32`; greedy main answer; 5 samples at temperature 1.0, fixed seed.
- Labels: hallucinated = 1, correct = 0. Detector scores: higher = more likely hallucinated.
- NLI model `microsoft/deberta-large-mnli`, fp16 always.
- Probe layers at 25/50/75/100% depth; hidden states saved fp16; split 60/20/20 train/val/test, same split across configs.
- Metrics: AUROC (primary), AUPRC, 95% bootstrap CI with 1,000 resamples.
- Cost: batch size 1, first 100 TriviaQA-subset questions, 5 warm-up questions, `torch.cuda.synchronize()` around timed regions, single T4 only.
- Every result directory records library versions and GPU name.
- Budget $0; no paid services.

## Review Focus

1. **An alias that normalizes to the empty string** (e.g. gold alias `"The"`) must be ignored, not match every answer → test in Task 3.
2. **The model emits an empty answer** (EOS or newline first) → row still written, labeled hallucinated, detectors still produce finite scores → tests in Tasks 3 and 4.
3. **Kaggle session dies mid-stage, leaving a truncated last JSONL line** → rerun ignores the broken line, skips done IDs, produces no duplicates → tests in Tasks 1 and 5.
4. **A cell where one class has < 2 examples** (0.5B gets almost nothing right) → AUROC is NaN and flagged, bootstrap skips single-class resamples, evaluate does not crash → test in Task 9.
5. **fp16 produces non-finite logits** → row flagged `nonfinite=true`, its scores are NaN, evaluate drops and counts them → tests in Tasks 4 and 9.

---

## File Structure

```
pyproject.toml            # package metadata, pytest config + markers
requirements.txt          # exact pins (frozen in Task 1), torch excluded
.gitignore                # results/, data/cache/, __pycache__, *.egg-info
configs/
  smoke.yaml              # Qwen2.5-0.5B, fp16, TriviaQA, N=20
  qwen.yaml               # 4 sizes x 3 precisions x 2 datasets
  llama.yaml              # 2 sizes x 3 precisions x TriviaQA
src/hallu/
  config.py               # RunConfig, load_grid, run_dir
  io.py                   # JSONL append/read, done_ids, env_info, write_json
  data.py                 # Question, loaders, select_subset, split_ids, build_messages
  grade.py                # normalize, is_correct, grade_run
  model.py                # load_model, layer_indices, token_stats, truncate_answer_ids
  generate.py             # stage 1
  detectors/confidence.py # neg_mean_logprob, mean_entropy
  detectors/semantic_entropy.py  # cluster_by_entailment, discrete_semantic_entropy, NLIJudge
  detectors/probe.py      # train_select_probe, ProbeResult
  detect.py               # stage 3 orchestration
  evaluate.py             # stage 4: metrics, CIs, tables, figures
  timing.py               # controlled cost run
  audit.py                # label-audit sampling + agreement
scripts/run_stage.py      # CLI entrypoint
scripts/audit.py          # CLI: make / score audit sheet
kaggle/run_on_kaggle.py   # cell-by-cell Kaggle runner
kaggle/README.md          # how to run a config on Kaggle
tests/                    # one test file per module
```

### Shared on-disk formats (all tasks rely on these)

`results/<config_id>/`
- `meta.json` — `{config: <RunConfig as dict>, env: env_info()}`
- `generations.jsonl` — one row per question:
  `{qid, answer, empty_answer, nonfinite, token_logprobs: [float], token_entropies: [float], samples: [str]*5, t_greedy_s, t_samples_s}`
- `acts/shard_00000.npz` — `qids` (str array) and `acts` (fp16 array `[B, n_layers_saved, hidden]`)
- `labels.jsonl` — `{qid, label}` (1 = hallucinated)
- `scores.jsonl` — `{qid, neg_mean_logprob, mean_entropy, semantic_entropy, n_clusters, probe}` (`probe` is null outside the test split)
- `probe.json` — `{layer_val_auroc: {"0.25": float, ...}, chosen_frac: float, chosen_layer_pos: int}`; `probe.joblib` — fitted sklearn pipeline

`results/timing/<config_id>.jsonl`, `results/summary/*.csv`, `results/summary/figures/*.pdf`.
`data/subsets/<dataset>_n<N>_seed<seed>.json` — ordered list of qids.

---

### Task 1: Project skeleton, config grid, JSONL I/O

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`, `src/hallu/__init__.py`, `src/hallu/config.py`, `src/hallu/io.py`, `configs/smoke.yaml`, `configs/qwen.yaml`, `configs/llama.yaml`
- Test: `tests/test_config.py`, `tests/test_io.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) RunConfig: config_id: str, model: str, size_b: float, family: str, precision: str, dataset: str, n_questions: int, subset_seed: int, split_seed: int, sample_seed: int, n_samples: int, temperature: float, max_new_tokens: int, batch_size: int, layer_fracs: tuple[float, ...], results_root: str, timing_excluded: bool`
  - `load_grid(path: str | Path) -> list[RunConfig]`
  - `run_dir(cfg: RunConfig) -> Path` (`Path(results_root) / config_id`, created)
  - `append_jsonl(path, rows: Iterable[dict]) -> None`, `read_jsonl(path) -> list[dict]`, `done_ids(path, key="qid") -> set[str]`, `write_json(path, obj)`, `env_info() -> dict`

Grid YAML format (exact):

```yaml
family: qwen2.5
models:
  - {id: Qwen/Qwen2.5-0.5B-Instruct, size_b: 0.5, batch_size: 32}
  - {id: Qwen/Qwen2.5-1.5B-Instruct, size_b: 1.5, batch_size: 32}
  - {id: Qwen/Qwen2.5-3B-Instruct,   size_b: 3.0, batch_size: 16}
  - {id: Qwen/Qwen2.5-7B-Instruct,   size_b: 7.0, batch_size: 8}
precisions: [fp16, int8, nf4]
datasets: [triviaqa, nq_open]
timing_exclude: [qwen2.5-7b_fp16]      # prefix match on config_id
defaults:
  n_questions: 1500
  subset_seed: 0
  split_seed: 0
  sample_seed: 1
  n_samples: 5
  temperature: 1.0
  max_new_tokens: 32
  layer_fracs: [0.25, 0.5, 0.75, 1.0]
  results_root: results
```

`config_id = f"{short}_{precision}_{dataset}"` where `short` = part after `/`, lowercased, with `-instruct` removed (e.g. `qwen2.5-0.5b_fp16_triviaqa`). `llama.yaml`: family `llama3.2`, models `meta-llama/Llama-3.2-1B-Instruct` (1.0) and `meta-llama/Llama-3.2-3B-Instruct` (3.0), datasets `[triviaqa]`, empty `timing_exclude`. `smoke.yaml`: 0.5B only, `[fp16]`, `[triviaqa]`, `n_questions: 20`, `batch_size: 4`, `results_root: results_smoke`.

- [ ] **Step 1: `git init`, create `.gitignore`, `pyproject.toml`**

`pyproject.toml`: setuptools, `src` layout, package `hallu`, `requires-python = ">=3.11"`; `[tool.pytest.ini_options]` with `testpaths = ["tests"]`, `markers = ["gpu: needs CUDA + model download", "network: needs internet"]`, `addopts = "-m 'not gpu and not network'"`.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_config.py
def test_qwen_grid_expands_to_24(tmp_path):
    cfgs = load_grid("configs/qwen.yaml")
    assert len(cfgs) == 24
    ids = {c.config_id for c in cfgs}
    assert "qwen2.5-0.5b_fp16_triviaqa" in ids and "qwen2.5-7b_nf4_nq_open" in ids

def test_timing_exclusion_only_7b_fp16():
    cfgs = load_grid("configs/qwen.yaml")
    excluded = {c.config_id for c in cfgs if c.timing_excluded}
    assert excluded == {"qwen2.5-7b_fp16_triviaqa", "qwen2.5-7b_fp16_nq_open"}

def test_llama_grid_is_triviaqa_only():
    cfgs = load_grid("configs/llama.yaml")
    assert len(cfgs) == 6 and {c.dataset for c in cfgs} == {"triviaqa"}

def test_per_model_batch_size_and_defaults():
    c = next(c for c in load_grid("configs/qwen.yaml") if c.size_b == 7.0)
    assert c.batch_size == 8 and c.n_questions == 1500 and c.layer_fracs == (0.25, 0.5, 0.75, 1.0)
```

```python
# tests/test_io.py
def test_append_then_read_roundtrip(tmp_path): ...   # two appends -> 3 rows in order
def test_truncated_last_line_is_ignored(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, [{"qid": "a"}, {"qid": "b"}])
    with open(p, "a") as f: f.write('{"qid": "c", "ans')   # simulated crash
    assert [r["qid"] for r in read_jsonl(p)] == ["a", "b"]
    assert done_ids(p) == {"a", "b"}
def test_done_ids_missing_file_is_empty(tmp_path):
    assert done_ids(tmp_path / "nope.jsonl") == set()
def test_env_info_has_versions():
    e = env_info(); assert {"python", "torch", "transformers", "gpu_name"} <= e.keys()
```

- [ ] **Step 3: Run `python -m pytest tests/test_config.py tests/test_io.py -v`** — Expected: FAIL (import errors).

- [ ] **Step 4: Implement `config.py`, `io.py`, the three YAMLs**

`read_jsonl` skips a final line that fails `json.loads` (only the last line; a bad line in the middle raises). `append_jsonl` opens in append mode, writes one `json.dumps` per line, then `flush()` + `os.fsync()`. `env_info` reports `gpu_name` as `torch.cuda.get_device_name(0)` or `"cpu"`, plus `gpu_count`, and `bitsandbytes`/`datasets`/`sklearn` versions (`None` if not importable).

- [ ] **Step 5: Run tests** — Expected: PASS.

- [ ] **Step 6: Create venv, install, freeze pins**

`pip install -e . transformers accelerate bitsandbytes datasets scikit-learn numpy pandas matplotlib pyyaml joblib pytest` (torch installed separately from pytorch.org, CUDA build). Write `pip freeze` output for those packages only (not torch) into `requirements.txt` with `==` pins.

- [ ] **Step 7: Commit** — `git add -A && git commit -m "feat: project skeleton, grid configs, jsonl io"`

---

### Task 2: Datasets, fixed subset, splits, prompt

**Files:**
- Create: `src/hallu/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `RunConfig`, `write_json`
- Produces:
  - `@dataclass(frozen=True) Question: qid: str, question: str, aliases: tuple[str, ...]`
  - `select_subset(qids: Sequence[str], n: int, seed: int) -> list[str]`
  - `split_ids(qids: Sequence[str], seed: int) -> dict[str, list[str]]` (keys `train`, `val`, `test`)
  - `load_questions(cfg: RunConfig, subset_dir="data/subsets") -> list[Question]` (in subset order)
  - `build_messages(question: str) -> list[dict]`
  - `FEW_SHOT: list[tuple[str, str]]`, `SYSTEM_PROMPT: str`

- [ ] **Step 1: Write failing tests**

```python
def test_select_subset_deterministic_and_order_independent():
    ids = [f"q{i}" for i in range(5000)]
    a = select_subset(ids, 1500, seed=0)
    b = select_subset(list(reversed(ids)), 1500, seed=0)
    assert a == b and len(set(a)) == 1500

def test_split_is_60_20_20_disjoint_and_stable():
    ids = [f"q{i}" for i in range(1500)]
    s = split_ids(ids, seed=0)
    assert [len(s[k]) for k in ("train", "val", "test")] == [900, 300, 300]
    assert set(s["train"]).isdisjoint(s["test"]) and s == split_ids(ids, seed=0)

def test_messages_contain_system_fewshot_and_question():
    m = build_messages("Who painted the Mona Lisa?")
    assert m[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert len(m) == 1 + 2 * len(FEW_SHOT) + 1
    assert m[-1] == {"role": "user", "content": "Question: Who painted the Mona Lisa?"}

@pytest.mark.network
def test_load_triviaqa_subset_saved_and_reused(tmp_path): ...  # n=10; file exists; second call same qids
```

- [ ] **Step 2: Run `python -m pytest tests/test_data.py -v`** — Expected: FAIL.

- [ ] **Step 3: Implement `data.py`**

- `select_subset`: `random.Random(seed).sample(sorted(qids), n)`.
- `split_ids`: shuffle sorted ids with `random.Random(seed)`, cut at 60%/80%.
- `SYSTEM_PROMPT = "Answer the question with a short phrase only."`; `FEW_SHOT = [("What is the capital of France?", "Paris"), ("Who wrote the novel 1984?", "George Orwell"), ("How many legs does a spider have?", "Eight")]`; every user turn is `f"Question: {q}"`, assistant turn is the bare answer. Frozen from here on: changing them invalidates every run.
- TriviaQA: `load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")`, `qid = question_id`, `aliases = aliases + [value]` deduplicated. NQ-Open: `load_dataset("google-research-datasets/nq_open", split="validation")`, `qid = f"nq_{row_index}"`, `aliases = answer`.
- `load_questions` reads `data/subsets/{dataset}_n{n}_seed{subset_seed}.json` if present, else selects and writes it.

- [ ] **Step 4: Run tests (also `-m network` once)** — Expected: PASS.

- [ ] **Step 5: Commit** — `feat: dataset loading, fixed subsets, splits, prompt`

---

### Task 3: Grading

**Files:**
- Create: `src/hallu/grade.py`
- Test: `tests/test_grade.py`

**Interfaces:**
- Consumes: `read_jsonl`, `append_jsonl`, `done_ids`, `load_questions`, `run_dir`
- Produces: `normalize(text: str) -> str`, `is_correct(answer: str, aliases: Sequence[str]) -> bool`, `grade_run(cfg: RunConfig) -> Path` (writes `labels.jsonl`)

- [ ] **Step 1: Write failing tests**

```python
def test_normalize():
    assert normalize("  The Beatles!! ") == "beatles"
    assert normalize("A  Tale of Two Cities.") == "tale of two cities"

@pytest.mark.parametrize("ans,aliases,ok", [
    ("Paris", ["paris"], True),
    ("It is Paris, France", ["Paris"], True),     # containment
    ("1999", ["1"], False),                        # token boundary, not substring
    ("George Orwell", ["Orwell"], True),
    ("London", ["Paris"], False),
    ("", ["Paris"], False),                        # empty answer -> hallucinated
    ("anything", ["The", "Paris"], False),         # alias normalizing to "" ignored
])
def test_is_correct(ans, aliases, ok): assert is_correct(ans, aliases) is ok

def test_grade_run_labels_and_resumes(tmp_path): ...
    # fake cfg + questions monkeypatched; generations.jsonl with 3 rows -> labels {0,1,1};
    # second call appends nothing
```

- [ ] **Step 2: Run** — Expected: FAIL.

- [ ] **Step 3: Implement** — `normalize`: lowercase, replace punctuation with space, drop tokens `a/an/the`, collapse whitespace (SQuAD-style). `is_correct`: normalized alias tokens appear as a contiguous token subsequence of the normalized answer; skip aliases that normalize to `""`. `label = 0 if correct else 1`.

- [ ] **Step 4: Run** — Expected: PASS.

- [ ] **Step 5: Commit** — `feat: answer grading with alias matching`

---

### Task 4: Model loading and per-token statistics

**Files:**
- Create: `src/hallu/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces:
  - `load_model(model_id: str, precision: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]` — tokenizer `padding_side="left"`, `pad_token = eos_token` if missing; fp16 uses `torch_dtype=torch.float16`; all use `device_map="auto"`; `model.eval()`
  - `layer_indices(n_layers: int, fracs: Sequence[float]) -> list[int]` — indices into `outputs.hidden_states` (index 0 = embeddings)
  - `token_stats(logits: Tensor[k, V], token_ids: Tensor[k]) -> tuple[list[float], list[float], bool]` → `(logprobs, entropies, nonfinite)`
  - `truncate_answer_ids(gen_ids: list[int], tokenizer) -> list[int]` — cut at first EOS / pad, or at first token whose decoded text contains `"\n"`

- [ ] **Step 1: Write failing tests**

```python
def test_layer_indices():
    assert layer_indices(24, [0.25, 0.5, 0.75, 1.0]) == [6, 12, 18, 24]
    assert layer_indices(2, [0.25]) == [1]          # never the embedding layer

def test_token_stats_uniform_and_peaked():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]])
    lp, ent, bad = token_stats(logits, torch.tensor([1, 0]))
    assert lp[0] == pytest.approx(math.log(0.25)) and ent[0] == pytest.approx(math.log(4))
    assert lp[1] == pytest.approx(0.0, abs=1e-6) and ent[1] == pytest.approx(0.0, abs=1e-6)
    assert bad is False

def test_token_stats_flags_nonfinite():
    _, _, bad = token_stats(torch.tensor([[float("nan"), 0.0]]), torch.tensor([1]))
    assert bad is True

def test_truncate_answer_ids(fake_tokenizer):   # fixture: id 0=eos, 5 decodes "\n"
    assert truncate_answer_ids([3, 4, 0, 7], fake_tokenizer) == [3, 4]
    assert truncate_answer_ids([3, 5, 4], fake_tokenizer) == [3]
    assert truncate_answer_ids([0, 3], fake_tokenizer) == []
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** — `token_stats` in float32: `log_softmax`, gather, entropy `-(p*logp).sum(-1)`; `nonfinite = not torch.isfinite(logits).all()`. `layer_indices`: `max(1, round(f * n_layers))`. Quantization configs exactly as Global Constraints.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: model loading and token statistics`

---

### Task 5: Generation stage (resumable)

**Files:**
- Create: `src/hallu/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 4
- Produces:
  - `generate_run(cfg: RunConfig, limit: int | None = None) -> Path` — writes `meta.json`, `generations.jsonl`, `acts/shard_*.npz`
  - `greedy_pass(model, tok, questions: list[Question], cfg) -> list[GreedyResult]` and `sample_pass(model, tok, q: Question, cfg) -> list[str]` (reused by Task 10)
  - `@dataclass GreedyResult: qid, answer, empty_answer, nonfinite, token_logprobs, token_entropies, acts: np.ndarray[n_layers_saved, hidden] (fp16)`
  - `sample_seed_for(qid: str, base: int) -> int` = `base * 1_000_003 + zlib.crc32(qid.encode())` mod 2**31
  - `load_acts(run_dir: Path) -> tuple[list[str], np.ndarray]` — concatenates shards, keeps last occurrence of duplicate qids

Algorithm for `greedy_pass` (per batch, `torch.no_grad()`):
1. `apply_chat_template(build_messages(q), add_generation_prompt=True, tokenize=False)`; tokenize batch with left padding.
2. `model.generate(do_sample=False, max_new_tokens=cfg.max_new_tokens)`.
3. Per row: `ans_ids = truncate_answer_ids(new_ids)`; if empty, set `empty_answer=True` and score `new_ids[:1]` instead (so stats always have ≥1 token).
4. Scoring forward pass on `prompt_ids + scored_ids` (left-padded batch, `output_hidden_states=True`): logits at positions `[-k-1:-1]` → `token_stats`; hidden states at position `-1` for `layer_indices(...)` → fp16 numpy.
5. `answer = tok.decode(ans_ids, skip_special_tokens=True).strip()`.

`sample_pass`: `torch.manual_seed(sample_seed_for(qid, cfg.sample_seed))`, one `generate` call with `do_sample=True, temperature=cfg.temperature, top_p=1.0, top_k=0, num_return_sequences=cfg.n_samples`, each truncated with `truncate_answer_ids` and decoded.

Ordering for resumability: per batch, save the `.npz` shard first, then append the JSONL rows. Shard file name uses the next free index. `t_greedy_s` = batch greedy time / batch size; `t_samples_s` per question.

- [ ] **Step 1: Write failing tests**

```python
def test_sample_seed_is_stable(): assert sample_seed_for("q1", 1) == sample_seed_for("q1", 1) != sample_seed_for("q2", 1)

def test_load_acts_dedupes_keeping_last(tmp_path): ...  # two shards sharing qid "b"; returns 3 qids, "b" from shard 2

@pytest.mark.gpu
def test_generate_resume_no_duplicates(tmp_path):
    cfg = replace(load_grid("configs/smoke.yaml")[0], results_root=str(tmp_path))
    generate_run(cfg, limit=3); generate_run(cfg, limit=6)
    rows = read_jsonl(run_dir(cfg) / "generations.jsonl")
    assert len(rows) == 6 and len({r["qid"] for r in rows}) == 6
    assert all(len(r["samples"]) == 5 and len(r["token_logprobs"]) >= 1 for r in rows)
    qids, acts = load_acts(run_dir(cfg))
    assert acts.shape[:2] == (6, 4) and acts.dtype == np.float16
    assert "gpu_name" in json.load(open(run_dir(cfg) / "meta.json"))["env"]
```

- [ ] **Step 2: Run `python -m pytest tests/test_generate.py -v`** — Expected: FAIL.
- [ ] **Step 3: Implement `generate.py`.**
- [ ] **Step 4: Run, then on the laptop `python -m pytest tests/test_generate.py -m gpu -v`** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: resumable generation stage with hidden-state capture`

---

### Task 6: Confidence and semantic-entropy detectors

**Files:**
- Create: `src/hallu/detectors/__init__.py`, `src/hallu/detectors/confidence.py`, `src/hallu/detectors/semantic_entropy.py`
- Test: `tests/test_confidence.py`, `tests/test_semantic_entropy.py`

**Interfaces:**
- Produces:
  - `neg_mean_logprob(logprobs: Sequence[float]) -> float`, `mean_entropy(entropies: Sequence[float]) -> float`
  - `cluster_by_entailment(answers: Sequence[str], entails: Callable[[str, str], bool]) -> list[int]`
  - `discrete_semantic_entropy(cluster_ids: Sequence[int]) -> float` (natural log)
  - `class NLIJudge(model_name="microsoft/deberta-large-mnli", device="cuda")` with `entails(question: str, a: str, b: str) -> bool` and `semantic_entropy(question: str, answers: Sequence[str]) -> tuple[float, int]` → `(entropy, n_clusters)`

Clustering algorithm (fixed): iterate answers in order (greedy answer first, then 5 samples); compare against the first member of each existing cluster; join the first cluster where `normalize(a) == normalize(rep)` OR (`entails(a, rep)` and `entails(rep, a)`); otherwise open a new cluster. NLI input: premise `f"{question} {a}"`, hypothesis `f"{question} {b}"`; entails iff argmax label is the one whose `id2label` value upper-cases to `"ENTAILMENT"`. Results cached per `(a, b)` within one question.

- [ ] **Step 1: Write failing tests**

```python
def test_confidence_scores():
    assert neg_mean_logprob([-1.0, -3.0]) == 2.0
    assert mean_entropy([0.5, 1.5]) == 1.0

def test_semantic_entropy_values():
    assert discrete_semantic_entropy([0] * 6) == 0.0
    assert discrete_semantic_entropy([0, 1, 2, 3, 4, 5]) == pytest.approx(math.log(6))
    assert discrete_semantic_entropy([0, 0, 0, 1, 1, 1]) == pytest.approx(math.log(2))

def test_clustering_requires_bidirectional_entailment():
    one_way = lambda a, b: (a, b) == ("Paris, France", "Paris")
    assert cluster_by_entailment(["Paris", "Paris, France"], one_way) == [0, 1]
    both = lambda a, b: {a, b} == {"Paris", "Paris, France"}
    assert cluster_by_entailment(["Paris", "Paris, France", "Rome"], both) == [0, 0, 1]

def test_normalized_duplicates_skip_nli():
    never = lambda a, b: pytest.fail("NLI called")
    assert cluster_by_entailment(["The Beatles", "beatles!"], never) == [0, 0]

@pytest.mark.gpu
def test_nli_judge_smoke():
    j = NLIJudge()
    assert j.semantic_entropy("Capital of France?", ["Paris", "Paris", "Rome"])[1] == 2
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** (NLIJudge loads in fp16, `eval()`).
- [ ] **Step 4: Run (plus `-m gpu` on laptop)** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: confidence and semantic entropy detectors`

---

### Task 7: Internal-state probe

**Files:**
- Create: `src/hallu/detectors/probe.py`
- Test: `tests/test_probe.py`

**Interfaces:**
- Produces:
  - `make_probe() -> sklearn.pipeline.Pipeline` = `StandardScaler()` → `LogisticRegression(penalty="l2", C=1.0, max_iter=5000)`
  - `@dataclass ProbeResult: layer_val_auroc: dict[float, float], chosen_frac: float, chosen_layer_pos: int, model: Pipeline | None, test_scores: np.ndarray | None`
  - `train_select_probe(acts: np.ndarray[N, L, H], labels: np.ndarray[N], split_idx: dict[str, np.ndarray], layer_fracs: Sequence[float]) -> ProbeResult` — fit one probe per layer on train, pick max val AUROC (ties → deeper layer), refit nothing, score test with chosen probe (`predict_proba[:, 1]`)

- [ ] **Step 1: Write failing tests**

```python
def test_probe_picks_informative_layer():
    rng = np.random.default_rng(0); N, H = 600, 16
    y = rng.integers(0, 2, N)
    acts = rng.normal(size=(N, 4, H)).astype(np.float16)
    acts[:, 2, 0] += 3 * y                      # only layer pos 2 carries signal
    idx = np.arange(N); split = {"train": idx[:360], "val": idx[360:480], "test": idx[480:]}
    r = train_select_probe(acts, y, split, [0.25, 0.5, 0.75, 1.0])
    assert r.chosen_layer_pos == 2 and r.chosen_frac == 0.75
    assert r.layer_val_auroc[0.75] > 0.9 and r.test_scores.shape == (120,)

def test_single_class_train_returns_nan_result(): ...
    # all labels 0 -> model None, test_scores None, every layer_val_auroc is nan; no exception
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** (cast acts to float32 before fitting).
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: layer-selected logistic regression probe`

---

### Task 8: Detect stage

**Files:**
- Create: `src/hallu/detect.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `read_jsonl`, `load_acts`, `split_ids`, `load_questions`, Task 6, Task 7
- Produces: `detect_run(cfg: RunConfig, nli: NLIJudge | None = None) -> Path` — writes `scores.jsonl`, `probe.json`, `probe.joblib`; `split_index(cfg, qids: list[str]) -> dict[str, np.ndarray]`

Behavior: semantic entropy is computed per question and appended incrementally (resumable on `done_ids(scores.jsonl)`). Confidence scores computed alongside. Rows with `nonfinite=true` get `neg_mean_logprob`/`mean_entropy` = NaN (written as JSON `null`). The probe step runs once all rows are scored: loads acts + labels in subset order, trains, then rewrites `scores.jsonl` atomically (write `.tmp`, `os.replace`) with `probe` filled for test-split qids. If `probe.json` exists, skip probe training.

- [ ] **Step 1: Write failing test**

```python
def test_detect_run_on_fake_results(tmp_path):
    # Build a fake run dir: 50 generations rows (one with nonfinite=true), labels, acts shards
    # with signal; stub NLIJudge whose semantic_entropy returns (0.0, 1).
    detect_run(cfg, nli=StubJudge())
    rows = {r["qid"]: r for r in read_jsonl(run_dir(cfg) / "scores.jsonl")}
    assert len(rows) == 50
    assert rows[nonfinite_qid]["neg_mean_logprob"] is None
    test_q = set(split_ids(all_qids, cfg.split_seed)["test"])
    assert all((rows[q]["probe"] is not None) == (q in test_q) for q in rows)
    assert (run_dir(cfg) / "probe.joblib").exists()
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: detect stage producing per-question scores`

---

### Task 9: Evaluation — metrics, CIs, tables, figures

**Files:**
- Create: `src/hallu/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: run files from Tasks 3, 5, 8; `results/timing/*.jsonl` from Task 10 (optional — cost table skipped if absent)
- Produces:
  - `metric_ci(labels, scores, metric: Literal["auroc", "auprc"], n_boot=1000, seed=0) -> tuple[float, float, float]` → `(point, lo, hi)`
  - `evaluate_grid(cfgs: list[RunConfig], out_dir="results/summary") -> None` writing:
    - `metrics.csv`: `config_id, family, size_b, precision, dataset, detector, split (test|all), n, n_pos, n_neg, n_dropped_nonfinite, auroc, auroc_lo, auroc_hi, auprc, auprc_lo, auprc_hi, low_confidence`
    - `accuracy.csv`: `config_id, family, size_b, precision, dataset, n, accuracy, n_correct`
    - `transfer.csv`: `family, size_b, dataset, test_precision, chosen_frac, auroc, auroc_lo, auroc_hi` (probe from fp16 `probe.joblib` applied to the int8/nf4 run's test-split acts at the same layer position, scored against that run's own labels)
    - `cost.csv` (if timing exists): `config_id, detector, extra_s_mean, extra_s_std, total_s_mean, total_s_std, n_generations, peak_vram_gb`, joined with test-split AUROC
    - `figures/heatmap_<detector>_<dataset>_<family>.pdf` (rows = size, cols = precision, AUROC annotated) and `figures/auroc_vs_size_<dataset>_<family>.pdf` (one line per detector, one panel per precision, CI bands)

Rules: AUROC/AUPRC via `sklearn.metrics.roc_auc_score` / `average_precision_score`. Rows with NaN score are dropped and counted. If `n_pos < 2` or `n_neg < 2` → all metric fields NaN. Bootstrap: resample rows with replacement, skip resamples with a single class, CI = 2.5/97.5 percentiles of the valid ones. `low_confidence = n_correct_in_all < 100`. Probe has only `split=test`.

- [ ] **Step 1: Write failing tests**

```python
def test_metric_ci_perfect_and_random():
    y = np.array([0, 1] * 200); s = y.astype(float)
    p, lo, hi = metric_ci(y, s, "auroc"); assert p == 1.0 and lo == 1.0
    p, lo, hi = metric_ci(y, np.random.default_rng(0).random(400), "auroc")
    assert lo < 0.5 < hi

def test_metric_ci_single_class_is_nan():
    assert all(math.isnan(v) for v in metric_ci(np.array([1, 1, 1, 0]), np.arange(4.0), "auroc"))

def test_evaluate_grid_on_fake_runs(tmp_path):
    # two fake configs (fp16, nf4) of one model with labels/scores/probe/acts;
    # one score row is null
    evaluate_grid(cfgs, out_dir=tmp_path / "summary")
    m = pd.read_csv(tmp_path / "summary/metrics.csv")
    assert set(m.detector) == {"neg_mean_logprob", "mean_entropy", "semantic_entropy", "probe"}
    assert m.n_dropped_nonfinite.max() == 1
    assert len(pd.read_csv(tmp_path / "summary/transfer.csv")) == 1
    assert list((tmp_path / "summary/figures").glob("heatmap_*.pdf"))
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** (matplotlib `Agg` backend).
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: evaluation with bootstrap CIs, tables and figures`

---

### Task 10: Controlled timing stage

**Files:**
- Create: `src/hallu/timing.py`
- Test: `tests/test_timing.py`

**Interfaces:**
- Consumes: `greedy_pass`, `sample_pass` (Task 5), `NLIJudge`, confidence fns, `probe.joblib` + `probe.json` from the config's own main run (Task 8), `load_questions`
- Produces:
  - `cuda_timer()` context manager yielding an object with `.seconds`; calls `torch.cuda.synchronize()` on enter and exit when CUDA is available
  - `timing_run(cfg: RunConfig, n_warmup=5, n_timed=100) -> Path` writing `results/timing/<config_id>.jsonl`, one row per timed question: `{qid, t_greedy, t_conf, t_samples, t_nli, t_probe, peak_vram_gb}` plus a final summary row `{"summary": true, gpu_name, gpu_count}`

Rules: refuses (raises `RuntimeError`) if `cfg.timing_excluded` or `torch.cuda.device_count() != 1` or `cfg.dataset != "triviaqa"`. Questions: the timed set is positions 0–99 of the saved TriviaQA subset (the spec's "first 100"); warm-up uses positions 100–104 (untimed), so warm-up never overlaps timed questions. Batch size 1 for everything. `peak_vram_gb` from `torch.cuda.max_memory_allocated()` (reset after warm-up). Evaluate (Task 9) derives: extra cost — conf `t_conf`, SE `t_samples + t_nli`, probe `t_probe`; total = `t_greedy + extra`; `n_generations` = 1 / 6 / 1.

- [ ] **Step 1: Write failing tests**

```python
def test_cuda_timer_measures_cpu_time():
    with cuda_timer() as t: time.sleep(0.05)
    assert 0.04 < t.seconds < 0.5

def test_timing_refuses_excluded_config():
    cfg = next(c for c in load_grid("configs/qwen.yaml") if c.timing_excluded)
    with pytest.raises(RuntimeError): timing_run(cfg)

@pytest.mark.gpu
def test_timing_smoke(tmp_path): ...  # smoke cfg after generate+grade+detect, n_warmup=1, n_timed=3 -> 3 rows + summary
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run (plus `-m gpu`)** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: controlled batch-size-1 timing stage`

---

### Task 11: Label audit tooling

**Files:**
- Create: `src/hallu/audit.py`, `scripts/audit.py`
- Test: `tests/test_audit.py`

**Interfaces:**
- Produces:
  - `stratified_sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]` — strata = `(config_id, label)`; round-robin across sorted strata, random order within each stratum, until `n` rows or all exhausted
  - `make_audit_sheet(cfgs, out_csv, n=200, seed=0)` — CSV columns `config_id, qid, question, answer, aliases, auto_label, human_label` (`human_label` blank; aliases joined with ` | `)
  - `score_audit(csv_path) -> dict` → `{n, agreement, false_hallucination, false_correct}`; prints a warning line when `agreement < 0.90`
  - CLI: `python scripts/audit.py make --grid configs/qwen.yaml --out audit.csv` / `python scripts/audit.py score audit.csv`

- [ ] **Step 1: Write failing tests**

```python
def test_stratified_sample_balances_strata():
    rows = [{"config_id": c, "label": l, "qid": f"{c}{l}{i}"} for c in "ABCD" for l in (0, 1) for i in range(100)]
    s = stratified_sample(rows, 200, seed=0)
    assert len(s) == 200 and Counter((r["config_id"], r["label"]) for r in s) == {k: 25 for k in product("ABCD", (0, 1))}

def test_stratified_sample_small_strata_exhausted_first(): ...  # one stratum has 3 rows -> all 3 included, total still n

def test_score_audit(tmp_path):
    # 10 rows, 9 agree, 1 auto=1/human=0
    assert score_audit(p) == {"n": 10, "agreement": 0.9, "false_hallucination": 1, "false_correct": 0}
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: label audit sampling and agreement scoring`

---

### Task 12: CLI and laptop end-to-end smoke run

**Files:**
- Create: `scripts/run_stage.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_grid`, `generate_run`, `grade_run`, `detect_run`, `timing_run`, `evaluate_grid`
- Produces CLI: `python scripts/run_stage.py --grid PATH --stage {generate,grade,detect,timing,evaluate,all} [--config-id ID ...] [--limit N]`
  - No `--config-id` → every config in the grid. `all` = generate → grade → detect for each config, then evaluate over the selected configs. `timing` is never part of `all`. `evaluate` accepts multiple `--grid` flags.
  - Unknown `--config-id` → exit code 2 with the list of valid IDs.

- [ ] **Step 1: Write failing test**

```python
def test_unknown_config_id_exits_2(capsys):
    with pytest.raises(SystemExit) as e: main(["--grid", "configs/smoke.yaml", "--stage", "grade", "--config-id", "nope"])
    assert e.value.code == 2 and "qwen2.5-0.5b_fp16_triviaqa" in capsys.readouterr().err
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement with `argparse`; `main(argv=None)`.**
- [ ] **Step 4: Run full unit suite `python -m pytest -v`** — Expected: all PASS.
- [ ] **Step 5: Laptop smoke run**

Run: `python scripts/run_stage.py --grid configs/smoke.yaml --stage all`
Expected: `results_smoke/qwen2.5-0.5b_fp16_triviaqa/` holds 20 rows in each of `generations.jsonl`, `labels.jsonl`, `scores.jsonl`; `results_smoke/summary/metrics.csv` exists (most metrics NaN or low-confidence at N=20 — expected). Then kill the run halfway through a rerun after deleting `results_smoke/` and confirm restart finishes with 20 unique qids. Spot-read 10 answers vs labels by eye.

- [ ] **Step 6: Run GPU tests on laptop** — `python -m pytest -m gpu -v` — Expected: PASS.
- [ ] **Step 7: Commit** — `feat: CLI and end-to-end smoke run`

---

### Task 13: Kaggle runner

**Files:**
- Create: `kaggle/run_on_kaggle.py`, `kaggle/README.md`

**Interfaces:**
- Consumes: the CLI from Task 12; a public GitHub repo of this project; Kaggle secret `HF_TOKEN`

`run_on_kaggle.py` is written as `# %%` cells to paste into a Kaggle notebook: (1) `nvidia-smi` and assert GPU name contains `T4`; (2) `git clone` repo into `/kaggle/working/hallucination`, `pip install -r requirements.txt -e .` (keep Kaggle's torch); (3) `huggingface_hub.login(UserSecretsClient().get_secret("HF_TOKEN"))`; (4) set `GRID`, `CONFIG_IDS`, `STAGES` variables and call the CLI with `results_root` under `/kaggle/working/results` (the only persisted path); (5) zip `results/` for download. If a previous output exists as an attached Kaggle Dataset, copy it into `results/` first so stages resume.

`README.md`: accelerator choice ("GPU T4", or "GPU T4 x2" only for `qwen2.5-7b_fp16_*`), how to add the secret, how to attach previous output to resume, how to copy results back into the repo.

- [ ] **Step 1: Write both files.**
- [ ] **Step 2: Push repo to GitHub; run on Kaggle with `--grid configs/smoke.yaml --stage all`** — Expected: same files as Task 12 Step 5, and `meta.json` → `env.gpu_name` contains `T4`.
- [ ] **Step 3: Commit** — `feat: Kaggle runner and instructions`

---

## Runbook after the build (spec §7 steps 2–6; operational, no new code)

1. **Pilot + audit:** run `qwen.yaml` configs `qwen2.5-0.5b_fp16_triviaqa` and `qwen2.5-3b_nf4_triviaqa` with `--stage all`. Use wall time × 24 configs (+6 Llama) to estimate the quota. If over about 30 GPU-h, set `n_questions: 1000` in both grids **before** any full run. Then `scripts/audit.py make` with 200 rows over the pilot configs, label them by hand, `score`. If agreement < 0.90, fix `grade.py` (add tests for each failure) and regrade.
2. **Full Qwen grid** on Kaggle across sessions (resume via the attached Dataset).
3. **Llama grid** (`llama.yaml`, TriviaQA). Add NQ-Open to `llama.yaml` only if quota remains.
4. **Timing:** `--stage timing` for every non-excluded TriviaQA config, all on "GPU T4".
5. **Evaluate:** `python scripts/run_stage.py --grid configs/qwen.yaml --grid configs/llama.yaml --stage evaluate`. Then write the paper (ACL LaTeX in `paper/`), release the code, and submit to arXiv.
