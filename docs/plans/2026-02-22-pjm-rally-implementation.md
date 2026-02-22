# PJM Rally Risk Fast Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a fast, benchmark-first PJM rally-risk prototype with parallel worktree/PR execution and reproducible results.

**Architecture:** Build one shared benchmark harness first (data contract, split contract, metrics contract), then run model experimentation and storytelling as independent tracks that read the same artifacts. This keeps merge risk low while maximizing parallel velocity.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, pytest, matplotlib/seaborn (or plotly), optional lightgbm.

---

## Parallel Workstreams
- PR1 (Track A): Benchmark harness and baseline models (blocking foundation)
- PR2 (Track B): Improved model plugin + additional evaluation (depends on PR1 interfaces only)
- PR3 (Track C): Demo notebook/report and executive narrative (depends on PR1 artifacts, optionally PR2)

## Bootstrap (required before worktrees)

### Task 0: Initialize trunk so worktrees can exist

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `docs/plans/2026-02-22-pjm-rally-design.md`
- Create: `docs/plans/2026-02-22-pjm-rally-implementation.md`

**Step 1: Add minimal repository structure and ignores**
- Add Python/cache/data artifact ignores and `.worktrees/` ignore.

**Step 2: Commit bootstrap to `main`**
Run: `git add . && git commit -m "chore: bootstrap benchmark demo plan"`
Expected: First commit exists.

**Step 3: Verify worktree viability**
Run: `git worktree list`
Expected: Main worktree shown; new worktrees can now be created.

---

## PR1: Benchmark Harness (Track A)

### Task 1: Define strict dataset and split contracts

**Files:**
- Create: `src/time_copilot_demo/contracts.py`
- Create: `tests/test_contracts.py`

**Step 1: Write failing tests for schema and split invariants**
Run: `pytest tests/test_contracts.py -q`
Expected: FAIL (module/functions missing).

**Step 2: Implement minimal contracts**
- Dataclass for dataset bundle.
- Time split validator preventing overlap/leakage.

**Step 3: Re-run tests**
Run: `pytest tests/test_contracts.py -q`
Expected: PASS.

**Step 4: Commit**
Run: `git add tests/test_contracts.py src/time_copilot_demo/contracts.py && git commit -m "test: add dataset contract and temporal split guards"`

### Task 2: Build feature pipeline with leak-safe lagging

**Files:**
- Create: `src/time_copilot_demo/features.py`
- Create: `tests/test_features.py`

**Step 1: Write failing tests for lag/rolling behavior**
Run: `pytest tests/test_features.py -q`
Expected: FAIL.

**Step 2: Implement feature builder**
- Lags, rolling mean/std, calendar features.
- Ensure only past data used.

**Step 3: Re-run tests**
Run: `pytest tests/test_features.py -q`
Expected: PASS.

**Step 4: Commit**
Run: `git add tests/test_features.py src/time_copilot_demo/features.py && git commit -m "feat: add leak-safe feature engineering"`

### Task 3: Implement rally labeling and baseline models

**Files:**
- Create: `src/time_copilot_demo/labels.py`
- Create: `src/time_copilot_demo/baselines.py`
- Create: `tests/test_labels.py`
- Create: `tests/test_baselines.py`

**Step 1: Write failing tests for label correctness and baseline behavior**
Run: `pytest tests/test_labels.py tests/test_baselines.py -q`
Expected: FAIL.

**Step 2: Implement rally labeler + naive/logreg baselines**
- Rolling-quantile rally labels.
- Naive classifier and logistic regression baseline.

**Step 3: Re-run tests**
Run: `pytest tests/test_labels.py tests/test_baselines.py -q`
Expected: PASS.

**Step 4: Commit**
Run: `git add tests/test_labels.py tests/test_baselines.py src/time_copilot_demo/labels.py src/time_copilot_demo/baselines.py && git commit -m "feat: add rally labels and benchmark baselines"`

### Task 4: Evaluate and export benchmark artifacts

**Files:**
- Create: `src/time_copilot_demo/evaluate.py`
- Create: `scripts/run_benchmark.py`
- Create: `tests/test_evaluate.py`
- Create: `artifacts/.gitkeep`

**Step 1: Write failing tests for metrics computation**
Run: `pytest tests/test_evaluate.py -q`
Expected: FAIL.

**Step 2: Implement evaluator + CLI runner**
- Output `artifacts/metrics.json` and `artifacts/predictions.parquet`.

**Step 3: Re-run tests + smoke run**
Run: `pytest -q`
Run: `python scripts/run_benchmark.py`
Expected: Tests pass and artifacts generated.

**Step 4: Commit**
Run: `git add . && git commit -m "feat: add benchmark runner and metrics export"`

---

## PR2: Improved Model (Track B)

### Task 5: Add pluggable model registry

**Files:**
- Create: `src/time_copilot_demo/model_registry.py`
- Modify: `scripts/run_benchmark.py`
- Create: `tests/test_model_registry.py`

**Step 1: Write failing tests for model lookup and fit/predict contract**
Run: `pytest tests/test_model_registry.py -q`
Expected: FAIL.

**Step 2: Implement registry and default improved model**
- Add one stronger model (e.g., gradient boosting).

**Step 3: Re-run tests**
Run: `pytest tests/test_model_registry.py -q`
Expected: PASS.

**Step 4: Commit**
Run: `git add tests/test_model_registry.py src/time_copilot_demo/model_registry.py scripts/run_benchmark.py && git commit -m "feat: add model plugin architecture"`

### Task 6: Add comparative benchmark table

**Files:**
- Modify: `src/time_copilot_demo/evaluate.py`
- Create: `tests/test_benchmark_table.py`

**Step 1: Write failing test for baseline vs improved comparison output**
Run: `pytest tests/test_benchmark_table.py -q`
Expected: FAIL.

**Step 2: Implement table generation**
- Add lift columns and confidence interval via bootstrap.

**Step 3: Re-run tests + benchmark**
Run: `pytest -q`
Run: `python scripts/run_benchmark.py --models naive logreg gbdt`
Expected: Comparative table saved in `artifacts/benchmark_table.csv`.

**Step 4: Commit**
Run: `git add tests/test_benchmark_table.py src/time_copilot_demo/evaluate.py artifacts/benchmark_table.csv && git commit -m "feat: add comparative benchmark reporting"`

---

## PR3: Demo Narrative (Track C)

### Task 7: Create colleague-facing notebook/report

**Files:**
- Create: `notebooks/pjm_rally_demo.ipynb`
- Create: `reports/pjm_rally_demo.md`

**Step 1: Build notebook from frozen artifacts**
- Load artifacts, plot risk windows and calibration.

**Step 2: Write narrative with business analog mapping**
- Translate outputs to freight/refinery-style decisions.

**Step 3: Verify notebook executes end-to-end**
Run: `jupyter nbconvert --to notebook --execute notebooks/pjm_rally_demo.ipynb --output /tmp/pjm_rally_demo.executed.ipynb`
Expected: Successful execution.

**Step 4: Commit**
Run: `git add notebooks/pjm_rally_demo.ipynb reports/pjm_rally_demo.md && git commit -m "docs: add colleague-ready demo narrative"`

---

## Worktree / PR Execution Layout

### Branches
- `codex/bootstrap` (initial commit + skeleton)
- `codex/track-a-benchmark-harness`
- `codex/track-b-modeling`
- `codex/track-c-demo-story`

### Worktree commands (after bootstrap commit)
- `git worktree add .worktrees/track-a -b codex/track-a-benchmark-harness`
- `git worktree add .worktrees/track-b -b codex/track-b-modeling`
- `git worktree add .worktrees/track-c -b codex/track-c-demo-story`

### Merge order
1. PR1 (Track A) first.
2. Rebase PR2 and PR3 on updated `main`.
3. Merge PR2, then PR3.

## Definition of Done
- One-command benchmark run from clean checkout.
- Reproducible artifact set in `artifacts/`.
- Benchmark table showing improved model lift vs baseline.
- Colleague-ready report connecting results to freight/refinery analogs.
