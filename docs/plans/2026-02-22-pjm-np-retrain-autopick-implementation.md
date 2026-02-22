# PJM+NP Retrain Benchmarks With Auto-Pick Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build dual-market dual-task benchmark pipelines with retrained strong baselines and automatic champion selection.

**Architecture:** Use one shared feature/split contract for both markets and tasks, then train/evaluate classical and neural models via task-specific registries. Emit full benchmark artifacts and champion picks for each market and task.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, pytest, optional torch (MPS), pyarrow.

---

### Task 1: Add benchmark task contracts and champion rules

**Files:**
- Create: `src/time_copilot_demo/champion.py`
- Create: `tests/test_champion.py`

**Step 1: Write failing tests**
Run: `pytest tests/test_champion.py -q`
Expected: FAIL (`module not found`)

**Step 2: Implement champion rules**
- Forecast champion by min sMAPE (tie-break MAE)
- Rally champion by max PR-AUC (tie-break min Brier)

**Step 3: Run tests**
Run: `pytest tests/test_champion.py -q`
Expected: PASS

**Step 4: Commit**
Run: `git add tests/test_champion.py src/time_copilot_demo/champion.py && git commit -m "feat: add champion selection rules"`

### Task 2: Extend metrics module for forecast benchmark

**Files:**
- Modify: `src/time_copilot_demo/evaluate.py`
- Create: `tests/test_forecast_metrics.py`

**Step 1: Write failing tests**
Run: `pytest tests/test_forecast_metrics.py -q`
Expected: FAIL

**Step 2: Implement MAE/RMSE/sMAPE metrics**

**Step 3: Run tests**
Run: `pytest tests/test_forecast_metrics.py -q`
Expected: PASS

**Step 4: Commit**
Run: `git add tests/test_forecast_metrics.py src/time_copilot_demo/evaluate.py && git commit -m "feat: add forecast benchmark metrics"`

### Task 3: Add model registries for forecast and rally

**Files:**
- Create: `src/time_copilot_demo/model_registry.py`
- Create: `src/time_copilot_demo/dnn.py`
- Create: `tests/test_model_registry.py`

**Step 1: Write failing tests**
Run: `pytest tests/test_model_registry.py -q`
Expected: FAIL

**Step 2: Implement registries**
- Forecast: naive, lear, gbdt_reg, dnn_reg
- Rally: naive, logreg, gbdt_cls, dnn_cls

**Step 3: Run tests**
Run: `pytest tests/test_model_registry.py -q`
Expected: PASS

**Step 4: Commit**
Run: `git add tests/test_model_registry.py src/time_copilot_demo/model_registry.py src/time_copilot_demo/dnn.py && git commit -m "feat: add forecast and rally model registries"`

### Task 4: Build dual-market benchmark runner

**Files:**
- Modify: `src/time_copilot_demo/data.py`
- Modify: `src/time_copilot_demo/pipeline.py`
- Modify: `scripts/run_benchmark.py`
- Create: `tests/test_dual_market_pipeline.py`

**Step 1: Write failing integration test**
Run: `pytest tests/test_dual_market_pipeline.py -q`
Expected: FAIL

**Step 2: Implement runner**
- Run both markets: PJM + NP
- Emit per-market:
  - `forecast_benchmark.csv`
  - `rally_benchmark.csv`
  - `champions.json`

**Step 3: Run tests and smoke benchmark**
Run: `pytest tests -q`
Run: `python scripts/run_benchmark.py --markets PJM,NP --artifacts-dir artifacts/dual_market`
Expected: PASS + artifacts

**Step 4: Commit**
Run: `git add tests/test_dual_market_pipeline.py src/time_copilot_demo/data.py src/time_copilot_demo/pipeline.py scripts/run_benchmark.py && git commit -m "feat: add dual-market dual-task benchmark runner"`

### Task 5: Add final summary report generation

**Files:**
- Create: `src/time_copilot_demo/reporting.py`
- Create: `scripts/generate_report.py`
- Create: `tests/test_reporting.py`

**Step 1: Write failing test**
Run: `pytest tests/test_reporting.py -q`
Expected: FAIL

**Step 2: Implement report generation**
- Combined market comparison tables
- Champion highlights and business interpretation section

**Step 3: Run tests and report command**
Run: `pytest tests -q`
Run: `python scripts/generate_report.py --artifacts-dir artifacts/dual_market --output reports/dual_market_benchmark.md`
Expected: PASS + markdown report

**Step 4: Commit**
Run: `git add tests/test_reporting.py src/time_copilot_demo/reporting.py scripts/generate_report.py reports/dual_market_benchmark.md && git commit -m "docs: add dual-market benchmark report generator"`

### Task 6: Final verification

**Files:**
- Modify: `README.md`

**Step 1: Run full verification commands**
Run: `pytest tests`
Run: `python scripts/run_benchmark.py --markets PJM,NP --artifacts-dir artifacts/dual_market`
Run: `python scripts/generate_report.py --artifacts-dir artifacts/dual_market --output reports/dual_market_benchmark.md`
Expected: all pass and outputs exist

**Step 2: Update README with final commands and outputs**

**Step 3: Commit**
Run: `git add README.md && git commit -m "docs: document dual-market retrain benchmark workflow"`
