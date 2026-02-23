# Benchmark Chart Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate executive and analytical seaborn chart packs from dual-market benchmark artifacts.

**Architecture:** Implement a chart module that loads benchmark CSV/parquet files, composes five deterministic chart functions, and writes outputs to `reports/charts`. Keep CLI orchestration in a small script.

**Tech Stack:** Python, pandas, seaborn, matplotlib, scikit-learn, pytest.

---

### Task 1: Add failing tests for chart generation contract

**Files:**
- Create: `tests/test_charts.py`

**Step 1: Write failing tests**
- Validate chart generation creates 5 expected PNG files and one markdown index.

**Step 2: Run test to verify failure**
Run: `pytest tests/test_charts.py -q`
Expected: FAIL (`module/function not found`)

### Task 2: Implement chart generation module

**Files:**
- Create: `src/time_copilot_demo/charts.py`

**Step 1: Implement artifact loading and chart functions**
- Champion scorecard
- Forecast sMAPE comparison
- Rally PR-AUC comparison
- Forecast MAE vs sMAPE scatter
- Rally calibration curves

**Step 2: Save charts with deterministic filenames**
- `01_champion_scorecard.png`
- `02_forecast_smape.png`
- `03_rally_prauc.png`
- `04_forecast_efficiency.png`
- `05_rally_calibration.png`

**Step 3: Add markdown chart index generator**
- `reports/charts/README.md` listing images.

### Task 3: Add script entrypoint and dependencies

**Files:**
- Create: `scripts/generate_charts.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Step 1: Add CLI**
- Input: `--artifacts-dir`, `--output-dir`
- Output: chart pack files

**Step 2: Add dependencies**
- `seaborn`, `matplotlib`

### Task 4: Verify and commit

**Files:**
- Modify: `reports/charts/*` (generated)

**Step 1: Run tests**
Run: `pytest tests/test_charts.py -q`
Expected: PASS

**Step 2: Generate charts from real artifacts**
Run: `PYTHONPATH=src python scripts/generate_charts.py --artifacts-dir artifacts/dual_market --output-dir reports/charts`
Expected: 5 PNGs + README generated.

**Step 3: Run full verification**
Run: `pytest tests`
Expected: PASS.

**Step 4: Commit**
Run: `git add ... && git commit -m "feat: add executive and analytical benchmark chart pack"`
