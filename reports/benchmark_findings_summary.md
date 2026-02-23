# Time Copilot Prototype: Findings Summary

## 1) What was evaluated
This prototype evaluates whether a Time Copilot style workflow can reliably pick strong forecasting models on public data.

- Markets: `PJM` and `NP` (Nord Pool)
- Tasks:
  - **Point Forecasting**: predict next-hour price level
  - **Rally Risk Forecasting**: estimate probability of a price rally event in the near horizon
- Model families compared:
  - `naive`: simple reference baseline (no advanced learning)
  - `lear`: linear autoregressive model (regularized linear baseline)
  - `logreg`: logistic regression classifier
  - `gbdt_reg` / `gbdt_cls`: gradient-boosted decision trees (regression/classification)
  - `dnn_reg` / `dnn_cls`: neural network models

Why two tasks: one measures raw price forecast quality, the other measures decision-oriented risk prediction.

## 2) How winners were selected (auto-pick logic)
The system automatically selects champions by task and market:

- Forecast champion: **lowest sMAPE** (tie-breaker: MAE)
- Rally champion: **highest PR-AUC** (tie-breaker: lowest Brier)

## 3) Headline results
### Champion summary
- **PJM forecast champion**: `gbdt_reg` (sMAPE: `12.01`)
- **PJM rally champion**: `gbdt_cls` (PR-AUC: `0.870`)
- **NP forecast champion**: `dnn_reg` (sMAPE: `5.54`)
- **NP rally champion**: `dnn_cls` (PR-AUC: `0.789`)

### Lift versus naive baseline
- **PJM forecast**: sMAPE improved by **34.6%** (`18.36` -> `12.01`)
- **NP forecast**: sMAPE improved by **15.3%** (`6.54` -> `5.54`)
- **PJM rally**: PR-AUC improved by **+0.493** (`0.377` -> `0.870`), about **2.31x** naive
- **NP rally**: PR-AUC improved by **+0.464** (`0.325` -> `0.789`), about **2.43x** naive

Interpretation: the approach is not only beating naive baselines; it is finding materially better models for both predictive tasks.

## 4) Why this matters for business users
For a reader who is not model-focused, the key point is:

- The workflow can **benchmark multiple model families**,
- **auto-select the best model per context**, and
- produce repeatable evidence that is easy to review.

This maps well to freight/refinery use cases where the same platform can be reused with different targets and features.

## 5) Important caveats
- This is validated on public electricity market data, not Vitol internal data.
- DNN models currently show convergence warnings in some runs; results are still valid for ranking, but further tuning would improve training stability.
- Model winners differ across markets, which is expected and reinforces the need for automatic selection.

## 6) Artifacts to review quickly
- Benchmark report: `/Users/chinardankhara/Documents/GitHub/time-copilot-test/reports/dual_market_benchmark.md`
- Executive + analytical chart pack:
  - `/Users/chinardankhara/Documents/GitHub/time-copilot-test/reports/charts/01_champion_scorecard.png`
  - `/Users/chinardankhara/Documents/GitHub/time-copilot-test/reports/charts/02_forecast_smape.png`
  - `/Users/chinardankhara/Documents/GitHub/time-copilot-test/reports/charts/03_rally_prauc.png`
  - `/Users/chinardankhara/Documents/GitHub/time-copilot-test/reports/charts/04_forecast_efficiency.png`
  - `/Users/chinardankhara/Documents/GitHub/time-copilot-test/reports/charts/05_rally_calibration.png`

## 7) Recommended next step
Run the same framework on internal freight/refinery targets using desk-specific features and compare against current desk heuristics as the baseline.
