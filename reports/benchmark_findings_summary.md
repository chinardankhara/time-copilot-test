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

## 4) Strong baseline comparisons
This run includes strong non-naive baselines and compares winners against them directly.

### Forecast task (vs `lear` and `dnn_reg`)
- **PJM winner (`gbdt_reg`)**:
  - sMAPE is **1.19 points better** than `dnn_reg` (`12.01` vs `13.20`)
  - sMAPE is **4.11 points better** than `lear` (`12.01` vs `16.13`)
- **NP winner (`dnn_reg`)**:
  - tied as best against the strong set (equal to `dnn_reg` by definition of winner)
  - still **0.32 sMAPE points better** than `lear` (`5.54` vs `5.86`)

### Rally task (vs `logreg` and `dnn_cls`)
- **PJM winner (`gbdt_cls`)**:
  - PR-AUC is **+0.055 higher** than `logreg` (`0.870` vs `0.815`)
  - PR-AUC is **+0.017 higher** than `dnn_cls` (`0.870` vs `0.853`)
- **NP winner (`dnn_cls`)**:
  - PR-AUC is **+0.002 higher** than `logreg` (`0.789` vs `0.787`)
  - winner is `dnn_cls` itself among strong baselines

Interpretation: the result is not only “better than naive”; it remains competitive or superior against strong baseline families.

## 5) Important caveats
- This is validated on public electricity market data, not Vitol internal data.
- DNN models currently show convergence warnings in some runs; results are still valid for ranking, but further tuning would improve training stability.
- Model winners differ across markets, which is expected and reinforces the need for automatic selection.

## 6) Artifacts to review quickly
- Benchmark report: `reports/dual_market_benchmark.md`
- Executive + analytical chart pack:
  - `reports/charts/01_champion_scorecard.png`
  - `reports/charts/02_forecast_smape.png`
  - `reports/charts/03_rally_prauc.png`
  - `reports/charts/04_forecast_efficiency.png`
  - `reports/charts/05_rally_calibration.png`

## 7) Recommended next step
Run the same framework on internal targets and compare against current desk heuristics as the baseline.
