# PJM Rally Risk Demo Design

## Goal
Build a fast, benchmarked prototype that demonstrates Time Copilot style forecasting for "rally risk" on public data, then extend the same architecture to a second risk stream (asset outage analog).

## Why PJM First
- Public benchmark ecosystem exists (epftoolbox market datasets + common baselines).
- Clear analogue to freight/refinery desks: predict rare adverse/upside regime shifts under non-stationarity.
- Faster delivery than dual-domain build from day one.

## Problem Framing
Primary task:
- Input: hourly day-ahead prices and derived lag/calendar features.
- Output: probability that next horizon enters a rally event.

Rally definition (MVP):
- Event at time t if price(t) >= rolling quantile threshold (e.g., q95 over trailing 30 days).
- Horizons:
  - H1: rally in next 24h.
  - H2: rally in next 7d.

Secondary task (for benchmark credibility):
- Point forecast on day-ahead prices to compare with standard EPF metrics.

## Architecture
1. Data layer
- Deterministic loader for PJM benchmark data.
- Time-aware split strategy (train/validation/test by date).
- Feature store from pure functions (lags, rolling stats, calendar).

2. Modeling layer
- Baselines:
  - Naive persistence baseline.
  - Linear / tree baseline for classification.
- Advanced model (PR2): compact sequence model (e.g., LightGBM with lag stacks or temporal net).

3. Evaluation layer
- Classification metrics for rally task: PR-AUC, ROC-AUC, F1@operating point, Brier.
- Forecast metrics for price benchmark: MAE, RMSE, sMAPE.
- Relative lift vs baseline (primary success metric).

4. Story layer
- Notebook/report with decision-oriented outputs:
  - Top rally-risk windows.
  - Calibration curves.
  - Error decomposition by volatility regime.

## Parallel Delivery Model
After bootstrap, split into independent tracks:
- Track A: benchmark harness + data + metrics (foundation, merge first).
- Track B: improved model plugged into Track A contract.
- Track C: colleague-facing narrative/report that consumes frozen outputs from Track A/B.

## Non-Goals (MVP)
- No heavy MLOps deployment.
- No proprietary data connectors.
- No live intraday execution system.

## Success Criteria
- Reproducible run command from clean checkout.
- Benchmark table with baseline vs improved model on held-out test period.
- Rally-risk model shows measurable lift over naive baseline on PR-AUC.
- Short narrative ties results to freight/refinery decision analogs.

## Risks and Mitigations
- Data leakage from temporal features:
  - Mitigation: strict lagging and split guards with tests.
- Weak event signal from threshold choice:
  - Mitigation: sensitivity analysis over threshold window and quantiles.
- Overfitting on one market regime:
  - Mitigation: walk-forward validation and calibration checks.
