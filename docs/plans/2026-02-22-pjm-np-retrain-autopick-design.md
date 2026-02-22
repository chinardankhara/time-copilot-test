# PJM+NP Retrain Benchmarks With Auto-Pick Design

## Goal
Deliver a benchmark-rigorous prototype on two public markets (PJM and Nord Pool) that retrains strong baseline families and automatically selects champions per task.

## Scope
- Markets: PJM and NP
- Tasks:
  - Point forecast benchmark (regression)
  - Rally-risk benchmark (classification)
- Model families:
  - Classical: naive, LEAR-like linear autoregression, gradient boosting
  - Neural: DNN for regression and classification
- Selection:
  - Auto champion per market/task using explicit metric rules

## Benchmark Definitions
1. Point Forecast Benchmark
- Target: next-hour price forecast
- Metrics: MAE, RMSE, sMAPE
- Champion rule: lowest sMAPE, tie-break MAE

2. Rally-Risk Benchmark
- Target: P(rally in next H hours)
- Rally label: threshold crossing based on trailing quantile
- Metrics: PR-AUC, ROC-AUC, F1, Brier
- Champion rule: highest PR-AUC, tie-break lowest Brier

## Architecture
1. Data layer
- Load PJM/NP open CSVs from EPF source cache.
- Preserve timestamp and price columns, optional exogenous columns for expansion.

2. Feature layer
- Leak-safe lag and rolling features.
- Shared feature contract consumed by both regression and classification tasks.

3. Model layer
- Forecast models: naive, lear, gbdt_reg, dnn_reg.
- Rally models: naive, logreg, gbdt_cls, dnn_cls.
- DNN backend uses MPS when torch+mps are available, otherwise CPU fallback.

4. Evaluation + Auto-pick layer
- Per-market benchmark tables for forecast and rally.
- Champion picker writes concise machine-readable outputs.

5. Reporting layer
- Combined markdown summary for both markets with benchmark + champion tables.

## Parallel PR/Worktree Strategy
- Track A: contracts, splits, dual-market runner, benchmark outputs.
- Track B: classical model suite (LEAR/GBDT/logreg) and metrics.
- Track C: DNN reg+cls with MPS support and fallback behavior.
- Track D: auto-pick logic + final report generation.

## Risks
- DNN instability across machines:
  - Mitigation: deterministic seeds, fallback backend, explicit epoch configs.
- Data leakage:
  - Mitigation: split validators + feature tests.
- Metric mismatch across tasks:
  - Mitigation: explicit task-specific evaluation modules and schema tests.

## Definition of Done
- Both markets run end-to-end from one command.
- Forecast and rally benchmarks produced for both markets.
- Auto-pick outputs champions per market/task.
- DNN path supports MPS when available and remains runnable without it.
