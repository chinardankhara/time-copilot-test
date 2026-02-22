# Time Copilot Dual-Market Benchmark Demo

Benchmark-rigorous prototype for public electricity markets (PJM + NP) with automatic champion selection.

## Scope
- Markets: `PJM`, `NP`
- Tasks:
  - Point forecast benchmark (MAE, RMSE, sMAPE)
  - Rally-risk benchmark (`P(rally in next horizon)`, PR-AUC/ROC-AUC/F1/Brier)
- Model families:
  - Forecast: `naive`, `lear`, `gbdt_reg`, `dnn_reg`
  - Rally: `naive`, `logreg`, `gbdt_cls`, `dnn_cls`
- Auto-pick champions per market/task based on metric rules.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest tests
PYTHONPATH=src python scripts/run_benchmark.py --markets PJM,NP --source public --artifacts-dir artifacts/dual_market
PYTHONPATH=src python scripts/generate_report.py --artifacts-dir artifacts/dual_market --output reports/dual_market_benchmark.md
```

## Outputs
- `artifacts/dual_market/PJM/forecast_benchmark.csv`
- `artifacts/dual_market/PJM/rally_benchmark.csv`
- `artifacts/dual_market/PJM/champions.json`
- `artifacts/dual_market/NP/forecast_benchmark.csv`
- `artifacts/dual_market/NP/rally_benchmark.csv`
- `artifacts/dual_market/NP/champions.json`
- `artifacts/dual_market/champion_summary.csv`
- `reports/dual_market_benchmark.md`

## Data Sources
- `--source public` (default): loads open EPF market data from Zenodo and caches in `datasets/`.
- `--source synthetic`: generates synthetic market frames for fast smoke testing.
- `--source csv --csv-path <file>`: single-market custom CSV mode with `timestamp,price`.

## Auto-Pick Rules
- Forecast champion: lowest `sMAPE` (tie-break `MAE`).
- Rally champion: highest `PR-AUC` (tie-break lowest `Brier`).

See `/Users/chinardankhara/Documents/GitHub/time-copilot-test/.worktrees/track-a/docs/plans/2026-02-22-pjm-np-retrain-autopick-design.md` and `/Users/chinardankhara/Documents/GitHub/time-copilot-test/.worktrees/track-a/docs/plans/2026-02-22-pjm-np-retrain-autopick-implementation.md`.
