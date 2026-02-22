# Time Copilot PJM Rally Demo

Benchmark-first prototype to demonstrate rally-risk forecasting on public data.

## Scope
- Fast baseline benchmark on PJM-like time series.
- Rally event probability prediction (`P(rally in next horizon)`).
- Reproducible artifacts for colleague-facing review.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest tests
python scripts/run_benchmark.py --source public-pjm --artifacts-dir artifacts/public_pjm
```

Artifacts are written to `artifacts/`:
- `metrics.json`
- `predictions.parquet`
- `benchmark_table.csv`

## Data Sources
- `--source public-pjm` (default): loads open PJM data from EPF Zenodo and caches in `datasets/`.
- `--source synthetic`: uses generated data for fast local smoke tests.
- `--source csv --csv-path <file>`: run on a custom dataset with `timestamp,price`.

## Parallelization Strategy
- Track A: data + benchmark harness.
- Track B: improved model plugins.
- Track C: report/notebook narrative.

See `/Users/chinardankhara/Documents/GitHub/time-copilot-test/docs/plans/2026-02-22-pjm-rally-design.md` and `/Users/chinardankhara/Documents/GitHub/time-copilot-test/docs/plans/2026-02-22-pjm-rally-implementation.md`.
