from pathlib import Path

import pandas as pd

from time_copilot_demo.reporting import write_dual_market_report


def test_write_dual_market_report_contains_both_markets(tmp_path: Path):
    art = tmp_path / "artifacts"
    for market in ["PJM", "NP"]:
        d = art / market
        d.mkdir(parents=True)
        pd.DataFrame([
            {"model": "naive", "mae": 3.0, "rmse": 5.0, "smape": 10.0},
            {"model": "gbdt_reg", "mae": 2.5, "rmse": 4.0, "smape": 8.0},
        ]).to_csv(d / "forecast_benchmark.csv", index=False)
        pd.DataFrame([
            {"model": "naive", "pr_auc": 0.4, "roc_auc": 0.5, "f1": 0.5, "brier": 0.2},
            {"model": "gbdt_cls", "pr_auc": 0.8, "roc_auc": 0.9, "f1": 0.7, "brier": 0.12},
        ]).to_csv(d / "rally_benchmark.csv", index=False)

    pd.DataFrame([
        {"market": "PJM", "task": "forecast", "model": "gbdt_reg", "primary_metric": "smape", "primary_value": 8.0},
        {"market": "PJM", "task": "rally", "model": "gbdt_cls", "primary_metric": "pr_auc", "primary_value": 0.8},
        {"market": "NP", "task": "forecast", "model": "gbdt_reg", "primary_metric": "smape", "primary_value": 8.0},
        {"market": "NP", "task": "rally", "model": "gbdt_cls", "primary_metric": "pr_auc", "primary_value": 0.8},
    ]).to_csv(art / "champion_summary.csv", index=False)

    out = tmp_path / "report.md"
    write_dual_market_report(art, out)

    text = out.read_text(encoding="utf-8")
    assert "PJM" in text
    assert "NP" in text
    assert "Champion" in text
    assert "freight" in text.lower()
