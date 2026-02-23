from pathlib import Path

import numpy as np
import pandas as pd

from time_copilot_demo.charts import generate_chart_pack


EXPECTED_FILES = {
    "01_champion_scorecard.png",
    "02_forecast_smape.png",
    "03_rally_prauc.png",
    "04_forecast_efficiency.png",
    "05_rally_calibration.png",
    "README.md",
}


def _seed_artifacts(root: Path) -> None:
    rows = []
    rng = np.random.default_rng(7)

    for market in ["PJM", "NP"]:
        d = root / market
        d.mkdir(parents=True, exist_ok=True)

        forecast = pd.DataFrame(
            [
                {"model": "naive", "mae": 5.0, "rmse": 7.0, "smape": 18.0},
                {"model": "lear", "mae": 4.0, "rmse": 6.0, "smape": 16.0},
                {"model": "gbdt_reg", "mae": 3.0, "rmse": 5.0, "smape": 12.0},
                {"model": "dnn_reg", "mae": 3.3, "rmse": 5.2, "smape": 13.0},
            ]
        )
        forecast.to_csv(d / "forecast_benchmark.csv", index=False)

        rally = pd.DataFrame(
            [
                {"model": "naive", "pr_auc": 0.35, "roc_auc": 0.5, "f1": 0.52, "brier": 0.23},
                {"model": "logreg", "pr_auc": 0.80, "roc_auc": 0.88, "f1": 0.75, "brier": 0.14},
                {"model": "gbdt_cls", "pr_auc": 0.86, "roc_auc": 0.91, "f1": 0.76, "brier": 0.12},
                {"model": "dnn_cls", "pr_auc": 0.84, "roc_auc": 0.89, "f1": 0.74, "brier": 0.14},
            ]
        )
        rally.to_csv(d / "rally_benchmark.csv", index=False)

        n = 200
        y_true = rng.binomial(1, 0.35, size=n)
        preds = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
                "y_true": y_true,
                "naive_prob": np.full(n, y_true.mean()),
                "logreg_prob": np.clip(y_true * 0.65 + rng.normal(0.2, 0.1, n), 0, 1),
                "gbdt_cls_prob": np.clip(y_true * 0.7 + rng.normal(0.15, 0.1, n), 0, 1),
                "dnn_cls_prob": np.clip(y_true * 0.68 + rng.normal(0.18, 0.1, n), 0, 1),
            }
        )
        preds.to_parquet(d / "rally_predictions.parquet", index=False)

        rows.append(
            {
                "market": market,
                "task": "forecast",
                "model": "gbdt_reg",
                "primary_metric": "smape",
                "primary_value": 12.0,
            }
        )
        rows.append(
            {
                "market": market,
                "task": "rally",
                "model": "gbdt_cls",
                "primary_metric": "pr_auc",
                "primary_value": 0.86,
            }
        )

    pd.DataFrame(rows).to_csv(root / "champion_summary.csv", index=False)


def test_generate_chart_pack_outputs_expected_files(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "charts"
    _seed_artifacts(artifacts)

    generate_chart_pack(artifacts_dir=artifacts, output_dir=output)

    actual = {p.name for p in output.iterdir()}
    assert EXPECTED_FILES.issubset(actual)
