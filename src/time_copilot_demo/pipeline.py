from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from time_copilot_demo.champion import pick_forecast_champion, pick_rally_champion
from time_copilot_demo.data import load_epf_market
from time_copilot_demo.evaluate import classification_metrics, forecast_metrics
from time_copilot_demo.features import build_features
from time_copilot_demo.labels import build_rally_labels, label_future_rally
from time_copilot_demo.model_registry import (
    available_forecast_models,
    available_rally_models,
    predict_forecast,
    predict_rally_probability,
)


def synthetic_pjm_like(n_hours: int = 24 * 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    daily = 12 * np.sin(np.arange(n_hours) * (2 * np.pi / 24))
    weekly = 8 * np.sin(np.arange(n_hours) * (2 * np.pi / (24 * 7)))
    noise = rng.normal(0, 3.5, n_hours)
    spikes = (rng.random(n_hours) < 0.015) * rng.normal(20, 5, n_hours)
    price = 45 + daily + weekly + noise + spikes
    return pd.DataFrame({"timestamp": ts, "price": price})


def _build_supervised_frame(df: pd.DataFrame, *, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    data = df.sort_values("timestamp").reset_index(drop=True).copy()
    data["rally"] = build_rally_labels(data["price"], quantile=0.95, lookback=24 * 30)
    data["target_rally"] = label_future_rally(data["rally"], horizon=horizon)
    data["target_price"] = data["price"].shift(-1)

    feats = build_features(
        data[["timestamp", "price"]],
        lags=(1, 2, 24, 48, 24 * 7),
        rolling_windows=(24, 24 * 7),
    )
    out = feats.join(data[["target_rally", "target_price"]]).dropna().reset_index(drop=True)
    feature_cols = [
        c
        for c in out.columns
        if c.startswith("lag_") or c.startswith("roll_") or c in {"hour", "dayofweek", "month"}
    ]
    return out, feature_cols


def _train_test_split(frame: pd.DataFrame, ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(frame) * ratio)
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def run_market_benchmark(
    *,
    market: str,
    df: pd.DataFrame,
    artifacts_dir: Path,
    horizon: int = 24,
    forecast_models: list[str] | None = None,
    rally_models: list[str] | None = None,
) -> dict[str, dict[str, float | str]]:
    frame, feature_cols = _build_supervised_frame(df, horizon=horizon)
    train, test = _train_test_split(frame)

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_reg_train = train["target_price"]
    y_reg_test = test["target_price"]
    y_cls_train = train["target_rally"]
    y_cls_test = test["target_rally"]

    if forecast_models is None:
        forecast_models = list(available_forecast_models())
    if rally_models is None:
        rally_models = list(available_rally_models())

    market_dir = artifacts_dir / market
    market_dir.mkdir(parents=True, exist_ok=True)

    forecast_rows: list[dict[str, float | str]] = []
    forecast_preds = pd.DataFrame({"timestamp": test["timestamp"].to_numpy(), "y_true": y_reg_test.to_numpy()})
    for model_name in forecast_models:
        pred = predict_forecast(model_name, X_train, y_reg_train, X_test)
        forecast_preds[f"{model_name}_pred"] = pred
        forecast_rows.append({"model": model_name, **forecast_metrics(y_reg_test, pred)})

    rally_rows: list[dict[str, float | str]] = []
    rally_preds = pd.DataFrame({"timestamp": test["timestamp"].to_numpy(), "y_true": y_cls_test.to_numpy()})
    for model_name in rally_models:
        prob = predict_rally_probability(model_name, X_train, y_cls_train, X_test)
        rally_preds[f"{model_name}_prob"] = prob
        threshold = float(y_cls_train.mean()) if model_name == "naive" else 0.5
        rally_rows.append({"model": model_name, **classification_metrics(y_cls_test, prob, threshold=threshold)})

    forecast_table = pd.DataFrame(forecast_rows).sort_values(["smape", "mae"], ascending=[True, True]).reset_index(drop=True)
    rally_table = pd.DataFrame(rally_rows).sort_values(["pr_auc", "brier"], ascending=[False, True]).reset_index(drop=True)

    forecast_table.to_csv(market_dir / "forecast_benchmark.csv", index=False)
    rally_table.to_csv(market_dir / "rally_benchmark.csv", index=False)
    forecast_preds.to_parquet(market_dir / "forecast_predictions.parquet", index=False)
    rally_preds.to_parquet(market_dir / "rally_predictions.parquet", index=False)

    champions = {
        "market": market,
        "forecast": pick_forecast_champion(forecast_table),
        "rally": pick_rally_champion(rally_table),
    }
    (market_dir / "champions.json").write_text(json.dumps(champions, indent=2), encoding="utf-8")
    return champions


def run_dual_market_benchmark(
    *,
    market_frames: dict[str, pd.DataFrame] | None,
    artifacts_dir: Path,
    horizon: int = 24,
    markets: list[str] | None = None,
    data_dir: str = "datasets",
) -> pd.DataFrame:
    if markets is None:
        markets = ["PJM", "NP"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if market_frames is None:
        market_frames = {market: load_epf_market(market, data_dir=data_dir) for market in markets}

    summary_rows: list[dict[str, float | str]] = []
    for market in markets:
        champions = run_market_benchmark(
            market=market,
            df=market_frames[market],
            artifacts_dir=artifacts_dir,
            horizon=horizon,
        )
        summary_rows.append(
            {
                "market": market,
                "task": "forecast",
                "model": str(champions["forecast"]["model"]),
                "primary_metric": "smape",
                "primary_value": float(champions["forecast"]["smape"]),
            }
        )
        summary_rows.append(
            {
                "market": market,
                "task": "rally",
                "model": str(champions["rally"]["model"]),
                "primary_metric": "pr_auc",
                "primary_value": float(champions["rally"]["pr_auc"]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(artifacts_dir / "champion_summary.csv", index=False)
    return summary


def run_benchmark_pipeline(df: pd.DataFrame | None, artifacts_dir: Path, horizon: int) -> None:
    # Backward-compatible single-market entrypoint used by legacy tests.
    if df is None:
        df = synthetic_pjm_like()

    champions = run_market_benchmark(market="PJM", df=df, artifacts_dir=artifacts_dir, horizon=horizon)
    market_dir = artifacts_dir / "PJM"
    rally_table = pd.read_csv(market_dir / "rally_benchmark.csv")
    rally_preds = pd.read_parquet(market_dir / "rally_predictions.parquet")

    metrics = {
        row["model"]: {
            "pr_auc": float(row["pr_auc"]),
            "roc_auc": float(row["roc_auc"]),
            "f1": float(row["f1"]),
            "brier": float(row["brier"]),
        }
        for _, row in rally_table.iterrows()
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rally_preds.to_parquet(artifacts_dir / "predictions.parquet", index=False)
    rally_table.to_csv(artifacts_dir / "benchmark_table.csv", index=False)
    (artifacts_dir / "champions.json").write_text(json.dumps(champions, indent=2), encoding="utf-8")

