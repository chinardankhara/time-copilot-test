from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from time_copilot_demo.baselines import fit_logreg_baseline, naive_probability_baseline
from time_copilot_demo.evaluate import classification_metrics
from time_copilot_demo.features import build_features
from time_copilot_demo.labels import build_rally_labels, label_future_rally


def synthetic_pjm_like(n_hours: int = 24 * 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    daily = 12 * np.sin(np.arange(n_hours) * (2 * np.pi / 24))
    weekly = 8 * np.sin(np.arange(n_hours) * (2 * np.pi / (24 * 7)))
    noise = rng.normal(0, 3.5, n_hours)
    spikes = (rng.random(n_hours) < 0.015) * rng.normal(20, 5, n_hours)
    price = 45 + daily + weekly + noise + spikes
    return pd.DataFrame({"timestamp": ts, "price": price})


def run_benchmark_pipeline(df: pd.DataFrame | None, artifacts_dir: Path, horizon: int) -> None:
    if df is None:
        df = synthetic_pjm_like()

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["rally"] = build_rally_labels(df["price"], quantile=0.95, lookback=24 * 30)
    df["target"] = label_future_rally(df["rally"], horizon=horizon)

    # Build features from raw market columns only to avoid target leakage/collisions.
    feats = build_features(df[["timestamp", "price"]], lags=(1, 24, 48), rolling_windows=(24, 24 * 7))
    model_df = feats.join(df[["target"]]).dropna().reset_index(drop=True)

    split_idx = int(len(model_df) * 0.8)
    train = model_df.iloc[:split_idx]
    test = model_df.iloc[split_idx:]

    feature_cols = [
        c
        for c in model_df.columns
        if c.startswith("lag_") or c.startswith("roll_") or c in {"hour", "dayofweek", "month"}
    ]
    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    baseline_prob = naive_probability_baseline(y_train)
    naive_pred = np.full(len(y_test), baseline_prob)
    naive_metrics = classification_metrics(y_test, naive_pred, threshold=baseline_prob)

    model = fit_logreg_baseline(X_train, y_train)
    model_prob = model.predict_proba(X_test)[:, 1]
    logreg_metrics = classification_metrics(y_test, model_prob)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"naive": naive_metrics, "logreg": logreg_metrics}
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    preds = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "y_true": y_test.to_numpy(),
            "naive_prob": naive_pred,
            "logreg_prob": model_prob,
        }
    )
    preds.to_parquet(artifacts_dir / "predictions.parquet", index=False)

    table = pd.DataFrame([{"model": "naive", **naive_metrics}, {"model": "logreg", **logreg_metrics}])
    table["pr_auc_lift_vs_naive"] = table["pr_auc"] - float(naive_metrics["pr_auc"])
    table.to_csv(artifacts_dir / "benchmark_table.csv", index=False)

