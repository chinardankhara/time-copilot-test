from __future__ import annotations

import pandas as pd


def build_rally_labels(
    price: pd.Series, *, quantile: float = 0.95, lookback: int = 24 * 30
) -> pd.Series:
    """Label timesteps where price crosses a trailing quantile threshold."""
    threshold = price.shift(1).rolling(lookback, min_periods=lookback).quantile(quantile)
    labels = (price >= threshold).astype("float").fillna(0.0).astype(int)
    return pd.Series(labels, index=price.index, name="rally")


def label_future_rally(rally: pd.Series, *, horizon: int = 24) -> pd.Series:
    """Target whether a rally appears in the next horizon steps."""
    future_max = rally.shift(-1).rolling(window=horizon, min_periods=1).max()
    return future_max.fillna(0).astype(int).rename("target_future_rally")

