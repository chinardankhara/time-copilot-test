from __future__ import annotations

import pandas as pd


def build_features(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    lags: tuple[int, ...] = (1, 24, 48),
    rolling_windows: tuple[int, ...] = (24, 24 * 7),
) -> pd.DataFrame:
    """Create lagged and rolling features without future leakage."""
    out = df.copy()
    out = out.sort_values(timestamp_col).reset_index(drop=True)

    for lag in lags:
        out[f"lag_{lag}"] = out[price_col].shift(lag)

    shifted = out[price_col].shift(1)
    for window in rolling_windows:
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std(ddof=0)

    dt = pd.to_datetime(out[timestamp_col])
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    return out

