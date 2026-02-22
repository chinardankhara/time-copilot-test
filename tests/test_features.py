import pandas as pd

from time_copilot_demo.features import build_features


def test_build_features_uses_past_values_only():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=6, freq="h"),
            "price": [10, 20, 30, 40, 50, 60],
        }
    )

    out = build_features(df, lags=(1, 2), rolling_windows=(3,))

    row = out.iloc[3]
    assert row["lag_1"] == 30
    assert row["lag_2"] == 20
    assert row["roll_mean_3"] == (10 + 20 + 30) / 3
