import pandas as pd

from time_copilot_demo.labels import build_rally_labels


def test_build_rally_labels_uses_trailing_quantile():
    price = pd.Series([10, 12, 15, 20, 45, 11, 13, 16])

    labels = build_rally_labels(price, quantile=0.8, lookback=4)

    # At index 4, trailing window is [10,12,15,20], q80 ~= 17 -> 45 is rally
    assert labels.iloc[4] == 1
