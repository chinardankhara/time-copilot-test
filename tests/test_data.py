import pandas as pd

from time_copilot_demo.data import normalize_epf_frame


def test_normalize_epf_frame_outputs_timestamp_price():
    idx = pd.date_range("2020-01-01", periods=3, freq="h")
    raw = pd.DataFrame(
        {
            "Price": [20.0, 25.0, 30.0],
            "Exogenous 1": [100, 101, 102],
        },
        index=idx,
    )

    out = normalize_epf_frame(raw)

    assert list(out.columns) == ["timestamp", "price"]
    assert out["timestamp"].iloc[0] == idx[0]
    assert out["price"].iloc[2] == 30.0


def test_normalize_epf_frame_handles_public_pjm_column_name():
    idx = pd.date_range("2020-01-01", periods=2, freq="h")
    raw = pd.DataFrame(
        {
            " Zonal COMED price": [10.0, 11.0],
            " System load forecast": [100.0, 120.0],
        },
        index=idx,
    )

    out = normalize_epf_frame(raw)

    assert list(out.columns) == ["timestamp", "price"]
    assert out["price"].tolist() == [10.0, 11.0]
