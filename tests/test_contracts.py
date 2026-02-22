import pandas as pd
import pytest

from time_copilot_demo.contracts import TimeSplit, validate_time_split


def test_validate_time_split_rejects_overlap():
    split = TimeSplit(
        train_end=pd.Timestamp("2020-01-10"),
        val_start=pd.Timestamp("2020-01-09"),
        val_end=pd.Timestamp("2020-01-12"),
        test_start=pd.Timestamp("2020-01-13"),
    )

    with pytest.raises(ValueError, match="overlap"):
        validate_time_split(split)


def test_validate_time_split_accepts_ordered_ranges():
    split = TimeSplit(
        train_end=pd.Timestamp("2020-01-10"),
        val_start=pd.Timestamp("2020-01-11"),
        val_end=pd.Timestamp("2020-01-12"),
        test_start=pd.Timestamp("2020-01-13"),
    )

    validate_time_split(split)
