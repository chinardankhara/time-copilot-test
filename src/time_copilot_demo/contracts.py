from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp


def validate_time_split(split: TimeSplit) -> None:
    """Validate chronological split boundaries for time series modeling."""
    if split.train_end >= split.val_start:
        raise ValueError("overlap between train and validation windows")
    if split.val_start > split.val_end:
        raise ValueError("validation window is inverted")
    if split.val_end >= split.test_start:
        raise ValueError("overlap between validation and test windows")

