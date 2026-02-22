from __future__ import annotations

from pathlib import Path

import pandas as pd

EPF_ZENODO_BASE = "https://zenodo.org/records/4624805/files"


def normalize_epf_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize EPF-style frame to the project schema."""
    working = raw.copy()
    working.columns = [str(c).strip() for c in working.columns]
    price_candidates = [c for c in working.columns if "price" in c.lower()]
    if not price_candidates:
        raise ValueError("expected a price-like column in EPF frame")
    price_col = price_candidates[0]

    out = working.reset_index()
    timestamp_col = out.columns[0]
    out = out.rename(columns={timestamp_col: "timestamp", price_col: "price"})
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out[["timestamp", "price"]]


def load_epf_market(dataset: str = "PJM", data_dir: str = "datasets") -> pd.DataFrame:
    """Load market data from local cache or EPF Zenodo source."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    local_file = data_path / f"{dataset}.csv"

    if local_file.exists():
        raw = pd.read_csv(local_file, index_col=0)
    else:
        url = f"{EPF_ZENODO_BASE}/{dataset}.csv"
        raw = pd.read_csv(url, index_col=0)
        raw.to_csv(local_file)

    raw.index = pd.to_datetime(raw.index)
    return normalize_epf_frame(raw)
