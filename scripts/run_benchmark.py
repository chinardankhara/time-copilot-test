from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from time_copilot_demo.data import load_epf_market
from time_copilot_demo.pipeline import run_dual_market_benchmark, synthetic_pjm_like


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dual-market benchmark pipeline.")
    parser.add_argument("--markets", default="PJM,NP", help="Comma-separated markets (e.g. PJM,NP).")
    parser.add_argument(
        "--source",
        choices=["public", "synthetic", "csv"],
        default="public",
        help="Data source mode.",
    )
    parser.add_argument("--csv-path", default=None, help="CSV path for --source=csv (single-market mode).")
    parser.add_argument("--data-dir", default="datasets", help="Cache directory for public datasets.")
    parser.add_argument("--artifacts-dir", default="artifacts/dual_market", help="Output directory.")
    parser.add_argument("--horizon", type=int, default=24, help="Forward horizon for rally target.")
    args = parser.parse_args()

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    market_frames: dict[str, pd.DataFrame]

    if args.source == "public":
        market_frames = {market: load_epf_market(market, data_dir=args.data_dir) for market in markets}
    elif args.source == "synthetic":
        market_frames = {market: synthetic_pjm_like() for market in markets}
    else:
        if not args.csv_path:
            raise ValueError("--csv-path is required when --source=csv")
        if len(markets) != 1:
            raise ValueError("--source=csv supports a single market name")
        market_frames = {markets[0]: pd.read_csv(args.csv_path, parse_dates=["timestamp"])}

    run_dual_market_benchmark(
        market_frames=market_frames,
        artifacts_dir=Path(args.artifacts_dir),
        horizon=args.horizon,
        markets=markets,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()

