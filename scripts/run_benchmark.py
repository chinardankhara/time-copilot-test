from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from time_copilot_demo.data import load_epf_market
from time_copilot_demo.pipeline import run_benchmark_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PJM rally benchmark pipeline.")
    parser.add_argument(
        "--source",
        choices=["public-pjm", "synthetic", "csv"],
        default="public-pjm",
        help="Dataset source to run the benchmark against.",
    )
    parser.add_argument("--csv-path", default=None, help="Optional CSV with timestamp,price columns.")
    parser.add_argument("--data-dir", default="datasets", help="Cache directory for downloaded public datasets.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Output directory for benchmark artifacts.")
    parser.add_argument("--horizon", type=int, default=24, help="Forward horizon (hours) for rally target.")
    args = parser.parse_args()

    df = None
    if args.source == "csv":
        if not args.csv_path:
            raise ValueError("--csv-path is required when --source=csv")
        df = pd.read_csv(args.csv_path, parse_dates=["timestamp"])
    elif args.source == "public-pjm":
        df = load_epf_market("PJM", data_dir=args.data_dir)
    run_benchmark_pipeline(df=df, artifacts_dir=Path(args.artifacts_dir), horizon=args.horizon)


if __name__ == "__main__":
    main()
