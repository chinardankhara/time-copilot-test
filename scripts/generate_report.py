from __future__ import annotations

import argparse
from pathlib import Path

from time_copilot_demo.reporting import write_dual_market_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dual-market benchmark report.")
    parser.add_argument("--artifacts-dir", default="artifacts/dual_market", help="Directory with benchmark artifacts.")
    parser.add_argument("--output", default="reports/dual_market_benchmark.md", help="Output report path.")
    args = parser.parse_args()

    write_dual_market_report(Path(args.artifacts_dir), Path(args.output))


if __name__ == "__main__":
    main()

