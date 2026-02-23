from __future__ import annotations

import argparse
from pathlib import Path

from time_copilot_demo.charts import generate_chart_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate executive and analytical benchmark chart packs.")
    parser.add_argument("--artifacts-dir", default="artifacts/dual_market", help="Benchmark artifact directory.")
    parser.add_argument("--output-dir", default="reports/charts", help="Chart output directory.")
    args = parser.parse_args()

    generate_chart_pack(artifacts_dir=Path(args.artifacts_dir), output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()

