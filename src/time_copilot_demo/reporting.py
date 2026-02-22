from __future__ import annotations

from pathlib import Path

import pandas as pd


def _to_markdown_table(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def write_dual_market_report(artifacts_dir: Path, output_path: Path) -> None:
    champion_summary = pd.read_csv(artifacts_dir / "champion_summary.csv")

    parts: list[str] = []
    parts.append("# Dual-Market Time Copilot Benchmark Report")
    parts.append("")
    parts.append("## Champion Summary")
    parts.append(_to_markdown_table(champion_summary))
    parts.append("")
    parts.append("## Market Details")

    for market in sorted(champion_summary["market"].unique()):
        forecast = pd.read_csv(artifacts_dir / market / "forecast_benchmark.csv")
        rally = pd.read_csv(artifacts_dir / market / "rally_benchmark.csv")

        parts.append(f"### {market}")
        parts.append("")
        parts.append("Forecast benchmark:")
        parts.append(_to_markdown_table(forecast))
        parts.append("")
        parts.append("Rally benchmark:")
        parts.append(_to_markdown_table(rally))
        parts.append("")

    parts.append("## Business Interpretation")
    parts.append(
        "- The same forecasting stack can map to freight and refinery decision workflows by replacing targets and desk-specific features."
    )
    parts.append("- Champion selection is automated per market/task, enabling repeatable model governance.")
    parts.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")

