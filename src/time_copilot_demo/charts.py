from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

matplotlib.use("Agg")

EXEC_FILES = [
    "01_champion_scorecard.png",
    "02_forecast_smape.png",
    "03_rally_prauc.png",
]
ANALYTICAL_FILES = [
    "04_forecast_efficiency.png",
    "05_rally_calibration.png",
]


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def _load_market_tables(artifacts_dir: Path, market: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast = pd.read_csv(artifacts_dir / market / "forecast_benchmark.csv")
    rally = pd.read_csv(artifacts_dir / market / "rally_benchmark.csv")
    preds = pd.read_parquet(artifacts_dir / market / "rally_predictions.parquet")
    forecast["market"] = market
    rally["market"] = market
    preds["market"] = market
    return forecast, rally, preds


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _chart_champion_scorecard(champion_summary: pd.DataFrame, output_dir: Path) -> None:
    forecast = champion_summary[champion_summary["task"] == "forecast"].copy()
    rally = champion_summary[champion_summary["task"] == "rally"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=forecast, x="market", y="primary_value", hue="model", ax=axes[0], palette="crest")
    axes[0].set_title("Forecast Champions (Lower sMAPE Better)")
    axes[0].set_ylabel("sMAPE")
    axes[0].set_xlabel("")

    sns.barplot(data=rally, x="market", y="primary_value", hue="model", ax=axes[1], palette="magma")
    axes[1].set_title("Rally Champions (Higher PR-AUC Better)")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 1)

    _save(fig, output_dir / "01_champion_scorecard.png")


def _chart_forecast_smape(forecast_all: pd.DataFrame, output_dir: Path) -> None:
    order = (
        forecast_all.groupby("model", as_index=False)["smape"]
        .mean()
        .sort_values("smape")["model"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=forecast_all,
        x="model",
        y="smape",
        hue="market",
        order=order,
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Forecast Benchmark: sMAPE by Model and Market")
    ax.set_ylabel("sMAPE (Lower Better)")
    ax.set_xlabel("")
    _save(fig, output_dir / "02_forecast_smape.png")


def _chart_rally_prauc(rally_all: pd.DataFrame, output_dir: Path) -> None:
    order = (
        rally_all.groupby("model", as_index=False)["pr_auc"]
        .mean()
        .sort_values("pr_auc", ascending=False)["model"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=rally_all,
        x="model",
        y="pr_auc",
        hue="market",
        order=order,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Rally Benchmark: PR-AUC by Model and Market")
    ax.set_ylabel("PR-AUC (Higher Better)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    _save(fig, output_dir / "03_rally_prauc.png")


def _chart_forecast_efficiency(forecast_all: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    sns.scatterplot(
        data=forecast_all,
        x="mae",
        y="smape",
        hue="model",
        style="market",
        size="rmse",
        sizes=(80, 280),
        palette="tab10",
        ax=ax,
    )
    ax.set_title("Analytical View: Forecast Efficiency Frontier")
    ax.set_xlabel("MAE (Lower Better)")
    ax.set_ylabel("sMAPE (Lower Better)")
    _save(fig, output_dir / "04_forecast_efficiency.png")


def _chart_rally_calibration(rally_preds_all: pd.DataFrame, rally_all: pd.DataFrame, output_dir: Path) -> None:
    markets = sorted(rally_preds_all["market"].unique().tolist())
    fig, axes = plt.subplots(1, len(markets), figsize=(13, 5), sharey=True)
    if len(markets) == 1:
        axes = [axes]

    for ax, market in zip(axes, markets):
        pred_df = rally_preds_all[rally_preds_all["market"] == market]
        model_table = rally_all[rally_all["market"] == market].sort_values("pr_auc", ascending=False)
        models = model_table["model"].tolist()
        for model in models:
            prob_col = f"{model}_prob"
            if prob_col not in pred_df.columns:
                continue
            frac_pos, mean_pred = calibration_curve(
                pred_df["y_true"].to_numpy(),
                pred_df[prob_col].to_numpy(),
                n_bins=10,
                strategy="quantile",
            )
            ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=model)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#4a4a4a", linewidth=1)
        ax.set_title(f"{market} Calibration")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", frameon=True)

    fig.suptitle("Analytical View: Rally Calibration Curves", y=1.03)
    _save(fig, output_dir / "05_rally_calibration.png")


def _write_chart_index(output_dir: Path) -> None:
    lines = [
        "# Benchmark Chart Pack",
        "",
        "## Executive Pack",
        "",
    ]
    for file_name in EXEC_FILES:
        lines.append(f"![{file_name}]({file_name})")
        lines.append("")

    lines.append("## Analytical Pack")
    lines.append("")
    for file_name in ANALYTICAL_FILES:
        lines.append(f"![{file_name}]({file_name})")
        lines.append("")

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def generate_chart_pack(artifacts_dir: Path, output_dir: Path) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)

    champion_summary = pd.read_csv(artifacts_dir / "champion_summary.csv")
    forecast_frames = []
    rally_frames = []
    rally_preds = []

    for market in sorted(champion_summary["market"].unique().tolist()):
        forecast, rally, preds = _load_market_tables(artifacts_dir, market)
        forecast_frames.append(forecast)
        rally_frames.append(rally)
        rally_preds.append(preds)

    forecast_all = pd.concat(forecast_frames, ignore_index=True)
    rally_all = pd.concat(rally_frames, ignore_index=True)
    rally_preds_all = pd.concat(rally_preds, ignore_index=True)

    _chart_champion_scorecard(champion_summary, output_dir)
    _chart_forecast_smape(forecast_all, output_dir)
    _chart_rally_prauc(rally_all, output_dir)
    _chart_forecast_efficiency(forecast_all, output_dir)
    _chart_rally_calibration(rally_preds_all, rally_all, output_dir)
    _write_chart_index(output_dir)

