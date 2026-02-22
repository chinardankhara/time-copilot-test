from pathlib import Path

import pandas as pd

from time_copilot_demo.pipeline import run_dual_market_benchmark, synthetic_pjm_like


def test_run_dual_market_benchmark_writes_market_artifacts(tmp_path: Path):
    market_frames = {
        "PJM": synthetic_pjm_like(24 * 90),
        "NP": synthetic_pjm_like(24 * 90),
    }

    run_dual_market_benchmark(market_frames=market_frames, artifacts_dir=tmp_path, horizon=24)

    for market in ["PJM", "NP"]:
        market_dir = tmp_path / market
        assert (market_dir / "forecast_benchmark.csv").exists()
        assert (market_dir / "rally_benchmark.csv").exists()
        assert (market_dir / "champions.json").exists()

    summary = pd.read_csv(tmp_path / "champion_summary.csv")
    assert set(summary["market"]) == {"PJM", "NP"}
