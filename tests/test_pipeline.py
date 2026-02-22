from pathlib import Path

from time_copilot_demo.pipeline import run_benchmark_pipeline


def test_run_benchmark_pipeline_writes_artifacts(tmp_path: Path):
    run_benchmark_pipeline(df=None, artifacts_dir=tmp_path, horizon=24)

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "predictions.parquet").exists()
    assert (tmp_path / "benchmark_table.csv").exists()
