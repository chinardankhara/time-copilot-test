from time_copilot_demo.evaluate import forecast_metrics


def test_forecast_metrics_returns_mae_rmse_smape():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]

    metrics = forecast_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"mae", "rmse", "smape"}
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["smape"] > 0
