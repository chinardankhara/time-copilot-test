import pandas as pd

from time_copilot_demo.model_registry import (
    available_forecast_models,
    available_rally_models,
    predict_forecast,
    predict_rally_probability,
)


def _toy_data():
    X = pd.DataFrame(
        {
            "lag_1": [1, 2, 3, 4, 5, 6, 7, 8],
            "lag_2": [0, 1, 2, 3, 4, 5, 6, 7],
            "hour": [0, 1, 2, 3, 4, 5, 6, 7],
            "dayofweek": [0, 0, 0, 0, 0, 0, 0, 0],
            "month": [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    y_reg = pd.Series([2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    y_cls = pd.Series([0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    return X, y_reg, y_cls


def test_available_model_catalogs_include_expected_entries():
    assert set(available_forecast_models()) == {"naive", "lear", "gbdt_reg", "dnn_reg"}
    assert set(available_rally_models()) == {"naive", "logreg", "gbdt_cls", "dnn_cls"}


def test_predict_forecast_and_rally_probability_shapes():
    X, y_reg, y_cls = _toy_data()

    for model in ["naive", "lear", "gbdt_reg", "dnn_reg"]:
        pred = predict_forecast(model, X, y_reg, X)
        assert len(pred) == len(X)

    for model in ["naive", "logreg", "gbdt_cls", "dnn_cls"]:
        pred = predict_rally_probability(model, X, y_cls, X)
        assert len(pred) == len(X)
        assert ((pred >= 0) & (pred <= 1)).all()
