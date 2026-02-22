from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from time_copilot_demo.dnn import fit_predict_dnn_classification, fit_predict_dnn_regression


def available_forecast_models() -> tuple[str, ...]:
    return ("naive", "lear", "gbdt_reg", "dnn_reg")


def available_rally_models() -> tuple[str, ...]:
    return ("naive", "logreg", "gbdt_cls", "dnn_cls")


def predict_forecast(
    model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    if model_name == "naive":
        if "lag_1" in X_test.columns:
            return X_test["lag_1"].to_numpy(dtype=float)
        return np.full(len(X_test), float(y_train.mean()))

    if model_name == "lear":
        # LEAR-like high-dimensional linear autoregressive baseline.
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "reg",
                    ElasticNet(
                        alpha=0.001,
                        l1_ratio=0.9,
                        max_iter=3000,
                        random_state=7,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)

    if model_name == "gbdt_reg":
        model = HistGradientBoostingRegressor(max_depth=6, max_iter=80, learning_rate=0.05, random_state=7)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    if model_name == "dnn_reg":
        return fit_predict_dnn_regression(X_train, y_train, X_test)

    raise ValueError(f"Unknown forecast model: {model_name}")


def predict_rally_probability(
    model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    if model_name == "naive":
        return np.full(len(X_test), float(y_train.mean()))

    if model_name == "logreg":
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]

    if model_name == "gbdt_cls":
        model = HistGradientBoostingClassifier(max_depth=6, max_iter=80, learning_rate=0.05, random_state=7)
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]

    if model_name == "dnn_cls":
        return fit_predict_dnn_classification(X_train, y_train, X_test)

    raise ValueError(f"Unknown rally model: {model_name}")
