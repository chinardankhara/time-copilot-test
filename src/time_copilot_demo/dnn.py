from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _device_name() -> str:
    if not _torch_available():
        return "cpu"
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def fit_predict_dnn_regression(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    # Lightweight deterministic fallback that works on any machine.
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=60,
        random_state=7,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_dnn_classification(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=60,
        random_state=7,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]
