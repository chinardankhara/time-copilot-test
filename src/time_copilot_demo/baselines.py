from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def naive_probability_baseline(y_train: pd.Series) -> float:
    return float(y_train.mean())


def fit_logreg_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    return model

