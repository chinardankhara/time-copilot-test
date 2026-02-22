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
    if _torch_available():
        import torch
        from torch import nn

        device = torch.device(_device_name())
        Xtr = torch.tensor(X_train.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        ytr = torch.tensor(y_train.to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=device)
        Xte = torch.tensor(X_test.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)

        model = nn.Sequential(
            nn.Linear(Xtr.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(60):
            opt.zero_grad()
            pred = model(Xtr)
            loss = loss_fn(pred, ytr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(Xte).squeeze(-1).detach().cpu().numpy()
        return out

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
    if _torch_available():
        import torch
        from torch import nn

        device = torch.device(_device_name())
        Xtr = torch.tensor(X_train.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        ytr = torch.tensor(y_train.to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=device)
        Xte = torch.tensor(X_test.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)

        model = nn.Sequential(
            nn.Linear(Xtr.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(60):
            opt.zero_grad()
            logits = model(Xtr)
            loss = loss_fn(logits, ytr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xte)).squeeze(-1).detach().cpu().numpy()
        return probs

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
