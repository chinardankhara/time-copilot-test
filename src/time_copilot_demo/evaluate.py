from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def classification_metrics(
    y_true: Iterable[int], y_prob: Iterable[float], *, threshold: float = 0.5
) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true))
    y_prob_arr = np.asarray(list(y_prob))
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(y_true_arr, y_prob_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "f1": float(f1_score(y_true_arr, y_pred_arr)),
        "brier": float(brier_score_loss(y_true_arr, y_prob_arr)),
    }
    return metrics

