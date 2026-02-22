import pandas as pd

from time_copilot_demo.baselines import fit_logreg_baseline


def test_fit_logreg_baseline_predicts_probabilities():
    X = pd.DataFrame(
        {
            "lag_1": [1, 2, 3, 4, 5, 6],
            "lag_2": [0, 1, 2, 3, 4, 5],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])

    model = fit_logreg_baseline(X, y)
    preds = model.predict_proba(X)[:, 1]

    assert preds.shape[0] == len(X)
    assert ((preds >= 0) & (preds <= 1)).all()
