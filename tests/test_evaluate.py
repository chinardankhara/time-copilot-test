from time_copilot_demo.evaluate import classification_metrics


def test_classification_metrics_returns_key_fields():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.3, 0.6, 0.9]

    metrics = classification_metrics(y_true, y_prob, threshold=0.5)

    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert "brier" in metrics
