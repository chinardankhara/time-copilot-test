import pandas as pd

from time_copilot_demo.champion import pick_forecast_champion, pick_rally_champion


def test_pick_forecast_champion_prefers_lowest_smape_then_mae():
    table = pd.DataFrame(
        [
            {"model": "a", "smape": 10.0, "mae": 2.0},
            {"model": "b", "smape": 9.0, "mae": 2.5},
            {"model": "c", "smape": 9.0, "mae": 2.1},
        ]
    )

    champion = pick_forecast_champion(table)
    assert champion["model"] == "c"


def test_pick_rally_champion_prefers_highest_pr_auc_then_lowest_brier():
    table = pd.DataFrame(
        [
            {"model": "x", "pr_auc": 0.7, "brier": 0.20},
            {"model": "y", "pr_auc": 0.8, "brier": 0.25},
            {"model": "z", "pr_auc": 0.8, "brier": 0.15},
        ]
    )

    champion = pick_rally_champion(table)
    assert champion["model"] == "z"
