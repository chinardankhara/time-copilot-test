from __future__ import annotations

import pandas as pd


def pick_forecast_champion(table: pd.DataFrame) -> dict[str, float | str]:
    ranked = table.sort_values(["smape", "mae"], ascending=[True, True]).reset_index(drop=True)
    return ranked.iloc[0].to_dict()


def pick_rally_champion(table: pd.DataFrame) -> dict[str, float | str]:
    ranked = table.sort_values(["pr_auc", "brier"], ascending=[False, True]).reset_index(drop=True)
    return ranked.iloc[0].to_dict()

