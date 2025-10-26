
from typing import Dict, Any
import pandas as pd
import numpy as np

def replay_actual(driver_id: int, race_id: int, laps_df: pd.DataFrame, results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Replays a driver's actual race by summing lap_times, then compares with official milliseconds (results.csv).
    Returns: dict with actual_sum_ms, official_ms, abs_error_ms, pct_error
    """
    dlap = laps_df[(laps_df["raceId"] == race_id) & (laps_df["driverId"] == driver_id)]
    if dlap.empty:
        return {"error": "No laps found for driver/race."}

    actual_sum_ms = pd.to_numeric(dlap["milliseconds"], errors="coerce").sum()
    row = results_df[(results_df["raceId"] == race_id) & (results_df["driverId"] == driver_id)]
    if row.empty:
        return {"error": "No official result found for driver/race."}
    official_ms = pd.to_numeric(row["milliseconds"], errors="coerce").iloc[0]

    if pd.isna(official_ms) or official_ms == 0:
        return {"error": "Official milliseconds missing for driver/race."}

    abs_error = float(abs(actual_sum_ms - official_ms))
    pct_error = float(abs_error / official_ms * 100.0)
    return {
        "actual_sum_ms": float(actual_sum_ms),
        "official_ms": float(official_ms),
        "abs_error_ms": abs_error,
        "pct_error": pct_error
    }
