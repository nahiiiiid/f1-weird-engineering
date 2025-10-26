from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from .tyre_models import stint_adjustment_seconds, DEFAULT_TYRE_PARAMS
from .safety_car import laptime_multiplier, pitdelta_multiplier
from .traffic import TrafficConfig, compute_traffic_penalty
from .ml_predictor import MLConfig, LapTimeML


# ------------------------------
# Dataclasses (no mutable defaults!)
# ------------------------------

@dataclass
class StintSpec:
    pit_lap: int         # lap number on which pit occurs (1-indexed). For no-stop race, provide empty list.
    duration_sec: float  # nominal stationary time (no in/out), app will still apply SC/VSC pit delta multipliers
    compound: str        # "Soft"/"Medium"/"Hard"/"Inter"/"Wet"


@dataclass
class SimulationConfig:
    use_ml: bool = False
    ml_config: MLConfig = field(default_factory=lambda: MLConfig(enable=False))
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    safety_windows: Optional[List[Tuple[int, int, str]]] = None  # [(start,end,"SC"/"VSC"), ...]
    tyre_params: Dict[str, Any] = None  # optional override


# ------------------------------
# Utilities
# ------------------------------

def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _race_driver_filter(df: pd.DataFrame, race_id: int, driver_id: int) -> pd.DataFrame:
    return df[(df["raceId"] == race_id) & (df["driverId"] == driver_id)].copy()


# ------------------------------
# Core simulator
# ------------------------------

def simulate_strategy(
    year: int,
    race_name: str,
    driver_name: str,
    data: Dict[str, pd.DataFrame],
    stints: List[StintSpec],
    cfg: SimulationConfig
) -> Dict[str, Any]:
    """
    Core simulator. Returns:
      - 'lap_times_original': DataFrame of original laps (lap, ms, sec)
      - 'lap_times_simulated': DataFrame of simulated laps (same shape)
      - 'predicted_position': int
      - 'original_position': int (if available)
      - 'fig': Plotly figure (if plotly installed), else None
      - 'details': dict with totals, deltas, modelling notes
    """
    # Resolve raceId, driverId
    races = data["races"]
    drivers = data["drivers"]
    results = _ensure_numeric(data["results"], ["milliseconds", "positionOrder"])
    laps = _ensure_numeric(data["lap_times"], ["milliseconds", "lap"])
    pits = _ensure_numeric(data.get("pit_stops", pd.DataFrame()), ["milliseconds", "lap"])

    race_row = races[(races["year"] == year) & (races["name"].str.lower() == race_name.lower())]
    if race_row.empty:
        return {"error": f"Race '{race_name}' in {year} not found."}
    race_id = int(race_row["raceId"].iloc[0])

    dr = drivers[(drivers["forename"] + " " + drivers["surname"]).str.lower() == driver_name.lower()]
    if dr.empty:
        return {"error": f"Driver '{driver_name}' not found."}
    driver_id = int(dr["driverId"].iloc[0])

    dlap = _race_driver_filter(laps, race_id, driver_id).sort_values("lap")
    if dlap.empty:
        return {"error": "No lap times for selected driver/race."}
    dlap["lap"] = dlap["lap"].astype(int)

    # Baseline totals and position
    original_total_ms = dlap["milliseconds"].sum()
    rrow = results[(results["raceId"] == race_id) & (results["driverId"] == driver_id)]
    original_pos = int(rrow["positionOrder"].iloc[0]) if not rrow.empty and not np.isnan(rrow["positionOrder"].iloc[0]) else None

    # Build per-lap features for ML and effects
    max_lap = int(dlap["lap"].max())
    tyre_params = cfg.tyre_params or DEFAULT_TYRE_PARAMS

    # Convert stints list into a mapping from lap -> (is_pit, duration_ms, compound)
    # Stints are defined by the lap on which pit happens. Tyre compound applies to laps AFTER the pit (new tyres).
    pit_map: Dict[int, Tuple[bool, int, str]] = {}
    for s in stints or []:
        pit_map[int(s.pit_lap)] = (True, int(round(s.duration_sec * 1000)), s.compound)

    # Determine stint_lap counter per lap, starting from first lap with "pre-race" compound inferred
    # as from first entry if present, else Medium. We assume the driver starts on the first stint's compound.
    first_comp = stints[0].compound if stints else "Medium"
    current_comp = first_comp
    stint_lap_counter = 0
    pit_count_so_far = 0

    # Compute other drivers totals for eventual predicted position
    race_laps = laps[laps["raceId"] == race_id]
    other_totals = (race_laps.groupby("driverId")["milliseconds"].sum().reset_index())
    other_totals["milliseconds"] = pd.to_numeric(other_totals["milliseconds"], errors="coerce")

    # Baseline rank map by final totals (proxy): rank lower total as better
    baseline_rank = other_totals.sort_values("milliseconds").reset_index(drop=True)
    baseline_rank["rank"] = baseline_rank.index + 1
    baseline_place_of_me = int(baseline_rank[baseline_rank["driverId"] == driver_id]["rank"].iloc[0]) if (baseline_rank["driverId"] == driver_id).any() else None

    # Safety windows
    windows = cfg.safety_windows or []

    # ML preparation (if enabled)
    ml = LapTimeML(cfg.ml_config)
    if cfg.use_ml:
        # Rough model: use all laps in this race (could be expanded to season-wide)
        train = race_laps.copy()
        train = train.dropna(subset=["milliseconds", "lap"])
        train["lap"] = train["lap"].astype(int)
        # Approximate features (no compound unless provided historically)
        train["stint_lap"] = train["lap"]  # proxy
        train["pit_count_so_far"] = 0

        # mark SC/VSC
        def in_window(l):
            sc = vsc = 0
            for s, e, t in windows:
                if s <= l <= e:
                    if t == "SC":
                        sc = 1
                    if t == "VSC":
                        vsc = 1
            return sc, vsc

        scv = train["lap"].apply(in_window)
        train["is_sc"] = [int(a[0]) for a in scv]
        train["is_vsc"] = [int(a[1]) for a in scv]
        train["y"] = pd.to_numeric(train["milliseconds"], errors="coerce") / 1000.0
        train = train.dropna(subset=["y"])
        ml.fit(train)

    # Simulation loop
    simulated_ms = []
    stint_info = []
    rejoin_penalty_window = 0
    rejoin_rank_delta = 0

    for lap in range(1, max_lap + 1):
        base_ms = int(dlap.loc[dlap["lap"] == lap, "milliseconds"].iloc[0])

        # Apply SC/VSC lap-time multiplier for this lap
        lt_mult = laptime_multiplier(lap, windows)

        # Check if pit at this lap (apply pit delta and change tyres AFTER lap time is counted)
        is_pit = lap in pit_map
        pit_add_ms = 0
        if is_pit:
            _is_pit_flag, pit_nominal_ms, next_comp = pit_map[lap]
            pit_mult = pitdelta_multiplier(lap, windows)
            pit_add_ms = int(round(pit_nominal_ms * pit_mult))

        # Tyre adjustment: increases with stint_lap_counter (before pit change takes effect)
        if cfg.use_ml and ml.model is not None:
            feat = pd.DataFrame([{
                "lap": lap,
                "stint_lap": stint_lap_counter if stint_lap_counter > 0 else 1,
                "pit_count_so_far": pit_count_so_far,
                "is_sc": 1 if lt_mult < 1.0 and lt_mult <= 0.82 else 0,
                "is_vsc": 1 if lt_mult < 1.0 and lt_mult > 0.82 else 0,
                "y": base_ms / 1000.0
            }])
            pred_sec = float(ml.predict(feat)[0])
            adj_ms = int(round(pred_sec * 1000)) - base_ms
        else:
            # deterministic tyre decay model
            adj_sec = stint_adjustment_seconds(stint_lap_counter, current_comp, cfg.tyre_params)
            adj_ms = int(round(adj_sec * 1000))

        # Traffic penalty (if applicable)
        traffic_penalty_ms = 0
        if rejoin_penalty_window > 0:
            # index within window = max_laps_after_rejoin - remaining
            lap_in_window = cfg.traffic.max_laps_after_rejoin - rejoin_penalty_window
            traffic_penalty_sec = compute_traffic_penalty(rejoin_rank_delta, lap_in_window, cfg.traffic)
            traffic_penalty_ms = int(round(traffic_penalty_sec * 1000))
            rejoin_penalty_window -= 1

        # Lap time with all effects
        lap_ms = int(round(base_ms * lt_mult)) + adj_ms + traffic_penalty_ms

        # Append this lap
        simulated_ms.append(lap_ms)

        # If pit this lap, apply pit add and update stint state
        if is_pit:
            simulated_ms[-1] += pit_add_ms
            pit_count_so_far += 1

            # Approximate rejoin rank delta: compare our cumulative sum to others' final totals (proxy)
            my_partial = sum(simulated_ms)
            worse_than = (other_totals["milliseconds"] < my_partial).sum()
            simulated_rank_now = worse_than + 1
            if baseline_place_of_me is not None:
                rejoin_rank_delta = max(0, simulated_rank_now - baseline_place_of_me)
            else:
                rejoin_rank_delta = 0

            # Start traffic window if worsened
            if cfg.traffic.enable and rejoin_rank_delta >= cfg.traffic.trigger_places_delta:
                rejoin_penalty_window = cfg.traffic.max_laps_after_rejoin

            # Change tyres for next laps
            current_comp = next_comp
            stint_lap_counter = 0  # reset after pit
        else:
            stint_lap_counter += 1

        # Save info for debugging/inspection
        stint_info.append({
            "lap": lap,
            "compound": current_comp,
            "stint_lap": stint_lap_counter,
            "is_pit": bool(is_pit),
            "pit_add_ms": pit_add_ms,
            "lt_mult": lt_mult,
            "adj_ms": adj_ms,
            "traffic_ms": traffic_penalty_ms
        })

    simulated_total_ms = int(sum(simulated_ms))

    # Predicted finishing position by comparing totals
    predicted_pos = int((other_totals["milliseconds"] < simulated_total_ms).sum() + 1)

    # Optional plot
    fig = None
    try:
        import plotly.graph_objects as go
        x = list(range(1, max_lap + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=[int(dlap.loc[dlap["lap"] == i, "milliseconds"].iloc[0]) for i in x],
            name="Original Lap",
            mode="lines"
        ))
        fig.add_trace(go.Scatter(x=x, y=simulated_ms, name="Simulated Lap", mode="lines"))
        # pit markers
        for s in stints or []:
            fig.add_vline(
                x=int(s.pit_lap),
                line_dash="dot",
                annotation_text=f"Pit ({s.compound})",
                annotation_position="top"
            )
        fig.update_layout(title="Original vs Simulated Lap Times",
                          xaxis_title="Lap", yaxis_title="Milliseconds")
    except Exception:
        fig = None

    return {
        "lap_times_original": dlap[["lap", "milliseconds"]].assign(seconds=lambda d: d["milliseconds"] / 1000.0),
        "lap_times_simulated": pd.DataFrame({"lap": list(range(1, max_lap + 1)),
                                             "milliseconds": simulated_ms}).assign(seconds=lambda d: d["milliseconds"] / 1000.0),
        "predicted_position": predicted_pos,
        "original_position": original_pos,
        "details": {
            "original_total_ms": int(original_total_ms),
            "simulated_total_ms": int(simulated_total_ms),
            "delta_ms": int(simulated_total_ms - original_total_ms),
            "stint_info": stint_info
        },
        "fig": fig
    }




# import pandas as pd
# import plotly.graph_objects as go

# def simulate_strategy(results_df, lap_times_df, pit_df, year, race_name, driver_name, pit_laps, pit_durations, weather):
#     """
#     Simulate a race strategy for a given driver and race by adjusting lap times based on pit stops.
#     Returns a dict with original & simulated lap times and predicted finish.
#     """
#     # Identify raceId and driverId from inputs
#     race = results_df[(results_df['year'] == year) & (results_df['raceName'] == race_name)]
#     if race.empty:
#         return {"error": "Race not found."}
#     race_id = race.iloc[0]['raceId']
#     driver_row = results_df[(results_df['raceId']==race_id) & (results_df['fullName']==driver_name)]
#     if driver_row.empty:
#         return {"error": "Driver not found in this race."}
#     driver_id = driver_row.iloc[0]['driverId']
    
#     # Get actual lap times (milliseconds)
#     lap_data = lap_times_df[(lap_times_df['raceId']==race_id) & (lap_times_df['driverId']==driver_id)].sort_values('lap')
#     original_laps = lap_data['milliseconds'].tolist()
#     if not original_laps:
#         return {"error": "No lap time data available for this driver/race."}
    
#     # Get original pit stops info
#     orig_pits = pit_df[(pit_df['raceId']==race_id) & (pit_df['driverId']==driver_id)]
#     orig_stop_count = len(orig_pits)
    
#     # Simulate lap times: simple heuristic adjustments
#     new_stop_count = len(pit_laps)
#     # Clone original lap times
#     simulated_laps = original_laps.copy()
#     # If more stops, assume fresh tires give faster laps; if fewer, slower laps
#     lap_adjust = -500 if new_stop_count > orig_stop_count else 500 if new_stop_count < orig_stop_count else 0
#     simulated_laps = [max(0, t + lap_adjust) for t in simulated_laps]
    
#     # Incorporate pit durations: simply add to total time
#     # Compute totals
#     original_total = sum(original_laps) + orig_pits['milliseconds'].sum()
#     simulated_total = sum(simulated_laps) + sum([d*1000 for d in pit_durations])
    
#     # Predict finishing position by comparing to others in the same race
#     race_results = results_df[results_df['raceId'] == race_id].dropna(subset=['milliseconds'])
#     race_results['milliseconds'] = pd.to_numeric(race_results['milliseconds'], errors='coerce')
#     higher_count = (race_results['milliseconds'] < simulated_total).sum()
#     predicted_pos = int(higher_count) + 1
#     actual_pos = int(driver_row.iloc[0]['position'])
    
#     # Build comparison plot
#     laps_range = list(range(1, len(original_laps)+1))
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=laps_range, y=original_laps, mode='lines+markers',
#                              name='Original Lap Time', line=dict(color='blue')))
#     fig.add_trace(go.Scatter(x=laps_range, y=simulated_laps, mode='lines+markers',
#                              name='Simulated Lap Time', line=dict(color='red')))
#     fig.update_layout(
#         title=f"Lap Time Comparison for {driver_name} ({year} {race_name})",
#         xaxis_title="Lap", yaxis_title="Lap Time (ms)",
#         legend_title="Legend"
#     )
    
#     return {
#         "fig": fig,
#         "original_total": original_total,
#         "simulated_total": simulated_total,
#         "actual_position": actual_pos,
#         "predicted_position": predicted_pos
#     }

# # Example usage:
# # sim = simulate_strategy(results_full, lap_df, pit_df, 2020, "Australian Grand Prix", "Max Verstappen", [20, 35], [2.5, 3.0], "Dry")
# # sim["fig"], sim["predicted_position"], sim["actual_position"]
