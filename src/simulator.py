import pandas as pd
import plotly.graph_objects as go

def simulate_strategy(results_df, lap_times_df, pit_df, year, race_name, driver_name, pit_laps, pit_durations, weather):
    """
    Simulate a race strategy for a given driver and race by adjusting lap times based on pit stops.
    Returns a dict with original & simulated lap times and predicted finish.
    """
    # Identify raceId and driverId from inputs
    race = results_df[(results_df['year'] == year) & (results_df['raceName'] == race_name)]
    if race.empty:
        return {"error": "Race not found."}
    race_id = race.iloc[0]['raceId']
    driver_row = results_df[(results_df['raceId']==race_id) & (results_df['fullName']==driver_name)]
    if driver_row.empty:
        return {"error": "Driver not found in this race."}
    driver_id = driver_row.iloc[0]['driverId']
    
    # Get actual lap times (milliseconds)
    lap_data = lap_times_df[(lap_times_df['raceId']==race_id) & (lap_times_df['driverId']==driver_id)].sort_values('lap')
    original_laps = lap_data['milliseconds'].tolist()
    if not original_laps:
        return {"error": "No lap time data available for this driver/race."}
    
    # Get original pit stops info
    orig_pits = pit_df[(pit_df['raceId']==race_id) & (pit_df['driverId']==driver_id)]
    orig_stop_count = len(orig_pits)
    
    # Simulate lap times: simple heuristic adjustments
    new_stop_count = len(pit_laps)
    # Clone original lap times
    simulated_laps = original_laps.copy()
    # If more stops, assume fresh tires give faster laps; if fewer, slower laps
    lap_adjust = -500 if new_stop_count > orig_stop_count else 500 if new_stop_count < orig_stop_count else 0
    simulated_laps = [max(0, t + lap_adjust) for t in simulated_laps]
    
    # Incorporate pit durations: simply add to total time
    # Compute totals
    original_total = sum(original_laps) + orig_pits['milliseconds'].sum()
    simulated_total = sum(simulated_laps) + sum([d*1000 for d in pit_durations])
    
    # Predict finishing position by comparing to others in the same race
    race_results = results_df[results_df['raceId'] == race_id].dropna(subset=['milliseconds'])
    race_results['milliseconds'] = pd.to_numeric(race_results['milliseconds'], errors='coerce')
    higher_count = (race_results['milliseconds'] < simulated_total).sum()
    predicted_pos = int(higher_count) + 1
    actual_pos = int(driver_row.iloc[0]['position'])
    
    # Build comparison plot
    laps_range = list(range(1, len(original_laps)+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=laps_range, y=original_laps, mode='lines+markers',
                             name='Original Lap Time', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=laps_range, y=simulated_laps, mode='lines+markers',
                             name='Simulated Lap Time', line=dict(color='red')))
    fig.update_layout(
        title=f"Lap Time Comparison for {driver_name} ({year} {race_name})",
        xaxis_title="Lap", yaxis_title="Lap Time (ms)",
        legend_title="Legend"
    )
    
    return {
        "fig": fig,
        "original_total": original_total,
        "simulated_total": simulated_total,
        "actual_position": actual_pos,
        "predicted_position": predicted_pos
    }

# Example usage:
# sim = simulate_strategy(results_full, lap_df, pit_df, 2020, "Australian Grand Prix", "Max Verstappen", [20, 35], [2.5, 3.0], "Dry")
# sim["fig"], sim["predicted_position"], sim["actual_position"]
