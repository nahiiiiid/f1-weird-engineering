import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.mining import compute_correlations, perform_association_rules, lap_time_trend
from src.visualization import create_pit_scatter, create_lap_trend_line, create_heatmap
from src.simulator import simulate_strategy

st.set_page_config(page_title="F1 Strategic Race Simulator", layout="wide")
st.title("🏁 F1 Strategic Race Simulator")

# Load and preprocess data
data = load_data()
results_full, pit_df, lap_df = preprocess_data(data)

# Create tabs
tab1, tab2 = st.tabs(["Data Insights", "Simulation"])

with tab1:
    st.header("Pit Stop & Lap Time Analysis")
    
    # Pit Stop scatter plot
    fig_scatter = create_pit_scatter(pit_df, results_full)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.write("**Figure:** Each point is a pit stop (lap vs duration), colored by the driver’s final position. Fast, efficient pit stops often correlate with better finishes.")
    
    # Lap time trend chart
    trend_df = lap_time_trend(results_full, lap_df)
    fig_line = create_lap_trend_line(trend_df)
    st.plotly_chart(fig_line, use_container_width=True)
    st.write("**Figure:** Average winner lap time over the years. The downward trend shows performance improvements over decades.")
    
    # Correlation heatmap
    corr_matrix = compute_correlations(results_full, pit_df)
    fig_heatmap = create_heatmap(corr_matrix)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.write("**Figure:** Pearson correlation between features: grid position, final position, number of pit stops, and average pit duration.")

with tab2:
    st.header("Race Strategy Simulator")
    
    # User inputs
    year = st.selectbox("Select Year", sorted(results_full['year'].unique()))
    
    race_names = results_full[results_full['year'] == year]['raceName'].unique()
    race_name = st.selectbox("Select Grand Prix", sorted(race_names))
    
    drivers = results_full[(results_full['year'] == year) & (results_full['raceName'] == race_name)]['fullName'].unique()
    driver = st.selectbox("Select Driver", sorted(drivers))
    
    pit_count = st.slider("Number of Pit Stops", 0, 3, value=1)
    pit_laps = []
    pit_durations = []
    for i in range(1, pit_count + 1):
        lap = st.number_input(f"Lap of Pit Stop {i}", min_value=1, value=20, key=f"lap_{i}")
        dur = st.number_input(f"Pit Stop {i} Duration (sec)", min_value=1.0, value=2.0, key=f"dur_{i}")
        pit_laps.append(int(lap))
        pit_durations.append(float(dur))
    
    weather = st.selectbox("Weather Condition", ["Dry", "Wet"])
    
    # Run simulation
    sim_result = simulate_strategy(
        results_full, lap_df, pit_df,
        year, race_name, driver,
        pit_laps, pit_durations, weather
    )
    
    if "error" in sim_result:
        st.error(sim_result["error"])
    else:
        st.plotly_chart(sim_result["fig"], use_container_width=True)
        st.write(f"Original Finish: **{sim_result['actual_position']}**")
        st.write(f"Predicted Finish: **{sim_result['predicted_position']}**")
        st.write(f"Total Time (actual): {sim_result['original_total'] / 1000:.1f} s")
        st.write(f"Total Time (simulated): {sim_result['simulated_total'] / 1000:.1f} s")
