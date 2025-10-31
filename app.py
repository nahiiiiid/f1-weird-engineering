import streamlit as st
import pandas as pd
import numpy as np

from src.simulator import simulate_strategy, StintSpec, SimulationConfig
from src.tyre_models import DEFAULT_TYRE_PARAMS, TyreParams
from src.safety_car import laptime_multiplier, pitdelta_multiplier
from src.traffic import TrafficConfig
from src.ml_predictor import MLConfig
from src.validation import replay_actual

# Minimal data loader expecting your original CSVs in ./data/
@st.cache_data
def load_all():
    dfs = {}
    for name in ["races","results","lap_times","pit_stops","drivers","constructors","circuits"]:
        try:
            dfs[name] = pd.read_csv(f"data/{name}.csv")
        except Exception:
            dfs[name] = pd.DataFrame()
    return dfs

st.set_page_config(page_title="F1 Strategy Lab", layout="wide")
st.title("F1 Strategy Lab - Weird Simulations")

data = load_all()
if data["races"].empty or data["lap_times"].empty or data["results"].empty or data["drivers"].empty:
    st.error("Missing core CSVs (races, lap_times, results, drivers). Please place them under ./data/")
    st.stop()

# Sidebar: selection
years = sorted(data["races"]["year"].unique().tolist())
year = st.sidebar.selectbox("Year", years, index=max(0, len(years)-1))

races_this_year = data["races"][data["races"]["year"] == year]
race_name = st.sidebar.selectbox("Race", sorted(races_this_year["name"].unique().tolist()))

driver_full = (data["drivers"]["forename"] + " " + data["drivers"]["surname"]).tolist()
driver_name = st.sidebar.selectbox("Driver", sorted(driver_full))

st.sidebar.markdown("---")
st.sidebar.subheader("Safety Car / VSC Windows")
sc_windows = []
with st.sidebar.expander("Add SC/VSC windows"):
    num = st.number_input("How many windows?", min_value=0, max_value=10, value=0, step=1)
    for i in range(num):
        c1, c2, c3 = st.columns(3)
        s = c1.number_input(f"Start lap #{i+1}", min_value=1, value=1, step=1, key=f"s{i}")
        e = c2.number_input(f"End lap #{i+1}", min_value=s, value=s, step=1, key=f"e{i}")
        t = c3.selectbox(f"Type #{i+1}", ["SC","VSC"], key=f"t{i}")
        sc_windows.append((int(s), int(e), t))

st.sidebar.markdown("---")
st.sidebar.subheader("Traffic model")
traffic_enable = st.sidebar.checkbox("Enable traffic penalties", value=True)
pen = st.sidebar.number_input("Penalty per lap (s)", min_value=0.0, value=0.25, step=0.05)
kmax = st.sidebar.number_input("Penalty lasts K laps", min_value=0, value=5, step=1)
trigger = st.sidebar.number_input("Trigger if rejoin places worse by ≥", min_value=1, value=1, step=1)
traffic_cfg = TrafficConfig(enable=traffic_enable, penalty_per_lap_sec=float(pen), max_laps_after_rejoin=int(kmax), trigger_places_delta=int(trigger))

st.sidebar.markdown("---")
st.sidebar.subheader("Tyre compounds (per-stint decay params)")
params = {}
for name, tp in DEFAULT_TYRE_PARAMS.items():
    with st.sidebar.expander(f"{name}"):
        base = st.number_input(f"{name} base (s)", value=float(tp.base), step=0.01, key=f"{name}_base")
        a = st.number_input(f"{name} a (s/lap)", value=float(tp.a), step=0.001, key=f"{name}_a")
        b = st.number_input(f"{name} b (s/lap^2)", value=float(tp.b), step=0.0001, format="%.4f", key=f"{name}_b")
        params[name] = TyreParams(base=base, a=a, b=b)

st.sidebar.markdown("---")
st.sidebar.subheader("ML Lap-time predictor")
use_ml = st.sidebar.checkbox("Use ML model", value=False)
if use_ml:
    n_est = st.sidebar.number_input("n_estimators", min_value=10, value=200, step=10)
    depth = st.sidebar.number_input("max_depth (0=auto)", min_value=0, value=0, step=1)
    ml_cfg = MLConfig(enable=True, n_estimators=int(n_est), max_depth=None if depth==0 else int(depth))
else:
    ml_cfg = MLConfig(enable=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Stints & pits")
num_stints = st.sidebar.number_input("Number of pit events", min_value=0, max_value=10, value=2, step=1)
stints = []
for i in range(num_stints):
    c1, c2, c3 = st.sidebar.columns(3)
    lap = c1.number_input(f"Pit lap #{i+1}", min_value=1, value=(20 if i==0 else 40), step=1, key=f"pl{i}")
    dur = c2.number_input(f"Stationary time (s) #{i+1}", min_value=0.0, value=2.5, step=0.1, key=f"pd{i}")
    comp = c3.selectbox(f"Next compound #{i+1}", ["Soft","Medium","Hard","Inter","Wet"], key=f"pc{i}")
    stints.append(StintSpec(pit_lap=int(lap), duration_sec=float(dur), compound=comp))

cfg = SimulationConfig(
    use_ml=use_ml,
    ml_config=ml_cfg,
    traffic=traffic_cfg,
    safety_windows=sc_windows,
    tyre_params=params
)

tab1, tab2 = st.tabs(["🔧 Simulate", "🧪 Validation"])

with tab1:
    st.subheader("Simulate custom strategy")
    if st.button("Run simulation", type="primary"):
        out = simulate_strategy(year, race_name, driver_name, data, stints, cfg)
        if "error" in out:
            st.error(out["error"])
        else:
            st.write(f"**Original pos**: {out['original_position']} — **Predicted pos**: {out['predicted_position']}")
            d = out["details"]
            st.write(f"Original total: {d['original_total_ms']/1000:.3f}s")
            st.write(f"Simulated total: {d['simulated_total_ms']/1000:.3f}s  (Δ {d['delta_ms']/1000:.3f}s)")
            st.dataframe(pd.DataFrame(out["details"]["stint_info"]).head(15))
            if out["fig"] is not None:
                st.plotly_chart(out["fig"], use_container_width=True)
            st.download_button("Download simulated laps (CSV)",
                               data=out["lap_times_simulated"].to_csv(index=False).encode("utf-8"),
                               file_name="simulated_laps.csv", mime="text/csv")

with tab2:
    st.subheader("Replay actual race (validation mode)")
    # Find ids
    races = data["races"]
    drivers = data["drivers"]
    race_id = int(races[(races["year"]==year) & (races["name"].str.lower()==race_name.lower())]["raceId"].iloc[0])
    driver_id = int(drivers[(drivers["forename"] + " " + drivers["surname"]).str.lower()==driver_name.lower()]["driverId"].iloc[0])

    if st.button("Run validation"):
        res = replay_actual(driver_id, race_id, data["lap_times"], data["results"])
        if "error" in res:
            st.error(res["error"])
        else:
            st.json(res)

# can be added tab 3  for real simulation of data insights 


# import streamlit as st
# import pandas as pd
# from src.data_loader import load_data
# from src.preprocessing import preprocess_data
# from src.mining import compute_correlations, perform_association_rules, lap_time_trend
# from src.visualization import create_pit_scatter, create_lap_trend_line, create_heatmap
# from src.simulator import simulate_strategy

# st.set_page_config(page_title="F1 Strategic Race Simulator", layout="wide")
# st.title("🏁 F1 Strategic Race Simulator")

# # Load and preprocess data
# data = load_data()
# results_full, pit_df, lap_df = preprocess_data(data)

# # Create tabs
# tab1, tab2 = st.tabs(["Data Insights", "Simulation"])

# with tab1:
#     st.header("Pit Stop & Lap Time Analysis")
    
#     # Pit Stop scatter plot
#     fig_scatter = create_pit_scatter(pit_df, results_full)
#     st.plotly_chart(fig_scatter, use_container_width=True)
#     st.write("**Figure:** Each point is a pit stop (lap vs duration), colored by the driver’s final position. Fast, efficient pit stops often correlate with better finishes.")
    
#     # Lap time trend chart
#     trend_df = lap_time_trend(results_full, lap_df)
#     fig_line = create_lap_trend_line(trend_df)
#     st.plotly_chart(fig_line, use_container_width=True)
#     st.write("**Figure:** Average winner lap time over the years. The downward trend shows performance improvements over decades.")
    
#     # Correlation heatmap
#     corr_matrix = compute_correlations(results_full, pit_df)
#     fig_heatmap = create_heatmap(corr_matrix)
#     st.plotly_chart(fig_heatmap, use_container_width=True)
#     st.write("**Figure:** Pearson correlation between features: grid position, final position, number of pit stops, and average pit duration.")

# with tab2:
#     st.header("Race Strategy Simulator")
    
#     # User inputs
#     year = st.selectbox("Select Year", sorted(results_full['year'].unique()))
    
#     race_names = results_full[results_full['year'] == year]['raceName'].unique()
#     race_name = st.selectbox("Select Grand Prix", sorted(race_names))
    
#     drivers = results_full[(results_full['year'] == year) & (results_full['raceName'] == race_name)]['fullName'].unique()
#     driver = st.selectbox("Select Driver", sorted(drivers))
    
#     pit_count = st.slider("Number of Pit Stops", 0, 3, value=1)
#     pit_laps = []
#     pit_durations = []
#     for i in range(1, pit_count + 1):
#         lap = st.number_input(f"Lap of Pit Stop {i}", min_value=1, value=20, key=f"lap_{i}")
#         dur = st.number_input(f"Pit Stop {i} Duration (sec)", min_value=1.0, value=2.0, key=f"dur_{i}")
#         pit_laps.append(int(lap))
#         pit_durations.append(float(dur))
    
#     weather = st.selectbox("Weather Condition", ["Dry", "Wet"])
    
#     # Run simulation
#     sim_result = simulate_strategy(
#         results_full, lap_df, pit_df,
#         year, race_name, driver,
#         pit_laps, pit_durations, weather
#     )
    
#     if "error" in sim_result:
#         st.error(sim_result["error"])
#     else:
#         st.plotly_chart(sim_result["fig"], use_container_width=True)
#         st.write(f"Original Finish: **{sim_result['actual_position']}**")
#         st.write(f"Predicted Finish: **{sim_result['predicted_position']}**")
#         st.write(f"Total Time (actual): {sim_result['original_total'] / 1000:.1f} s")
#         st.write(f"Total Time (simulated): {sim_result['simulated_total'] / 1000:.1f} s")
