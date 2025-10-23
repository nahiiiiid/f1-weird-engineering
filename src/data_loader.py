import pandas as pd
from pathlib import Path

def load_data():
    """
    Load all CSV files from the data/ directory into pandas DataFrames.
    Returns a dictionary mapping file stems to DataFrames.
    """
    data_dir = Path(__file__).parent.parent / "data"
    file_list = [
        "races.csv", "results.csv", "pit_stops.csv", "lap_times.csv",
        "drivers.csv", "constructors.csv", "circuits.csv",
        "qualifying.csv", "seasons.csv", "status.csv"
    ]
    data = {}
    for fname in file_list:
        path = data_dir / fname
        if path.exists():
            key = fname.replace('.csv', '')
            df = pd.read_csv(path)
            data[key] = df
        else:
            print(f"Warning: {fname} not found in data/ directory.")
    return data

# Example usage:
# data = load_data()
# races_df = data['races']
# results_df = data['results']
