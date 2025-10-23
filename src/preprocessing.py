import pandas as pd

def preprocess_data(data):
    """
    Merge and clean Formula 1 dataset tables into a unified DataFrame for analysis.
    Returns:
        - merged results DataFrame with driver, constructor, race, and circuit info
        - raw pit stop DataFrame
        - raw lap time DataFrame
    """
    # Load individual dataframes
    races = data.get('races', pd.DataFrame())
    results = data.get('results', pd.DataFrame())
    drivers = data.get('drivers', pd.DataFrame())
    constructors = data.get('constructors', pd.DataFrame())
    circuits = data.get('circuits', pd.DataFrame())
    qualifying = data.get('qualifying', None)
    pit_stops = data.get('pit_stops', pd.DataFrame())
    lap_times = data.get('lap_times', pd.DataFrame())

    # Confirm necessary columns exist
    if 'name' not in races.columns:
        raise KeyError("'name' column (race name) is missing from races.csv")

    # Rename race name to avoid collision
    races = races.rename(columns={'name': 'raceName'})

    # Merge race info into results
    merged = results.merge(
        races[['raceId', 'year', 'raceName', 'circuitId']],
        on='raceId', how='left'
    )

    # Rename circuit name to avoid conflict
    circuits_small = circuits[['circuitId', 'name']].rename(columns={'name': 'circuitName'})
    merged = merged.merge(circuits_small, on='circuitId', how='left')

    # Add full driver name
    drivers['fullName'] = drivers['forename'] + ' ' + drivers['surname']
    merged = merged.merge(
        drivers[['driverId', 'fullName']],
        on='driverId', how='left'
    )

    # Add constructor name, handle conflict
    constructors_small = constructors[['constructorId', 'name']].rename(columns={'name': 'constructorName'})
    merged = merged.merge(constructors_small, on='constructorId', how='left')

    # Merge qualifying info (optional)
    if qualifying is not None and not qualifying.empty:
        qual = qualifying[['raceId', 'driverId', 'position']].rename(columns={'position': 'qualifyingPosition'})
        merged = merged.merge(qual, on=['raceId', 'driverId'], how='left')

    # Replace invalid strings and clean
    merged.replace('\\N', pd.NA, inplace=True)

    # Convert important columns to numeric
    for col in ['grid', 'position', 'points']:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    # Drop rows with missing core data
    merged.dropna(subset=['position', 'grid'], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    return merged, pit_stops, lap_times

# Example usage:
# from src.data_loader import load_data
# data = load_data()
# results_full, pit_df, lap_df = preprocess_data(data)
