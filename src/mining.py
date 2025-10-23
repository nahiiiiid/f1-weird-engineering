import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def compute_correlations(results_df, pit_df):
    """
    Compute Pearson correlation matrix between key race features.
    Includes grid position, finish position, and pit stop metrics.
    """
    import numpy as np

    # Aggregate pit stop data per driver-race
    pit_agg = pit_df.groupby(['raceId', 'driverId']).agg(
        stop_count=('stop', 'count'),
        avg_pit_ms=('milliseconds', 'mean')
    ).reset_index()

    # Merge with results to align stop counts with final positions
    merged = results_df.merge(pit_agg, on=['raceId', 'driverId'], how='left')
    merged[['stop_count', 'avg_pit_ms']] = merged[['stop_count', 'avg_pit_ms']].fillna(0)

    # Replace string "\\N" with np.nan
    merged.replace("\\N", np.nan, inplace=True)

    # Convert columns to numeric
    for col in ['grid', 'position', 'stop_count', 'avg_pit_ms']:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')

    # Drop rows with NaNs to avoid correlation issues
    corr_matrix = merged[['grid', 'position', 'stop_count', 'avg_pit_ms']].dropna().corr()

    return corr_matrix


def perform_association_rules(results_df, pit_df, min_support=0.05, min_confidence=0.6):
    """
    Identify frequent patterns relating pit strategies to finishing position.
    Returns association rules from Apriori.
    """
    # Merge results and pit stops to create transactions
    # Prepare one transaction per driver-race
    pit_agg = pit_df.groupby(['raceId', 'driverId']).agg(
        stop_count=('stop', 'count'),
        avg_pit_ms=('milliseconds', 'mean')
    ).reset_index()
    data = results_df.merge(pit_agg, on=['raceId', 'driverId'], how='left')
    data[['stop_count', 'avg_pit_ms']] = data[['stop_count', 'avg_pit_ms']].fillna(0)
    
    transactions = []
    for _, row in data.iterrows():
        items = []
        # Number of stops
        stops = int(row['stop_count'])
        items.append(f"stops_{stops}")
        # Fast vs slow pit category (threshold ~2.5s)
        if row['avg_pit_ms'] > 0:
            if row['avg_pit_ms'] < 2500:
                items.append("fast_pit")
            else:
                items.append("slow_pit")
        # Race outcome category
        if row['position'] <= 3:
            items.append("top3_finish")
        elif row['position'] <= 10:
            items.append("top10_finish")
        # Qualifying advantage
        if 'qualifyingPosition' in row and row['qualifyingPosition'] == 1:
            items.append("pole_position")
        transactions.append(items)
    
    # Convert transactions to one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    # Run Apriori
    freq_items = apriori(df_trans, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
    return rules

def lap_time_trend(results_df, lap_times_df):
    """
    Analyze lap time evolution: average lap time of race winners by year.
    Returns a DataFrame (year, avg_lap_sec).
    """
    # Filter winning drivers
    winners = results_df[results_df['position'] == 1][['raceId', 'driverId', 'year']]

    # Join only necessary columns to avoid column overwriting
    lap_times_df = lap_times_df[['raceId', 'driverId', 'milliseconds']]

    # Merge winners with their lap times
    merged = pd.merge(winners, lap_times_df, on=['raceId', 'driverId'], how='left')

    # Check if milliseconds column exists post-merge
    if 'milliseconds' not in merged.columns:
        raise KeyError("'milliseconds' column not found after merging lap_times")

    # Convert to seconds
    merged['lap_sec'] = merged['milliseconds'] / 1000.0

    # Compute average per year
    trend = merged.groupby('year')['lap_sec'].mean().reset_index(name='avg_lap_sec')
    return trend



# Example usage:
# corr = compute_correlations(results_full, pit_df)
# rules = perform_association_rules(results_full, pit_df)
# trend_df = lap_time_trend(results_full, lap_df)
