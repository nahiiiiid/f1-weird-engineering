import plotly.express as px

def create_pit_scatter(pit_df, results_df):
    """
    Create a scatter plot of pit stop lap vs duration, colored by finishing position.
    """
    # Merge pit stops with final position
    df = pit_df.merge(
        results_df[['raceId','driverId','position']],
        on=['raceId','driverId'], how='left'
    )
    fig = px.scatter(
        df, x="lap", y="milliseconds", color="position",
        labels={"lap": "Lap Number", "milliseconds": "Pit Duration (ms)", "position": "Final Position"},
        title="Pit Stop Lap vs Duration (colored by finish position)"
    )
    return fig

def create_lap_trend_line(trend_df):
    """
    Create a line chart of average winner lap time (sec) over years.
    """
    fig = px.line(
        trend_df, x="year", y="avg_lap_sec",
        labels={"year": "Year", "avg_lap_sec": "Avg Lap Time (sec)"},
        title="Trend of Average Winner Lap Time by Year"
    )
    fig.update_traces(mode="lines+markers")
    return fig

def create_heatmap(corr_matrix):
    """
    Create a heatmap figure for a correlation matrix.
    """
    fig = px.imshow(
        corr_matrix,
        text_auto=True, aspect="auto",
        title="Correlation Matrix (Pearson's r)"
    )
    fig.update_layout(xaxis_title="Features", yaxis_title="Features")
    return fig

# Example usage:
# fig1 = create_pit_scatter(pit_df, results_full)
# fig2 = create_lap_trend_line(trend_df)
# fig3 = create_heatmap(corr)
