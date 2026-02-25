import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_container_anomaly_timeseries(hvac_df, anomaly_type="amplitude", num_containers=1):
    """
    Sample random containers for a given anomaly type and plot their time series.

    Parameters:
    - hvac_df: DataFrame with HVAC data
    - anomaly_type: str, one of 'amplitude', 'frequency', 'lag', 'normal'
    - num_containers: int, number of containers to sample (default: 1)

    Returns: Plotly figure with subplots if num_containers > 1
    """
    # Sample random containers for anomaly type
    containers = hvac_df[hvac_df["anomaly_type"] == anomaly_type]["container_id"].unique()
    sampled_containers = np.random.choice(
        containers, size=min(num_containers, len(containers)), replace=False
    )

    # Create subplots if multiple containers
    if num_containers > 1:
        # Create color mapping for units (uniform across containers)
        all_units = sorted(hvac_df["unit"].unique())
        colors = px.colors.qualitative.Plotly
        unit_colors = {
            unit: colors[i % len(colors)] for i, unit in enumerate(all_units)
        }

        fig = make_subplots(
            rows=num_containers,
            cols=1,
            subplot_titles=[f"Container {cid}" for cid in sampled_containers],
            specs=[[{"secondary_y": True}] for _ in range(num_containers)],
            vertical_spacing=0.1,
        )

        for row_idx, container_id in enumerate(sampled_containers, start=1):
            df_container = hvac_df[hvac_df["container_id"] == container_id].copy()
            df_container = df_container.sort_values("timestamp_et")

            # Add time series by unit
            for unit in df_container["unit"].unique():
                unit_data = df_container[df_container["unit"] == unit]
                fig.add_trace(
                    go.Scatter(
                        x=unit_data["timestamp_et"],
                        y=unit_data["TmpRet"],
                        name=f"Unit {unit}",
                        mode="lines",
                        line=dict(color=unit_colors[unit]),
                        showlegend=(row_idx == 1),
                        legendgroup=f"unit_{unit}",
                    ),
                    row=row_idx,
                    col=1,
                    secondary_y=False,
                )

            # Add anomaly flag on secondary y-axis
            if anomaly_type in ["lag", "frequency"]:
                for unit in df_container["unit"].unique():
                    unit_anomaly_data = df_container[
                        (df_container["unit"] == unit) & df_container["anomaly"]
                    ]
                    # Only add to legend if anomalies exist for this unit in this container
                    has_anomalies = len(unit_anomaly_data) > 0
                    fig.add_trace(
                        go.Scatter(
                            x=unit_anomaly_data["timestamp_et"],
                            y=unit_anomaly_data["anomaly"].astype(int),
                            name=f"Container {container_id} - Anomaly - Unit {unit}",
                            mode="markers",
                            marker=dict(size=5, color=unit_colors[unit]),
                            showlegend=has_anomalies,
                            legendgroup=f"anomaly_unit_{unit}",
                        ),
                        row=row_idx,
                        col=1,
                        secondary_y=True,
                    )
            else:
                # Check if this container has any anomalies
                has_anomalies = df_container["anomaly"].any()
                fig.add_trace(
                    go.Scatter(
                        x=df_container["timestamp_et"],
                        y=df_container["anomaly"].astype(int),
                        name=f"Container {container_id} - Anomaly Flag",
                        mode="markers",
                        marker=dict(size=3, color="red"),
                        showlegend=bool(has_anomalies),
                        legendgroup="anomaly_flag",
                    ),
                    row=row_idx,
                    col=1,
                    secondary_y=True,
                )

        # Update axes labels
        for row_idx in range(1, num_containers + 1):
            fig.update_yaxes(
                title_text="Temperature (TmpRet)", row=row_idx, secondary_y=False
            )
            fig.update_yaxes(
                title_text="Anomaly Flag", row=row_idx, secondary_y=True, range=[-0.1, 1.1]
            )
            fig.update_xaxes(title_text="Timestamp", row=row_idx)

        fig.update_layout(
            title_text=f"Containers - {anomaly_type.capitalize()} Anomaly Type",
            hovermode="x unified",
            height=300 * num_containers,
        )
    else:
        # Single container (original behavior)
        container_id = sampled_containers[0]
        df_container = hvac_df[hvac_df["container_id"] == container_id].copy()
        df_container = df_container.sort_values("timestamp_et")

        fig = go.Figure()

        # Add time series by unit (primary y-axis)
        for unit in df_container["unit"].unique():
            unit_data = df_container[df_container["unit"] == unit]
            fig.add_trace(
                go.Scatter(
                    x=unit_data["timestamp_et"],
                    y=unit_data["TmpRet"],
                    name=f"Unit {unit}",
                    mode="lines",
                    yaxis="y1",
                )
            )

        # Add anomaly flag on secondary y-axis
        if anomaly_type in ["lag", "frequency"]:
            for unit in df_container["unit"].unique():
                unit_anomaly_data = df_container[
                    (df_container["unit"] == unit) & df_container["anomaly"]
                ]
                fig.add_trace(
                    go.Scatter(
                        x=unit_anomaly_data["timestamp_et"],
                        y=unit_anomaly_data["anomaly"].astype(int),
                        name=f"Anomaly - Unit {unit}",
                        mode="markers",
                        marker=dict(size=5),
                        yaxis="y2",
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df_container["timestamp_et"],
                    y=df_container["anomaly"].astype(int),
                    name="Anomaly Flag",
                    mode="markers",
                    marker=dict(size=3, color="red"),
                    yaxis="y2",
                )
            )

        fig.update_layout(
            title=f"Container {container_id} - {anomaly_type.capitalize()} Anomaly Type",
            xaxis_title="Timestamp",
            yaxis=dict(title="Temperature (TmpRet)", side="left"),
            yaxis2=dict(title="Anomaly Flag", side="right", overlaying="y", range=[-0.1, 1.1]),
            hovermode="x unified",
            height=300,
        )
        

    return fig


def plot_flagged_vs_true(hvac_df, scores_df, anomaly_type, num_containers=2):
    """
    Compare model-flagged anomalies against ground-truth labels for sampled containers.

    Samples containers that were flagged by the algorithm for the given anomaly type,
    then plots raw temperatures with true labels and shaded day-level flags.

    Parameters:
    - hvac_df: raw HVAC dataset (columns: timestamp_et, container_id, unit, TmpRet, anomaly, anomaly_type)
    - scores_df: output of flag_anomalies() (columns: container_id, day, unit, anomaly_score, anomaly_flag, model)
    - anomaly_type: one of "lag", "frequency", "amplitude"
    - num_containers: how many containers to sample (default 2)

    Returns: Plotly figure
    """
    import datetime

    unit_colors = {0: '#5470C6', 1: '#EE6666', 2: '#5DBCD2'}

    # Containers flagged by the model
    flagged_containers = scores_df[scores_df["anomaly_flag"]]["container_id"].unique()
    # Containers with this anomaly type in ground truth
    type_containers = hvac_df[hvac_df["anomaly_type"] == anomaly_type]["container_id"].unique()

    # Intersect: flagged AND have this anomaly type
    candidates = np.intersect1d(flagged_containers, type_containers)

    if len(candidates) == 0:
        print(f"No containers flagged for anomaly type '{anomaly_type}'. Showing unflagged samples.")
        candidates = type_containers

    n = min(num_containers, len(candidates))
    sampled = np.random.choice(candidates, size=n, replace=False)

    # Build subplot titles with flagged unit info
    titles = []
    for cid in sampled:
        cid_flags = scores_df[
            (scores_df["container_id"] == cid) & scores_df["anomaly_flag"]
        ]
        flagged_units = sorted(cid_flags["unit"].unique())
        unit_str = ", ".join(str(u) for u in flagged_units) if len(flagged_units) > 0 else "none"
        titles.append(f"Container {cid} — {anomaly_type} | flagged unit(s): {unit_str}")

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=titles,
        specs=[[{"secondary_y": True}] for _ in range(n)],
        vertical_spacing=0.15,
    )

    for row_idx, cid in enumerate(sampled, start=1):
        show_legend = True
        df_c = hvac_df[hvac_df["container_id"] == cid].sort_values("timestamp_et")
        sf_c = scores_df[scores_df["container_id"] == cid]

        for unit in sorted(df_c["unit"].unique()):
            color = unit_colors.get(unit, '#999999')
            ud = df_c[df_c["unit"] == unit]

            # Raw temperature lines
            fig.add_trace(
                go.Scatter(
                    x=ud["timestamp_et"], y=ud["TmpRet"],
                    name=f"Unit {unit}", mode="lines",
                    line=dict(color=color),
                    legendgroup=f"c{cid}",
                    showlegend=show_legend,
                ),
                row=row_idx, col=1, secondary_y=False,
            )

            # True labels (circle markers)
            true_pts = ud[ud["anomaly"]]
            if len(true_pts) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=true_pts["timestamp_et"],
                        y=true_pts["anomaly"].astype(int),
                        name=f"True — Unit {unit}", mode="markers",
                        marker=dict(symbol="circle", size=5, color='green'),
                        legendgroup=f"c{cid}",
                        showlegend=show_legend,
                    ),
                    row=row_idx, col=1, secondary_y=True,
                )

        # Shade flagged days per unit
        flagged_days = sf_c[sf_c["anomaly_flag"]]
        for _, row in flagged_days.iterrows():
            day = row["day"]
            unit = row["unit"]
            color = unit_colors.get(unit, '#999999')
            day_start = datetime.datetime.combine(day, datetime.time.min)
            day_end = datetime.datetime.combine(day, datetime.time.max)
            fig.add_vrect(
                x0=day_start, x1=day_end,
                fillcolor=color, opacity=0.15,
                line_width=0,
                row=row_idx, col=1,
            )

    for row_idx in range(1, n + 1):
        fig.update_yaxes(title_text="Temperature (TmpRet)", row=row_idx, secondary_y=False)
        fig.update_yaxes(title_text="Anomaly", row=row_idx, secondary_y=True, range=[-0.1, 1.1])
        fig.update_xaxes(title_text="Timestamp", row=row_idx)

    fig.update_layout(
        title_text=f"Euclidean Distance Flagging — {anomaly_type.capitalize()} Anomaly",
        hovermode="x unified",
        height=350 * n,
        width=1200,
    )

    return fig


def plot_anomaly_type_distribution(hvac_df):
    """
    Visualize percentage of total container-days and container unit-days grouped by anomaly types.

    Parameters:
    - hvac_df: DataFrame with HVAC data

    Returns: Plotly figure with two pie charts side by side
    """
    # Create a container-day identifier (container_id + date from timestamp_et)
    df_copy = hvac_df.copy()
    df_copy["container_day"] = df_copy["container_id"].astype(str) + "_" + df_copy["timestamp_et"].dt.date.astype(str)

    # Count unique container-days per anomaly type
    anomaly_containerdays = df_copy.groupby("anomaly_type")["container_day"].nunique().reset_index()
    anomaly_containerdays.columns = ["anomaly_type", "container_days"]

    # Count unique container unit-days per anomaly type
    anomaly_unitdays = df_copy.groupby("anomaly_type")["cont_unit_day"].nunique().reset_index()
    anomaly_unitdays.columns = ["anomaly_type", "unit_days"]

    # Create subplots with two pie charts
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=("Container-Days", "Unit-Days"),
    )

    # Add container-days pie chart
    fig.add_trace(
        go.Pie(
            labels=anomaly_containerdays["anomaly_type"],
            values=anomaly_containerdays["container_days"],
            name="Container-Days",
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Container-Days: %{value}<br>Percentage: %{percent}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add unit-days pie chart
    fig.add_trace(
        go.Pie(
            labels=anomaly_unitdays["anomaly_type"],
            values=anomaly_unitdays["unit_days"],
            name="Unit-Days",
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Unit-Days: %{value}<br>Percentage: %{percent}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text="Distribution of Anomaly Types by Container-Days and Unit-Days",
        height=500,
    )

    return fig
