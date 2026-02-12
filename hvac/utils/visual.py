import numpy as np
import plotly.graph_objects as go
import plotly.express as px


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
        from plotly.subplots import make_subplots

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
            vertical_spacing=0.12,
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
            height=500,
        )

    return fig


def plot_anomaly_type_distribution(hvac_df):
    """
    Visualize percentage of total unit-days grouped by anomaly types.

    Parameters:
    - hvac_df: DataFrame with HVAC data

    Returns: Plotly figure
    """
    # Count unique unit-days per anomaly type
    anomaly_unitdays = hvac_df.groupby("anomaly_type")["cont_unit_day"].nunique().reset_index()
    anomaly_unitdays.columns = ["anomaly_type", "unit_days"]

    # Calculate percentage
    total_unitdays = anomaly_unitdays["unit_days"].sum()
    anomaly_unitdays["percentage"] = (anomaly_unitdays["unit_days"] / total_unitdays * 100).round(2)

    # Create pie chart
    fig = px.pie(
        anomaly_unitdays,
        values="unit_days",
        names="anomaly_type",
        title="Distribution of Unit-Days by Anomaly Type",
        labels={"unit_days": "Unit-Days", "anomaly_type": "Anomaly Type"},
    )

    # Add percentages to labels
    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Unit-Days: %{value}<br>Percentage: %{percent}<extra></extra>",
    )

    return fig
