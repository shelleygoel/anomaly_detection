"""Visualization helpers.

`plot_cases` renders one figure per sampled entity, stacking rows across one or
more stages (raw features, MP profile, anomaly scores) on a shared time axis.

Stages are passed as `PlotConfig` objects — a flat description of "here's a
DataFrame, here are the column names."  Any stage class can expose a
`to_plot_cfg()` method; viz doesn't care about the stage's actual type.

Sampling is not part of viz — call `TimeSeriesDataset.sample_entities()` to
pick the entity IDs first, then pass them in explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class PlotConfig:
    """Flat, typed description of one stage's data for plotting.

    Attributes:
        df: DataFrame with one row per (entity, time) — or per
            (entity, time, sub_entity) when sub_entity_col is set.
        entity_col: Column name for the entity ID.
        time_col: Column name for the time axis.
        value_cols: Columns to plot — each becomes its own subplot row.
        label_col: If set, anomaly markers are drawn from rows where this
            column is truthy.
        sub_entity_col: If set, value traces are split and colored per
            sub-entity.
        label_type_col: If set, used to resolve the dominant anomaly type
            per entity for figure titles and label_type-based sampling.
    """

    df: pd.DataFrame
    entity_col: str
    time_col: str
    value_cols: list[str]
    label_col: str | None = None
    sub_entity_col: str | None = None
    label_type_col: str | None = None


def plot_cases(
    configs: list[PlotConfig],
    entity_ids=None,
    *,
    sample_from=None,
    n_cases: int = 1,
    label_type: str | None = None,
    random_state: int | None = None,
    labels_from: PlotConfig | None = None,
    row_height: int = 120,
    width: int = 800,
) -> list[go.Figure]:
    """Render one figure per entity_id, stacking rows across stages.

    Each cfg contributes one subplot row per value_col.  Rows share the x-axis.
    Labels and sub-entity styling come from the first cfg in `configs` that
    carries the relevant column.

    Pass `entity_ids` for explicit selection, or `sample_from` (a dataset with
    a `.sample_entities(n_cases, label_type, random_state)` method) to let
    plot_cases sample for you.

    Args:
        configs: Non-empty list of PlotConfig.  All configs must share entity_col.
        entity_ids: Iterable of entity IDs to plot.  If None, sampling is used.
        sample_from: Dataset to sample from (uses duck-typed `.sample_entities`).
            Only used when entity_ids is None.
        n_cases: Number of entities to sample (when sampling).
        label_type: Filter sampled entities by label_type (when sampling).
        random_state: Sampling seed (when sampling).

    Returns:
        List of Plotly figures, one per entity_id.
    """
    if not configs:
        raise ValueError("configs must be non-empty")

    if entity_ids is None:
        if sample_from is None:
            raise ValueError("provide either entity_ids or sample_from")
        entity_ids = sample_from.sample_entities(
            n_cases=n_cases, label_type=label_type, random_state=random_state,
        )

    primary = configs[0]
    entity_col = primary.entity_col
    for s in configs[1:]:
        if s.entity_col != entity_col:
            raise ValueError(
                f"all configs must share entity_col; got {s.entity_col!r} vs {entity_col!r}"
            )

    # Labels can be provided explicitly via labels_from (useful when you don't
    # want to plot the raw ds itself), otherwise fall back to the first config
    # in `configs` that carries a label_col.
    label_cfg = labels_from if labels_from is not None else _first_with(configs, "label_col")
    sub_cfg = _first_with(configs, "sub_entity_col")

    colors = px.colors.qualitative.Plotly
    sub_colors: dict = {}
    if sub_cfg is not None:
        all_sub = sorted(sub_cfg.df[sub_cfg.sub_entity_col].unique())
        sub_colors = {s: colors[i % len(colors)] for i, s in enumerate(all_sub)}

    # Row ordering: each cfg contributes one row per value_col
    row_configs: list[tuple[PlotConfig, str]] = [
        (cfg, vcol) for cfg in configs for vcol in cfg.value_cols
    ]
    n_rows = len(row_configs)

    figures = []
    for eid in entity_ids:
        title = f"Entity {eid}"
        if label_cfg is not None and label_cfg.label_type_col is not None:
            ltype = _dominant_label_type(label_cfg, eid)
            title = f"Entity {eid} — {ltype.capitalize()} Anomaly"

        # Plotly requires vertical_spacing < 1 / (n_rows - 1); clamp to stay valid
        if n_rows > 1:
            vertical_spacing = min(0.01, 0.9 / (n_rows - 1))
        else:
            vertical_spacing = 0.01

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            specs=[[{"secondary_y": True}] for _ in range(n_rows)],
            shared_xaxes=True,
            vertical_spacing=vertical_spacing,
        )

        for row_idx, (cfg, vcol) in enumerate(row_configs, start=1):
            df_e = cfg.df[cfg.df[cfg.entity_col] == eid].sort_values(cfg.time_col)
            _add_value_traces(fig, cfg, df_e, row_idx, vcol, sub_colors)

            if label_cfg is not None:
                label_df_e = label_cfg.df[
                    label_cfg.df[label_cfg.entity_col] == eid
                ].sort_values(label_cfg.time_col)
                _add_anomaly_markers(fig, label_cfg, label_df_e, row_idx, sub_colors)

            if label_cfg is not None:
                fig.update_yaxes(
                    title_text="",
                    row=row_idx,
                    secondary_y=True,
                    range=[-0.1, 1.1],
                    showticklabels=False,
                )

            # In-plot subplot title (top-left of each subplot).
            fig.add_annotation(
                text=vcol,
                xref="x domain",
                yref="y domain",
                x=0.01,
                y=0.95,
                showarrow=False,
                row=row_idx,
                col=1,
                font=dict(size=11, color="gray"),
                xanchor="left",
                yanchor="top",
            )


        fig.update_layout(
            title_text=title,
            hovermode="x unified",
            height=row_height * n_rows,
            width=width,
            showlegend=False,
        )
        figures.append(fig)

    return figures


def _first_with(configs: list[PlotConfig], attr: str) -> PlotConfig | None:
    for s in configs:
        if getattr(s, attr) is not None:
            return s
    return None


def _dominant_label_type(cfg: PlotConfig, entity_id) -> str:
    if cfg.label_type_col is None:
        return "normal"
    df_e = cfg.df[cfg.df[cfg.entity_col] == entity_id]
    non_normal = df_e[df_e[cfg.label_type_col] != "normal"][cfg.label_type_col]
    if non_normal.empty:
        return "normal"
    return non_normal.mode().iloc[0]


def _add_value_traces(
    fig,
    cfg: PlotConfig,
    df_e: pd.DataFrame,
    row_idx: int,
    vcol: str,
    sub_colors: dict,
):
    colors = px.colors.qualitative.Plotly

    if cfg.sub_entity_col is not None and cfg.sub_entity_col in df_e.columns:
        for sub in sorted(df_e[cfg.sub_entity_col].unique()):
            sub_data = df_e[df_e[cfg.sub_entity_col] == sub]
            color = sub_colors.get(sub, colors[0])
            fig.add_trace(
                go.Scatter(
                    x=sub_data[cfg.time_col],
                    y=sub_data[vcol],
                    name=f"{cfg.sub_entity_col} {sub}",
                    mode="lines",
                    line=dict(color=color),
                    showlegend=(row_idx == 1),
                    legendgroup=f"sub_{sub}",
                ),
                row=row_idx,
                col=1,
                secondary_y=False,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=df_e[cfg.time_col],
                y=df_e[vcol],
                name=vcol,
                mode="lines",
                line=dict(color=colors[(row_idx - 1) % len(colors)]),
                showlegend=True,
                legendgroup=f"val_{vcol}_{row_idx}",
            ),
            row=row_idx,
            col=1,
            secondary_y=False,
        )


def _add_anomaly_markers(
    fig,
    cfg: PlotConfig,
    df_e: pd.DataFrame,
    row_idx: int,
    sub_colors: dict,
):
    if cfg.label_col is None or cfg.label_col not in df_e.columns:
        return

    if cfg.sub_entity_col is not None and cfg.sub_entity_col in df_e.columns:
        anomaly_data = df_e[df_e[cfg.label_col].astype(bool)]
        if len(anomaly_data) > 0:
            is_unit_level = (
                anomaly_data.groupby(cfg.time_col)[cfg.label_col].nunique().max() > 1
                or anomaly_data[cfg.sub_entity_col].nunique()
                < df_e[cfg.sub_entity_col].nunique()
            )
        else:
            is_unit_level = False

        if is_unit_level:
            for sub in sorted(df_e[cfg.sub_entity_col].unique()):
                sub_anom = df_e[
                    (df_e[cfg.sub_entity_col] == sub) & df_e[cfg.label_col].astype(bool)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=sub_anom[cfg.time_col],
                        y=sub_anom[cfg.label_col].astype(int),
                        name=f"Anomaly — {cfg.sub_entity_col} {sub}",
                        mode="markers",
                        marker=dict(size=5, color=sub_colors.get(sub, "red")),
                        showlegend=(len(sub_anom) > 0 and row_idx == 1),
                        legendgroup=f"anom_sub_{sub}",
                    ),
                    row=row_idx,
                    col=1,
                    secondary_y=True,
                )
            return

    anom_pts = df_e[df_e[cfg.label_col].astype(bool)]
    fig.add_trace(
        go.Scatter(
            x=anom_pts[cfg.time_col],
            y=anom_pts[cfg.label_col].astype(int),
            name="Anomaly Flag",
            mode="markers",
            marker=dict(size=3, color="red"),
            showlegend=(len(anom_pts) > 0 and row_idx == 1),
            legendgroup="anomaly_flag",
        ),
        row=row_idx,
        col=1,
        secondary_y=True,
    )
