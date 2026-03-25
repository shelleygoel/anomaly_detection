import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class TimeSeriesDataset:
    """Thin wrapper over a DataFrame that maps semantic roles to column names via col_map."""

    REQUIRED_KEYS = {"entity", "time", "value_cols"}
    OPTIONAL_KEYS = {"label", "label_type", "sub_entity"}
    ALL_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS

    def __init__(self, df: pd.DataFrame, col_map: dict):
        self.df = df
        self.col_map = col_map
        self._validate()

    def _validate(self):
        missing = self.REQUIRED_KEYS - set(self.col_map)
        if missing:
            raise ValueError(f"col_map missing required keys: {missing}")

        unknown = set(self.col_map) - self.ALL_KEYS
        if unknown:
            raise ValueError(f"col_map has unknown keys: {unknown}")

        # Validate all mapped columns exist in df
        cols_to_check = [self.col_map[k] for k in self.col_map if k != "value_cols"]
        cols_to_check.extend(self.col_map["value_cols"])

        missing_cols = [c for c in cols_to_check if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    def _require(self, key: str):
        if key not in self.col_map:
            raise ValueError(f"'{key}' not in col_map — required for this operation")


    def anomaly_types(self) -> np.ndarray:
        self._require("label_type")
        return self.df[self.col_map["label_type"]].unique()

    def anomaly_summary(self) -> pd.DataFrame:
        """Count entities grouped by their dominant anomaly type.

        For each entity, the anomaly type is the most frequent non-normal
        label_type; entities with only normal labels are counted as "normal".
        """
        self._require("label_type")
        entity_col = self.col_map["entity"]
        label_type_col = self.col_map["label_type"]

        def _dominant_type(grp):
            non_normal = grp[grp[label_type_col] != "normal"][label_type_col]
            if non_normal.empty:
                return "normal"
            return non_normal.mode().iloc[0]

        entity_types = self.df.groupby(entity_col).apply(_dominant_type)
        summary = entity_types.value_counts().reset_index()
        summary.columns = ["label_type", "entity_count"]
        return summary

    def _agg_labels_fn(self, grp, label_col, label_type_col=None):
        """Aggregate anomaly labels: take max of label, label_type from row with highest label."""
        max_label = grp[label_col].max()
        result = {label_col: max_label}
        if label_type_col is not None:
            if not max_label:
                result[label_type_col] = "normal"
            else:
                result[label_type_col] = grp.loc[grp[label_col].idxmax(), label_type_col]
        return pd.Series(result)
    def day_labels(self) -> pd.DataFrame:
        """Aggregate labels to entity-day level. Returns [entity, 'day', 'label', 'label_type']."""
        self._require("label")
        entity_col = self.col_map["entity"]
        time_col = self.col_map["time"]
        label_col = self.col_map["label"]
        label_type_col = self.col_map.get("label_type")

        df = self.df.copy()
        df["day"] = df[time_col].dt.date

        group_cols = [entity_col, "day"]

        def agg_fn(grp):
            return self._agg_labels_fn(grp, label_col, label_type_col)

        out = df.groupby(group_cols, sort=False).apply(agg_fn).reset_index()
        return out

    def ts_labels(self) -> pd.DataFrame:
        """Aggregate labels to entity-timestamp level. Returns [entity, time, 'label', 'label_type']."""
        self._require("label")
        entity_col = self.col_map["entity"]
        time_col = self.col_map["time"]
        label_col = self.col_map["label"]
        label_type_col = self.col_map.get("label_type")

        if "sub_entity" in self.col_map:
            # Aggregate across sub_entities per (entity, timestamp)
            group_cols = [entity_col, time_col]

            def agg_fn(grp):
                return self._agg_labels_fn(grp, label_col, label_type_col)

            out = self.df.groupby(group_cols, sort=False).apply(agg_fn).reset_index()
        else:
            # No sub_entity — just select relevant columns
            cols = [entity_col, time_col, label_col]
            if label_type_col is not None:
                cols.append(label_type_col)
            out = self.df[cols].copy()

        return out

    def sample_and_visualize_cases(
        self, n_cases: int = 1, entity_ids=None, label_type: str = None
    ) -> list[go.Figure]:
        """Generic visualization: one figure per entity, one subplot row per value_col.

        Returns a list of Plotly figures (one per sampled entity).
        """
        entity_col = self.col_map["entity"]
        time_col = self.col_map["time"]
        value_cols = self.col_map["value_cols"]
        has_label = "label" in self.col_map
        has_sub_entity = "sub_entity" in self.col_map
        sub_entity_col = self.col_map.get("sub_entity")
        colors = px.colors.qualitative.Plotly

        sampled = self._get_sampled_entities(n_cases, entity_ids, label_type)

        # Build color map for sub_entities if applicable
        sub_colors = {}
        if has_sub_entity:
            all_sub = sorted(self.df[sub_entity_col].unique())
            sub_colors = {s: colors[i % len(colors)] for i, s in enumerate(all_sub)}

        n_rows = len(value_cols)
        figures = []

        for eid in sampled:
            df_e = self.df[self.df[entity_col] == eid].sort_values(time_col)
            entity_label_type = self._get_entity_label_type(df_e)
            title = f"Entity {eid} — {entity_label_type.capitalize()} Anomaly"

            fig = make_subplots(
                rows=n_rows,
                cols=1,
                subplot_titles=[vcol for vcol in value_cols],
                specs=[[{"secondary_y": True}] for _ in range(n_rows)],
                shared_xaxes=True,
                vertical_spacing=max(0.05, 0.3 / n_rows),
            )

            for row_idx, vcol in enumerate(value_cols, start=1):
                self._add_value_traces_single(
                    fig, df_e, row_idx, vcol, has_sub_entity, sub_colors
                )
                if has_label:
                    self._add_anomaly_traces(
                        fig, df_e, row_idx, has_sub_entity, sub_colors
                    )

                fig.update_yaxes(title_text=vcol, row=row_idx, secondary_y=False)
                if has_label:
                    fig.update_yaxes(
                        title_text="Anomaly Flag",
                        row=row_idx,
                        secondary_y=True,
                        range=[-0.1, 1.1],
                    )
                fig.update_xaxes(title_text="Timestamp", row=row_idx)

            fig.update_layout(
                title_text=title,
                hovermode="x unified",
                height=300 * n_rows,
            )
            figures.append(fig)

        return figures
    
    def _get_sampled_entities(self, n_cases: int, entity_ids, label_type: str):

        """Select entities to visualize."""
        entity_col = self.col_map["entity"]
        label_type_col = self.col_map.get("label_type")

        if entity_ids is not None:
            return np.asarray(entity_ids)

        candidates = self.df[entity_col].unique()
        if label_type is not None and label_type_col is not None:
            mask = self.df[label_type_col] == label_type
            candidates = self.df.loc[mask, entity_col].unique()

        n = min(n_cases, len(candidates))
        return np.random.choice(candidates, size=n, replace=False)

    def _get_entity_label_type(self, df_e) -> str:
        """Return the most frequent non-normal label_type for an entity, or 'normal'."""
        label_type_col = self.col_map.get("label_type")
        if label_type_col is None:
            return "normal"
        non_normal = df_e[df_e[label_type_col] != "normal"][label_type_col]
        if non_normal.empty:
            return "normal"
        return non_normal.mode().iloc[0]

    def _add_value_traces_single(self, fig, df_e, row_idx, vcol, has_sub_entity, sub_colors):
        """Add traces for a single value column on one subplot row."""
        time_col = self.col_map["time"]
        sub_entity_col = self.col_map.get("sub_entity")
        colors = px.colors.qualitative.Plotly

        if has_sub_entity:
            for sub in sorted(df_e[sub_entity_col].unique()):
                sub_data = df_e[df_e[sub_entity_col] == sub]
                color = sub_colors[sub]
                fig.add_trace(
                    go.Scatter(
                        x=sub_data[time_col],
                        y=sub_data[vcol],
                        name=f"{sub_entity_col} {sub}",
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
                    x=df_e[time_col],
                    y=df_e[vcol],
                    name=vcol,
                    mode="lines",
                    line=dict(color=colors[(row_idx - 1) % len(colors)]),
                    showlegend=(row_idx == 1),
                    legendgroup=f"val_{vcol}",
                ),
                row=row_idx,
                col=1,
                secondary_y=False,
            )

    def _add_anomaly_traces(self, fig, df_e, row_idx, has_sub_entity, sub_colors):
        """Add anomaly marker traces."""
        time_col = self.col_map["time"]
        label_col = self.col_map.get("label")
        sub_entity_col = self.col_map.get("sub_entity")

        if label_col is None:
            return

        if has_sub_entity:
            # Determine if anomaly is unit-level or global
            anomaly_data = df_e[df_e[label_col].astype(bool)]
            if len(anomaly_data) > 0:
                is_unit_level = (
                    anomaly_data.groupby(time_col)[label_col].nunique().max() > 1
                    or anomaly_data[sub_entity_col].nunique()
                    < df_e[sub_entity_col].nunique()
                )
            else:
                is_unit_level = False

            if is_unit_level:
                for sub in sorted(df_e[sub_entity_col].unique()):
                    sub_anom = df_e[
                        (df_e[sub_entity_col] == sub)
                        & df_e[label_col].astype(bool)
                    ]
                    fig.add_trace(
                        go.Scatter(
                            x=sub_anom[time_col],
                            y=sub_anom[label_col].astype(int),
                            name=f"Anomaly — {sub_entity_col} {sub}",
                            mode="markers",
                            marker=dict(size=5, color=sub_colors[sub]),
                            showlegend=(len(sub_anom) > 0 and row_idx == 1),
                            legendgroup=f"anom_sub_{sub}",
                        ),
                        row=row_idx,
                        col=1,
                        secondary_y=True,
                    )
            else:
                anom_pts = df_e[df_e[label_col].astype(bool)]
                fig.add_trace(
                    go.Scatter(
                        x=anom_pts[time_col],
                        y=anom_pts[label_col].astype(int),
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
        else:
            anom_pts = df_e[df_e[label_col].astype(bool)]
            fig.add_trace(
                go.Scatter(
                    x=anom_pts[time_col],
                    y=anom_pts[label_col].astype(int),
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
