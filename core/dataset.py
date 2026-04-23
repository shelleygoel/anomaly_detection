import numpy as np
import pandas as pd

from core.viz import PlotConfig


class TimeSeriesDataset:
    """Thin wrapper over a DataFrame that maps semantic roles to column names via col_map."""

    REQUIRED_KEYS = {"entity", "time", "value_cols"}
    OPTIONAL_KEYS = {"label", "label_type", "sub_entity"}
    ALL_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS

    def __init__(self, df: pd.DataFrame, col_map: dict):
        self.df = df
        self.col_map = col_map
        self._day_labels_cache: pd.DataFrame | None = None
        self._validate()

    def to_plot_cfg(self) -> PlotConfig:
        """Build a PlotConfig from this dataset's col_map for use with core.viz.plot_cases."""
        return PlotConfig(
            df=self.df,
            entity_col=self.col_map["entity"],
            time_col=self.col_map["time"],
            value_cols=self.col_map["value_cols"],
            label_col=self.col_map.get("label"),
            sub_entity_col=self.col_map.get("sub_entity"),
            label_type_col=self.col_map.get("label_type"),
        )

    def sample_entities(
        self,
        n_cases: int = 1,
        label_type: str | None = None,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Return up to n_cases entity IDs, optionally filtered by dominant label_type.

        Args:
            n_cases: Maximum number of entity IDs to return.
            label_type: If set, only entities whose dominant label_type matches are considered.
                Requires 'label_type' in col_map.
            random_state: Optional seed for reproducibility.

        Returns:
            Array of sampled entity IDs.
        """
        entity_col = self.col_map["entity"]
        candidates = self.df[entity_col].unique()

        if label_type is not None:
            self._require("label_type")
            label_type_col = self.col_map["label_type"]
            if label_type == "normal":
                # Only entities whose rows are *entirely* normal.
                per_entity_all_normal = (
                    self.df.assign(_is_normal=self.df[label_type_col] == "normal")
                    .groupby(entity_col)["_is_normal"]
                    .all()
                )
                candidates = per_entity_all_normal[per_entity_all_normal].index.to_numpy()
            else:
                mask = self.df[label_type_col] == label_type
                candidates = self.df.loc[mask, entity_col].unique()

        n = min(n_cases, len(candidates))
        rng = np.random.default_rng(random_state)
        return rng.choice(candidates, size=n, replace=False)

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

        entity_types = self.df.groupby(entity_col).apply(_dominant_type, include_groups=False)
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
        """Aggregate labels to entity-day level. Returns [entity, 'day', 'label', 'label_type'].

        Memoized per instance — callers should not mutate the returned frame.
        """
        if self._day_labels_cache is not None:
            return self._day_labels_cache

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

        out = df.groupby(group_cols, sort=False, group_keys=False).apply(
            agg_fn, include_groups=False
        ).reset_index()
        self._day_labels_cache = out
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

            out = self.df.groupby(group_cols, sort=False).apply(
                agg_fn, include_groups=False
            ).reset_index()
        else:
            # No sub_entity — just select relevant columns
            cols = [entity_col, time_col, label_col]
            if label_type_col is not None:
                cols.append(label_type_col)
            out = self.df[cols].copy()

        return out
