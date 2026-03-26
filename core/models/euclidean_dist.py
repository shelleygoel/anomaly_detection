"""
Pairwise Euclidean distance anomaly detection model.

Computes rolling pairwise Euclidean distances between sub-entities within each
entity, then scores them using MAD (Median Absolute Deviation) z-scores.
Works on any TimeSeriesDataset that has a sub_entity column.
"""

import itertools

import numpy as np
import pandas as pd

from core.dataset import TimeSeriesDataset
from core.models.base import AnomalyModel


class EuclideanDistModel(AnomalyModel):
    """Anomaly detection via pairwise rolling Euclidean distance between sub-entities.

    Args:
        smooth_window: Rolling mean window for smoothing raw values before distance computation.
        dist_window: Rolling window size for Euclidean distance computation.
        strategy: Scoring strategy — "mad" or "iqr".
    """

    def __init__(self, feature_col: str, smooth_window: int = 10, dist_window: int = 60, strategy: str = "mad"):
        if strategy not in ("mad", "iqr"):
            raise ValueError(f"Unknown strategy: {strategy!r}. Use 'mad' or 'iqr'.")
        self.feature_col = feature_col
        self.smooth_window = smooth_window
        self.dist_window = dist_window
        self.strategy = strategy

    def score_anomalies(self, dataset: TimeSeriesDataset, level: str = "day") -> TimeSeriesDataset:
        """Score anomalies by computing pairwise distances then applying MAD/IQR scoring.

        Args:
            dataset: Must have 'sub_entity' in col_map.
            level: "day" for day-level aggregated scores, "timestamp" for raw scores.

        Returns:
            TimeSeriesDataset with value_cols=['max_eucl_dist', 'anomaly_score'].
        """
        dist_df = self._compute_pairwise_distances(dataset)
        return self._score_anomalies(dist_df, dataset, level=level)

    def _compute_pairwise_distances(self, dataset: TimeSeriesDataset) -> pd.DataFrame:
        """Compute rolling pairwise Euclidean distances between all sub-entity pairs.

        Args:
            dataset: Input dataset with sub_entity column.

        Returns:
            DataFrame with columns [entity_col, time_col, 'pair', 'distance'].
        """
        dataset._require("sub_entity")

        entity_col = dataset.col_map["entity"]
        time_col = dataset.col_map["time"]
        sub_entity_col = dataset.col_map["sub_entity"]
        # TODO: Add value_col name as model parameter, and rename to feature_col

        results = []

        for entity_id, grp in dataset.df.groupby(entity_col):
            # Smooth values per sub-entity
            smoothed = grp.copy()
            smoothed["_smooth"] = smoothed.groupby(sub_entity_col)[self.feature_col].transform(
                lambda x: x.rolling(self.smooth_window).mean()
            )

            # Pivot to wide format: one column per sub-entity
            pivot = smoothed.pivot_table(
                index=time_col, columns=sub_entity_col, values="_smooth"
            )

            # All unique sub-entity pairs
            sub_entities = sorted(pivot.columns)
            for a, b in itertools.combinations(sub_entities, 2):
                dist = np.sqrt(
                    ((pivot[a] - pivot[b]) ** 2).rolling(self.dist_window).sum()
                )
                pair_df = pd.DataFrame({
                    entity_col: entity_id,
                    time_col: pivot.index,
                    "pair": f"{a}_{b}",
                    "distance": dist.values,
                })
                results.append(pair_df)

        return pd.concat(results, ignore_index=True)

    def _score_anomalies(
        self, dist_df: pd.DataFrame, dataset: TimeSeriesDataset, level: str = "day"
    ) -> TimeSeriesDataset:
        """Score pairwise distances using MAD or IQR z-scores.

        1. Max distance across pairs per (entity, timestamp).
        2. Compute global baseline (median + MAD or IQR) across all entities.
        3. Z-score each (entity, timestamp).
        4. If level="timestamp", return timestamp-level scores.
        5. If level="day", aggregate to day level via 90th percentile.
        """
        entity_col = dataset.col_map["entity"]
        time_col = dataset.col_map["time"]
        value_cols = ["max_eucl_dist", "anomaly_score"]

        # Drop NaN distances (rolling window warmup)
        clean = dist_df.dropna(subset=["distance"])

        # Max across pairs per (entity, timestamp)
        entity_ts = (
            clean.groupby([entity_col, time_col])["distance"]
            .max()
            .reset_index(name="max_eucl_dist")
        )

        # Global baseline
        pool = entity_ts["max_eucl_dist"]

        if self.strategy == "mad":
            median = pool.median()
            mad = (pool - median).abs().median()
            scale = mad * 1.4826 if mad > 0 else 1.0
            entity_ts["anomaly_score"] = (entity_ts["max_eucl_dist"] - median) / scale
        else:  # iqr
            q1 = pool.quantile(0.25)
            q3 = pool.quantile(0.75)
            iqr = q3 - q1
            scale = iqr if iqr > 0 else 1.0
            entity_ts["anomaly_score"] = (entity_ts["max_eucl_dist"] - q3) / scale

        # Timestamp-level: return early
        if level == "timestamp":
            out_col_map = {
                "entity": entity_col,
                "time": time_col,
                "value_cols": value_cols,
            }
            return TimeSeriesDataset(entity_ts[[entity_col, time_col] + value_cols], out_col_map)

        # Day-level: aggregate via 90th percentile
        entity_ts["day"] = entity_ts[time_col].dt.date
        day_df = (
            entity_ts.groupby([entity_col, "day"])[value_cols]
            .quantile(0.9)
            .reset_index()
        )

        out_col_map = {
            "entity": entity_col,
            "time": "day",
            "value_cols": value_cols,
        }
        return TimeSeriesDataset(day_df, out_col_map)

    def score_anomalies_v2(
        self, dataset: TimeSeriesDataset, day_agg: str = "mean"
    ) -> TimeSeriesDataset:
        """Score anomalies by aggregating distances to day level per pair before z-scoring.

        Unlike score_anomalies which z-scores at timestamp level then aggregates,
        this method:
        1. Aggregates distance to day level per pair (mean or max).
        2. Z-scores the day-level distances using global baseline.
        3. Takes max score across pairs per (entity, day).

        Args:
            dataset: Must have 'sub_entity' in col_map.
            day_agg: "mean" or "max" — how to aggregate timestamp distances to day per pair.

        Returns:
            TimeSeriesDataset at (entity, day) level with value_cols=['anomaly_score'].
        """
        if day_agg not in ("mean", "max"):
            raise ValueError(f"Unknown day_agg: {day_agg!r}. Use 'mean' or 'max'.")

        dist_df = self._compute_pairwise_distances(dataset)
        entity_col = dataset.col_map["entity"]
        time_col = dataset.col_map["time"]

        clean = dist_df.dropna(subset=["distance"])
        clean = clean.copy()
        clean["day"] = clean[time_col].dt.date

        # Step 1: day-level aggregation of distance per pair
        day_pair = (
            clean.groupby([entity_col, "pair", "day"])["distance"]
            .agg(day_agg)
            .reset_index(name="day_dist")
        )

        # Step 2: z-score using global baseline (all entities, all pairs)
        pool = day_pair["day_dist"].dropna()

        if self.strategy == "mad":
            median = pool.median()
            mad = (pool - median).abs().median()
            scale = mad * 1.4826 if mad > 0 else 1.0
            day_pair["score"] = (day_pair["day_dist"] - median) / scale
        else:  # iqr
            q1 = pool.quantile(0.25)
            q3 = pool.quantile(0.75)
            iqr = q3 - q1
            scale = iqr if iqr > 0 else 1.0
            day_pair["score"] = (day_pair["day_dist"] - q3) / scale

        # Step 3: max score across pairs per (entity, day)
        result = (
            day_pair.groupby([entity_col, "day"])["score"]
            .max()
            .reset_index(name="anomaly_score")
        )

        out_col_map = {
            "entity": entity_col,
            "time": "day",
            "value_cols": ["anomaly_score"],
        }
        return TimeSeriesDataset(result, out_col_map)
