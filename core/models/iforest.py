"""IsolationForest anomaly detection on a TimeSeriesDataset feature matrix."""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from tqdm.auto import tqdm

from core.dataset import TimeSeriesDataset
from core.models.base import AnomalyModel


class IForestModel(AnomalyModel):
    """Isolation Forest anomaly detection over a (entity, time, features) dataset.

    Scores each (entity, timestamp) with HIGHER = more anomalous (negated
    sklearn `score_samples`, matching `Catch22MPModel.score`).

    Args:
        n_estimators, contamination, max_samples, random_state: forwarded to
            sklearn.ensemble.IsolationForest.
        fit_scope: "per_entity" fits one IForest per entity (each entity scored
            against its own baseline). "global" fits a single IForest on all
            entities pooled.
        feature_cols: Subset of value_cols to use. None = all value_cols.
    """

    DAY_AGG_STATS = {"max", "p90", "mean"}

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        max_samples: int | float | str = "auto",
        random_state: int | None = 42,
        fit_scope: Literal["per_entity", "global"] = "per_entity",
        feature_cols: list[str] | None = None,
    ):
        if fit_scope not in ("per_entity", "global"):
            raise ValueError(
                f"fit_scope must be 'per_entity' or 'global', got {fit_scope!r}"
            )
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.fit_scope = fit_scope
        self.feature_cols = feature_cols

    def score_anomalies(
        self,
        dataset: TimeSeriesDataset,
        level: str = "timestamp",
        day_agg_stat: str = "max",
    ) -> TimeSeriesDataset:
        if level not in ("timestamp", "day"):
            raise ValueError(f"level must be 'timestamp' or 'day', got {level!r}")
        if level == "day" and day_agg_stat not in self.DAY_AGG_STATS:
            raise ValueError(
                f"Unknown day_agg_stat: {day_agg_stat!r}. Use one of {self.DAY_AGG_STATS}."
            )

        entity_col = dataset.col_map["entity"]
        time_col = dataset.col_map["time"]
        feat_cols = (
            self.feature_cols
            if self.feature_cols is not None
            else dataset.col_map["value_cols"]
        )

        if self.fit_scope == "per_entity":
            ts_df = self._score_per_entity(dataset.df, entity_col, time_col, feat_cols)
        else:
            ts_df = self._score_global(dataset.df, entity_col, time_col, feat_cols)

        if level == "timestamp":
            return TimeSeriesDataset(
                ts_df,
                {"entity": entity_col, "time": time_col, "value_cols": ["anomaly_score"]},
            )

        ts_df = ts_df.copy()
        ts_df["day"] = pd.to_datetime(ts_df[time_col]).dt.date
        agg_func = self._get_agg_func(day_agg_stat)
        day_df = (
            ts_df.dropna(subset=["anomaly_score"])
            .groupby([entity_col, "day"])
            .agg(anomaly_score=("anomaly_score", agg_func))
            .reset_index()
        )
        return TimeSeriesDataset(
            day_df,
            {"entity": entity_col, "time": "day", "value_cols": ["anomaly_score"]},
        )

    def _score_per_entity(
        self, df: pd.DataFrame, entity_col: str, time_col: str, feat_cols: list[str]
    ) -> pd.DataFrame:
        out_rows = []
        for entity_id, grp in tqdm(df.groupby(entity_col), desc="IForest per-entity"):
            scores = self._fit_and_score(grp[feat_cols].values)
            out_rows.append(pd.DataFrame({
                entity_col: entity_id,
                time_col: grp[time_col].values,
                "anomaly_score": scores,
            }))
        return pd.concat(out_rows, ignore_index=True)

    def _score_global(
        self, df: pd.DataFrame, entity_col: str, time_col: str, feat_cols: list[str]
    ) -> pd.DataFrame:
        scores = self._fit_and_score(df[feat_cols].values)
        return pd.DataFrame({
            entity_col: df[entity_col].values,
            time_col: df[time_col].values,
            "anomaly_score": scores,
        })

    def _fit_and_score(self, X: np.ndarray) -> np.ndarray:
        mask = ~np.isnan(X).any(axis=1)
        scores = np.full(X.shape[0], np.nan)
        if mask.sum() == 0:
            return scores
        clf = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
        )
        clf.fit(X[mask])
        scores[mask] = -clf.score_samples(X[mask])
        return scores

    def _get_agg_func(self, stat: str):
        if stat == "max":
            return "max"
        elif stat == "p90":
            return lambda x: np.percentile(x, 90)
        elif stat == "mean":
            return "mean"
