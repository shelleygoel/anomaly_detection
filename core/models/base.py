"""Base class for anomaly detection models.

Defines the common interface but does not mandate a single API shape.  Models
with a simple one-shot API (e.g. EuclideanDistModel) override `score_anomalies`.
Models with a two-step profile/score split (e.g. Catch22MPModel) expose their
own methods (`fit_profile` + `score`) and leave `score_anomalies` unimplemented.
"""

from core.dataset import TimeSeriesDataset


class AnomalyModel:
    """Common base for anomaly detection models.

    Subclasses may either:
      - Override `score_anomalies` for a one-shot API, or
      - Provide their own staged API (e.g. `fit_profile` + `score`).
    """

    def score_anomalies(self, dataset: TimeSeriesDataset, level: str = "day") -> TimeSeriesDataset:
        """Score each (entity, time) for anomalousness.

        Args:
            dataset: Input TimeSeriesDataset (must have sub_entity if the model requires it).
            level: "day" for day-level aggregated scores, "timestamp" for raw timestamp-level scores.

        Returns:
            TimeSeriesDataset with 'anomaly_score' in value_cols.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the one-shot score_anomalies API. "
            f"Use its model-specific methods (e.g. fit_profile + score)."
        )
