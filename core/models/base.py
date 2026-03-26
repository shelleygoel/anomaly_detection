"""Abstract base class for anomaly detection models."""

from abc import ABC, abstractmethod

from core.dataset import TimeSeriesDataset


class AnomalyModel(ABC):
    """Abstract base class for anomaly detection models.

    Subclasses must implement score_anomalies(), which takes a TimeSeriesDataset
    and returns a new TimeSeriesDataset with an 'anomaly_score' column.
    """

    @abstractmethod
    def score_anomalies(self, dataset: TimeSeriesDataset, level: str = "day") -> TimeSeriesDataset:
        """Score each (entity, time) for anomalousness.

        Args:
            dataset: Input TimeSeriesDataset (must have sub_entity if the model requires it).
            level: "day" for day-level aggregated scores, "timestamp" for raw timestamp-level scores.

        Returns:
            TimeSeriesDataset with 'anomaly_score' in value_cols.
        """
        ...
