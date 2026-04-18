"""Feature weighting for Catch22 Matrix Profile models."""

from abc import ABC, abstractmethod


class FeatureWeighter(ABC):
    """Abstract base class for computing feature weights.

    Subclasses implement compute_weights() to return a dict mapping
    feature names to their weights, used to scale the feature profile
    before L2 distance computation in the left C22 matrix profile.
    """

    @abstractmethod
    def compute_weights(self, **kwargs) -> dict[str, float]:
        """Compute a weight for each feature column.

        Returns:
            Dict mapping feature name to its non-negative weight.
        """
        ...


class UniformWeighter(FeatureWeighter):
    """Equal weights for all features (no-op baseline)."""

    def compute_weights(self, feature_names: list[str]) -> dict[str, float]:
        return {name: 1.0 for name in feature_names}
