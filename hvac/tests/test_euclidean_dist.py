"""Smoke tests for hvac.utils.euclidean_dist module."""

import numpy as np
import pandas as pd
import pytest

from hvac.utils.euclidean_dist import (
    rolling_euclidean_distance,
    compute_pairwise_distances,
    compute_pairwise_correlations,
    flag_anomalies,
    score_anomalies,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_series():
    """Two constant series (identical) and one shifted series."""
    n = 100
    a = pd.Series(np.ones(n))
    b = pd.Series(np.ones(n))
    c = pd.Series(np.ones(n) + 5.0)  # offset by 5
    return a, b, c


@pytest.fixture
def hvac_df_with_anomaly():
    """Single container, 3 units, unit 1 diverges in the middle."""
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range("2026-01-01", periods=n, freq="min")
    rows = []
    for unit in [0, 1, 2]:
        base = np.sin(np.linspace(0, 4 * np.pi, n)) * 10 + 50
        if unit == 1:
            base[200:300] += 5  # inject divergence
        for i, t in enumerate(timestamps):
            rows.append(
                {
                    "container_id": 0,
                    "timestamp_et": t,
                    "unit": unit,
                    "TmpRet": base[i] + np.random.randn() * 0.1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def multi_container_df():
    """Two containers: container 0 has divergence, container 1 is normal."""
    np.random.seed(99)
    n = 300
    timestamps = pd.date_range("2026-01-01", periods=n, freq="min")
    rows = []
    for cid in [0, 1]:
        for unit in [0, 1, 2]:
            base = np.sin(np.linspace(0, 2 * np.pi, n)) * 10 + 50
            if cid == 0 and unit == 2:
                base[100:200] += 8  # divergence only in container 0
            for i, t in enumerate(timestamps):
                rows.append(
                    {
                        "container_id": cid,
                        "timestamp_et": t,
                        "unit": unit,
                        "TmpRet": base[i] + np.random.randn() * 0.1,
                    }
                )
    return pd.DataFrame(rows)


# ── rolling_euclidean_distance ───────────────────────────────────────────────


class TestRollingEuclideanDistance:
    def test_identical_series_returns_zero(self, simple_series):
        a, b, _ = simple_series
        result = rolling_euclidean_distance(a, b, window=10)
        # After warm-up, all values should be 0
        assert (result.dropna() == 0).all()

    def test_offset_series_returns_positive(self, simple_series):
        a, _, c = simple_series
        result = rolling_euclidean_distance(a, c, window=10)
        assert (result.dropna() > 0).all()

    def test_output_length_matches_input(self, simple_series):
        a, _, c = simple_series
        result = rolling_euclidean_distance(a, c, window=10)
        assert len(result) == len(a)

    def test_leading_nans_from_rolling(self, simple_series):
        a, _, c = simple_series
        window = 10
        result = rolling_euclidean_distance(a, c, window=window)
        assert result.isna().sum() == window - 1


# ── compute_pairwise_distances ───────────────────────────────────────────────


class TestComputePairwiseDistances:
    def test_output_columns(self, hvac_df_with_anomaly):
        dist_df = compute_pairwise_distances(
            hvac_df_with_anomaly, smooth_window=5, dist_window=30
        )
        assert set(dist_df.columns) == {
            "container_id",
            "timestamp_et",
            "pair",
            "distance",
        }

    def test_three_pairs(self, hvac_df_with_anomaly):
        dist_df = compute_pairwise_distances(
            hvac_df_with_anomaly, smooth_window=5, dist_window=30
        )
        assert sorted(dist_df["pair"].unique()) == ["0_1", "0_2", "1_2"]

    def test_divergent_unit_produces_higher_distances(self, hvac_df_with_anomaly):
        dist_df = compute_pairwise_distances(
            hvac_df_with_anomaly, smooth_window=5, dist_window=30
        )
        # Pairs involving unit 1 should have higher max distance than pair 0_2
        max_by_pair = dist_df.groupby("pair")["distance"].max()
        assert max_by_pair["0_1"] > max_by_pair["0_2"]
        assert max_by_pair["1_2"] > max_by_pair["0_2"]

    def test_multi_container_handled(self, multi_container_df):
        dist_df = compute_pairwise_distances(
            multi_container_df, smooth_window=5, dist_window=30
        )
        assert sorted(dist_df["container_id"].unique()) == [0, 1]
        # Each container should have 3 pairs
        for cid in [0, 1]:
            pairs = dist_df[dist_df["container_id"] == cid]["pair"].unique()
            assert len(pairs) == 3


# ── compute_pairwise_correlations ─────────────────────────────────────────────


class TestComputePairwiseCorrelations:
    def test_output_columns(self, hvac_df_with_anomaly):
        corr_df = compute_pairwise_correlations(
            hvac_df_with_anomaly, smooth_window=5, corr_window=30
        )
        assert set(corr_df.columns) == {
            "container_id",
            "timestamp_et",
            "pair",
            "distance",
        }

    def test_three_pairs(self, hvac_df_with_anomaly):
        corr_df = compute_pairwise_correlations(
            hvac_df_with_anomaly, smooth_window=5, corr_window=30
        )
        assert sorted(corr_df["pair"].unique()) == ["0_1", "0_2", "1_2"]

    def test_same_shape_as_distances(self, hvac_df_with_anomaly):
        dist_df = compute_pairwise_distances(
            hvac_df_with_anomaly, smooth_window=5, dist_window=30
        )
        corr_df = compute_pairwise_correlations(
            hvac_df_with_anomaly, smooth_window=5, corr_window=30
        )
        assert dist_df.shape == corr_df.shape

    def test_divergent_unit_produces_higher_values(self, hvac_df_with_anomaly):
        corr_df = compute_pairwise_correlations(
            hvac_df_with_anomaly, smooth_window=5, corr_window=30
        )
        # Pairs involving unit 1 (divergent) should have higher max 1-corr than pair 0_2
        max_by_pair = corr_df.groupby("pair")["distance"].max()
        assert max_by_pair["0_1"] > max_by_pair["0_2"]
        assert max_by_pair["1_2"] > max_by_pair["0_2"]

    def test_multi_container_handled(self, multi_container_df):
        corr_df = compute_pairwise_correlations(
            multi_container_df, smooth_window=5, corr_window=30
        )
        assert sorted(corr_df["container_id"].unique()) == [0, 1]
        for cid in [0, 1]:
            pairs = corr_df[corr_df["container_id"] == cid]["pair"].unique()
            assert len(pairs) == 3

    def test_plugs_into_score_anomalies(self, hvac_df_with_anomaly):
        corr_df = compute_pairwise_correlations(
            hvac_df_with_anomaly, smooth_window=5, corr_window=30
        )
        result = score_anomalies(corr_df, strategy="mad", model_name="correlation_mad")
        assert result["model"].iloc[0] == "correlation_mad"
        assert len(result) > 0


# ── score_anomalies ──────────────────────────────────────────────────────────


class TestScoreAnomalies:
    def _get_scores(self, df, strategy="mad"):
        dist_df = compute_pairwise_distances(df, smooth_window=5, dist_window=30)
        return score_anomalies(dist_df, strategy=strategy)

    def test_output_columns(self, hvac_df_with_anomaly):
        result = self._get_scores(hvac_df_with_anomaly)
        assert set(result.columns) == {
            "container_id", "day", "unit", "anomaly_score", "model",
        }

    def test_one_row_per_container_unit_day(self, hvac_df_with_anomaly):
        result = self._get_scores(hvac_df_with_anomaly)
        dupes = result.duplicated(subset=["container_id", "unit", "day"])
        assert not dupes.any()

    def test_divergent_unit_scores_higher(self, hvac_df_with_anomaly):
        result = self._get_scores(hvac_df_with_anomaly, strategy="mad")
        # Unit 1 has divergence — its max score should exceed units 0 and 2
        max_scores = result.groupby("unit")["anomaly_score"].max()
        assert max_scores[1] > max_scores[0]
        assert max_scores[1] > max_scores[2]

    def test_iqr_strategy(self, hvac_df_with_anomaly):
        result = self._get_scores(hvac_df_with_anomaly, strategy="iqr")
        assert result["model"].iloc[0] == "euclidean_distance_iqr"
        assert len(result) > 0

    def test_multi_container(self, multi_container_df):
        result = self._get_scores(multi_container_df)
        assert sorted(result["container_id"].unique()) == [0, 1]

    def test_custom_model_name(self, hvac_df_with_anomaly):
        dist_df = compute_pairwise_distances(
            hvac_df_with_anomaly, smooth_window=5, dist_window=30
        )
        result = score_anomalies(dist_df, strategy="mad", model_name="custom_model")
        assert result["model"].iloc[0] == "custom_model"

    def test_default_model_name(self, hvac_df_with_anomaly):
        dist_df = compute_pairwise_distances(
            hvac_df_with_anomaly, smooth_window=5, dist_window=30
        )
        result = score_anomalies(dist_df, strategy="mad")
        assert result["model"].iloc[0] == "euclidean_distance_mad"

    def test_invalid_strategy_raises(self, hvac_df_with_anomaly):
        dist_df = compute_pairwise_distances(
            hvac_df_with_anomaly, smooth_window=5, dist_window=30
        )
        with pytest.raises(ValueError, match="Unknown strategy"):
            score_anomalies(dist_df, strategy="bogus")

    def test_does_not_mutate_input(self, hvac_df_with_anomaly):
        dist_df = compute_pairwise_distances(
            hvac_df_with_anomaly, smooth_window=5, dist_window=30
        )
        original_cols = list(dist_df.columns)
        score_anomalies(dist_df, strategy="mad")
        assert list(dist_df.columns) == original_cols


# ── flag_anomalies ───────────────────────────────────────────────────────────


class TestFlagAnomalies:
    def _get_flagged(self, df, strategy="mad", **kwargs):
        dist_df = compute_pairwise_distances(df, smooth_window=5, dist_window=30)
        return flag_anomalies(dist_df, strategy=strategy, **kwargs)

    def test_output_columns(self, hvac_df_with_anomaly):
        result = self._get_flagged(hvac_df_with_anomaly)
        assert set(result.columns) == {
            "container_id", "day", "unit", "anomaly_score", "anomaly_flag", "model",
        }

    def test_mad_finds_anomalies(self, hvac_df_with_anomaly):
        result = self._get_flagged(hvac_df_with_anomaly, strategy="mad", k=3)
        assert result["anomaly_flag"].sum() > 0

    def test_iqr_finds_anomalies(self, hvac_df_with_anomaly):
        result = self._get_flagged(hvac_df_with_anomaly, strategy="iqr", k=1.5)
        assert result["anomaly_flag"].sum() > 0

    def test_divergent_unit_flagged(self, hvac_df_with_anomaly):
        """Unit 1 should be flagged; units 0 and 2 should not."""
        result = self._get_flagged(hvac_df_with_anomaly, strategy="mad", k=3)
        flagged_units = result[result["anomaly_flag"]]["unit"].unique()
        assert 1 in flagged_units
        assert 0 not in flagged_units
        assert 2 not in flagged_units

    def test_invalid_strategy_raises(self, hvac_df_with_anomaly):
        dist_df = compute_pairwise_distances(
            hvac_df_with_anomaly, smooth_window=5, dist_window=30
        )
        with pytest.raises(ValueError, match="Unknown strategy"):
            flag_anomalies(dist_df, strategy="bogus")

    def test_does_not_mutate_input(self, hvac_df_with_anomaly):
        dist_df = compute_pairwise_distances(
            hvac_df_with_anomaly, smooth_window=5, dist_window=30
        )
        original_cols = list(dist_df.columns)
        flag_anomalies(dist_df, strategy="mad", k=3)
        assert list(dist_df.columns) == original_cols
