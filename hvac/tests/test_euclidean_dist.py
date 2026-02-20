"""Smoke tests for hvac.utils.euclidean_dist module."""

import numpy as np
import pandas as pd
import pytest

from hvac.utils.euclidean_dist import (
    rolling_euclidean_distance,
    compute_pairwise_distances,
    flag_anomalies,
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


# ── flag_anomalies ───────────────────────────────────────────────────────────


class TestFlagAnomalies:
    def _get_flagged(self, df, strategy="mad", **kwargs):
        dist_df = compute_pairwise_distances(df, smooth_window=5, dist_window=30)
        return flag_anomalies(dist_df, strategy=strategy, **kwargs)

    def test_output_columns(self, hvac_df_with_anomaly):
        unit_df, _ = self._get_flagged(hvac_df_with_anomaly)
        assert set(unit_df.columns) == {
            "timestamp_et",
            "container_id",
            "unit",
            "dist_0",
            "dist_1",
            "dist_2",
            "anomaly_flag",
            "model",
        }

    def test_thresholds_df_schema(self, hvac_df_with_anomaly):
        _, thresholds_df = self._get_flagged(hvac_df_with_anomaly)
        assert set(thresholds_df.columns) == {"container_id", "threshold", "strategy"}
        # Single container → one row
        assert len(thresholds_df) == 1

    def test_mad_finds_anomalies(self, hvac_df_with_anomaly):
        unit_df, _ = self._get_flagged(hvac_df_with_anomaly, strategy="mad", k=3)
        assert unit_df["anomaly_flag"].sum() > 0

    def test_iqr_finds_anomalies(self, hvac_df_with_anomaly):
        unit_df, _ = self._get_flagged(hvac_df_with_anomaly, strategy="iqr", k=1.5)
        assert unit_df["anomaly_flag"].sum() > 0

    def test_thresholds_one_row_per_container(self, multi_container_df):
        _, thresholds_df = self._get_flagged(multi_container_df, strategy="mad", k=3)
        assert len(thresholds_df) == 2
        assert sorted(thresholds_df["container_id"].tolist()) == [0, 1]

    def test_per_container_thresholds_differ(self, multi_container_df):
        _, thresholds_df = self._get_flagged(multi_container_df, strategy="mad", k=3)
        thresholds = thresholds_df.set_index("container_id")["threshold"]
        # Container 0 has divergence so its threshold should differ from container 1
        assert thresholds[0] != thresholds[1]

    def test_self_distance_is_zero(self, hvac_df_with_anomaly):
        unit_df, _ = self._get_flagged(hvac_df_with_anomaly)
        for unit in [0, 1, 2]:
            mask = unit_df["unit"] == unit
            assert (unit_df.loc[mask, f"dist_{unit}"] == 0.0).all()

    def test_only_both_pairs_flagged(self, hvac_df_with_anomaly):
        """Only unit 1 should be flagged (both its pairs breach); units 0 and 2 should not."""
        unit_df, _ = self._get_flagged(hvac_df_with_anomaly, strategy="mad", k=3)
        flagged_units = unit_df[unit_df["anomaly_flag"]]["unit"].unique()
        # Unit 1 diverges → both pairs (0_1 and 1_2) breach → unit 1 flagged
        assert 1 in flagged_units
        # Units 0 and 2: only ONE of their pairs breaches (e.g. unit 0 has 0_1 breach
        # but 0_2 stays normal) → should NOT be flagged
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
