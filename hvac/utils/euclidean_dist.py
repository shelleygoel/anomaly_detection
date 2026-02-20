"""
Euclidean distance-based anomaly detection for HVAC sensor data.

Computes pairwise rolling Euclidean distances between HVAC units
and flags anomalies using robust statistical thresholds.
"""

import numpy as np
import pandas as pd


def rolling_euclidean_distance(series_a, series_b, window):
    """
    Compute rolling Euclidean distance between two time series.

    Args:
        series_a: pandas Series of temperature values
        series_b: pandas Series of temperature values
        window: rolling window size (number of timestamps)

    Returns:
        pandas Series of rolling Euclidean distances
    """
    return np.sqrt(((series_a - series_b) ** 2).rolling(window).sum())


def compute_pairwise_distances(df, smooth_window=10, dist_window=60):
    """
    Compute pairwise rolling Euclidean distances between all HVAC units per container.

    Args:
        df: DataFrame with columns [container_id, timestamp_et, unit, TmpRet]
        smooth_window: rolling mean window for smoothing raw temperatures
        dist_window: rolling window for Euclidean distance computation

    Returns:
        Long-format DataFrame with columns [container_id, timestamp_et, pair, distance]
    """
    unit_pairs = [(0, 1), (0, 2), (1, 2)]
    results = []

    for container_id, grp in df.groupby("container_id"):
        # Smooth temperatures per unit
        smoothed = grp.copy()
        smoothed["TmpRet_smooth"] = smoothed.groupby("unit")["TmpRet"].transform(
            lambda x: x.rolling(smooth_window).mean()
        )

        # Pivot to wide format: one column per unit
        pivot = smoothed.pivot_table(
            index="timestamp_et", columns="unit", values="TmpRet_smooth"
        )

        for i, j in unit_pairs:
            if i not in pivot.columns or j not in pivot.columns:
                continue
            dist = rolling_euclidean_distance(pivot[i], pivot[j], dist_window)
            pair_df = pd.DataFrame({
                "container_id": container_id,
                "timestamp_et": pivot.index,
                "pair": f"{i}_{j}",
                "distance": dist.values,
            })
            results.append(pair_df)

    return pd.concat(results, ignore_index=True)


def flag_anomalies(dist_df, strategy="mad", k=None):
    """
    Flag anomalous units based on pairwise distance thresholds per container.

    A unit is flagged anomalous only when BOTH its pairings exceed the threshold.

    Args:
        dist_df: DataFrame from compute_pairwise_distances
                 (columns: container_id, timestamp_et, pair, distance)
        strategy: "mad" (median absolute deviation) or "iqr" (interquartile range)
        k: multiplier for threshold (default 3 for MAD, 1.5 for IQR)

    Returns:
        (unit_df, thresholds_df) where:
        - unit_df: columns [timestamp_et, container_id, unit, dist_0, dist_1, dist_2,
                            anomaly_flag, model]
        - thresholds_df: columns [container_id, threshold, strategy]
    """
    if strategy not in ("mad", "iqr"):
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'mad' or 'iqr'.")

    dist_df = dist_df.copy()

    # Map from unit to its two pair names
    unit_pairs = {
        0: ("0_1", "0_2"),
        1: ("0_1", "1_2"),
        2: ("0_2", "1_2"),
    }
    # Map from (unit, pair) to the "other" unit in that pair
    pair_other = {
        (0, "0_1"): 1, (0, "0_2"): 2,
        (1, "0_1"): 0, (1, "1_2"): 2,
        (2, "0_2"): 0, (2, "1_2"): 1,
    }

    threshold_rows = []
    unit_frames = []

    for container_id, grp in dist_df.groupby("container_id"):
        all_dists = grp["distance"].dropna()

        if strategy == "mad":
            _k = 3 if k is None else k
            median = all_dists.median()
            mad = (all_dists - median).abs().median()
            threshold = median + _k * mad * 1.4826
        else:  # iqr
            _k = 1.5 if k is None else k
            q1 = all_dists.quantile(0.25)
            q3 = all_dists.quantile(0.75)
            iqr = q3 - q1
            threshold = q3 + _k * iqr

        threshold_rows.append({
            "container_id": container_id,
            "threshold": threshold,
            "strategy": strategy,
        })

        # Pivot pair distances to wide: one row per timestamp with dist columns per pair
        wide = grp.pivot_table(
            index="timestamp_et", columns="pair", values="distance"
        )

        for unit in (0, 1, 2):
            pair_a, pair_b = unit_pairs[unit]
            if pair_a not in wide.columns or pair_b not in wide.columns:
                continue

            # Map each "other" unit to the pair that connects it to this unit
            other_to_pair = {
                pair_other[(unit, pair_a)]: pair_a,
                pair_other[(unit, pair_b)]: pair_b,
            }

            unit_frame = pd.DataFrame({
                "timestamp_et": wide.index,
                "container_id": container_id,
                "unit": unit,
            })
            # Fill dist columns: self=0, others from pair distances
            for u in (0, 1, 2):
                if u == unit:
                    unit_frame[f"dist_{u}"] = 0.0
                else:
                    unit_frame[f"dist_{u}"] = wide[other_to_pair[u]].values

            unit_frame["anomaly_flag"] = (
                (wide[pair_a].values > threshold)
                & (wide[pair_b].values > threshold)
            )
            unit_frame["model"] = f"euclidean_distance_{strategy}"
            unit_frames.append(unit_frame)

    thresholds_df = pd.DataFrame(threshold_rows)
    unit_df = pd.concat(unit_frames, ignore_index=True)
    return unit_df, thresholds_df
