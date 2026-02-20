"""
Pairwise anomaly detection for HVAC sensor data.

Computes pairwise rolling features (Euclidean distance or correlation)
between HVAC units and flags anomalies using robust statistical thresholds.
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


def compute_pairwise_correlations(df, smooth_window=10, corr_window=60):
    """
    Compute pairwise rolling correlations between all HVAC units per container.

    Returns 1 - correlation as the "distance" so higher values indicate
    more anomalous (less correlated) behavior. Output format is identical
    to compute_pairwise_distances for drop-in use with score_anomalies().

    Args:
        df: DataFrame with columns [container_id, timestamp_et, unit, TmpRet]
        smooth_window: rolling mean window for smoothing raw temperatures
        corr_window: rolling window for correlation computation

    Returns:
        Long-format DataFrame with columns [container_id, timestamp_et, pair, distance]
    """
    unit_pairs = [(0, 1), (0, 2), (1, 2)]
    results = []

    for container_id, grp in df.groupby("container_id"):
        smoothed = grp.copy()
        smoothed["TmpRet_smooth"] = smoothed.groupby("unit")["TmpRet"].transform(
            lambda x: x.rolling(smooth_window).mean()
        )

        pivot = smoothed.pivot_table(
            index="timestamp_et", columns="unit", values="TmpRet_smooth"
        )

        for i, j in unit_pairs:
            if i not in pivot.columns or j not in pivot.columns:
                continue
            corr = pivot[i].rolling(corr_window).corr(pivot[j])
            pair_df = pd.DataFrame({
                "container_id": container_id,
                "timestamp_et": pivot.index,
                "pair": f"{i}_{j}",
                "distance": (1 - corr).values,
            })
            results.append(pair_df)

    return pd.concat(results, ignore_index=True)


def flag_anomalies(dist_df, strategy="mad", k=None, model_name=None):
    """
    Flag anomalous (container, day) tuples where the anomaly score exceeds k.

    Uses score_anomalies() to compute continuous scores, then thresholds at k.

    Args:
        dist_df: DataFrame from compute_pairwise_distances or compute_pairwise_correlations
                 (columns: container_id, timestamp_et, pair, distance)
        strategy: "mad" or "iqr"
        k: score threshold (default 3 for MAD, 1.5 for IQR)
        model_name: optional model label (default: "euclidean_distance_{strategy}")

    Returns:
        DataFrame with columns [container_id, day, anomaly_score, anomaly_flag, model]
        One row per (container, day).
    """
    if k is None:
        k = 3 if strategy == "mad" else 1.5

    scores_df = score_anomalies(dist_df, strategy=strategy, model_name=model_name)
    scores_df["anomaly_flag"] = scores_df["anomaly_score"] > k
    return scores_df


def score_anomalies(dist_df, strategy="mad", model_name=None):
    """
    Compute continuous anomaly scores per (container, day) using pairwise distances.

    Scores each pair using MAD or IQR z-scores with a global baseline (all containers),
    then takes the max across all pairs as the container-day anomaly score.

    Args:
        dist_df: DataFrame from compute_pairwise_distances or compute_pairwise_correlations
                 (columns: container_id, timestamp_et, pair, distance)
        strategy: "mad" or "iqr"
        model_name: optional model label (default: "euclidean_distance_{strategy}")

    Returns:
        DataFrame with columns [container_id, day, anomaly_score, model]
        One row per (container, day).
    """
    if strategy not in ("mad", "iqr"):
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'mad' or 'iqr'.")

    dist_df = dist_df.copy()
    dist_df["day"] = dist_df["timestamp_et"].dt.date

    # Step 1: day-level mean distance per pair for ALL containers
    day_pair_all = (
        dist_df.groupby(["container_id", "pair", "day"])["distance"]
        .mean()
        .reset_index(name="day_dist")
    )

    # Step 2: compute baseline stats from the global pool (all containers)
    pool = day_pair_all["day_dist"].dropna()

    if strategy == "mad":
        median = pool.median()
        mad = (pool - median).abs().median()
        scale = mad * 1.4826 if mad > 0 else 1.0
        day_pair_all["score"] = (day_pair_all["day_dist"] - median) / scale
    else:  # iqr
        q1 = pool.quantile(0.25)
        q3 = pool.quantile(0.75)
        iqr = q3 - q1
        scale = iqr if iqr > 0 else 1.0
        day_pair_all["score"] = (day_pair_all["day_dist"] - q3) / scale

    # Step 3: aggregate to container-day level — max across all pairs
    result = (
        day_pair_all.groupby(["container_id", "day"])["score"]
        .max()
        .reset_index(name="anomaly_score")
    )
    if model_name is None:
        model_name = f"euclidean_distance_{strategy}"
    result["model"] = model_name
    return result[["container_id", "day", "anomaly_score", "model"]]
