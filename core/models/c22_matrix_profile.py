"""Catch22 Matrix Profile anomaly detection model.

Two-step API:
  1. fit_profile(feat_ds, weights) -> (mp_ds, debug)
     Compute the left matrix profile per entity over catch22 features.
     Expensive; run once.
  2. score(mp_ds, level="day"|"timestamp") -> scored TimeSeriesDataset
     Z-normalize per entity, optionally aggregate to day level.
     Cheap; call multiple times with different levels.
"""

import numba
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from core.dataset import TimeSeriesDataset
from core.models.base import AnomalyModel


@numba.njit(cache=True, fastmath=True, parallel=True)
def _left_c22_mp_kernel(
    X: np.ndarray,
    exclude_zone: int,
    n_warmup: int,
    early_abandon: bool,
    look_back_window: int,      # -1 means None
    dynamic_bsf_update: bool,
    bsf_lookback: int,
):
    """Numba-compiled core of the left catch22 matrix profile.

    Returns (mp, n_skipped, n_full_back, bsf_history) — all numpy arrays /
    scalars.  The class method wraps this into the debug dict shape.

    Brute-force phases use prange (parallel across subsequences).  The ORR
    phase is sequential because bsf is updated across iterations.
    """
    n_subseq, n_features = X.shape
    mp = np.full(n_subseq, np.nan)
    bsf_hist = np.empty(n_subseq, dtype=np.float64)
    h = 0
    n_skipped = 0
    n_full_back = 0

    first_scorable = exclude_zone + 1
    if first_scorable >= n_subseq:
        return mp, n_skipped, n_full_back, bsf_hist[:0]

    warmup_end = min(first_scorable + n_warmup, n_subseq)

    # Phase 1: brute-force warmup — each i independent
    for i in numba.prange(first_scorable, warmup_end):
        best = np.inf
        for j in range(i - 1 - exclude_zone, -1, -1):
            d = 0.0
            for k in range(n_features):
                diff = X[i, k] - X[j, k]
                d += diff * diff
            d = np.sqrt(d)
            if d < best:
                best = d
        mp[i] = best

    if warmup_end >= n_subseq or not early_abandon:
        # Short series or brute-force requested — parallelize the rest
        for i in numba.prange(warmup_end, n_subseq):
            best = np.inf
            for j in range(i - 1 - exclude_zone, -1, -1):
                d = 0.0
                for k in range(n_features):
                    diff = X[i, k] - X[j, k]
                    d += diff * diff
                d = np.sqrt(d)
                if d < best:
                    best = d
            mp[i] = best
        return mp, n_skipped, n_full_back, bsf_hist[:0]

    # Phase 2: ORR with early abandon — sequential (bsf updates across iterations)
    warm_sorted = np.sort(mp[first_scorable:warmup_end])
    pct_idx = int(0.05 * (len(warm_sorted) - 1))
    bsf = warm_sorted[pct_idx]
    bsf_hist[h] = bsf
    h += 1

    for i in range(warmup_end, n_subseq):
        lbsf = np.inf
        look_back_limit = i - look_back_window if look_back_window >= 0 else 0

        handled = False
        for j in range(i - 1 - exclude_zone, -1, -1):
            d = 0.0
            for k in range(n_features):
                diff = X[i, k] - X[j, k]
                d += diff * diff
            d = np.sqrt(d)

            if d < lbsf:
                lbsf = d

            if d < bsf and j > look_back_limit:
                n_skipped += 1
                handled = True
                break
            elif d >= bsf and j <= look_back_limit:
                n_full_back += 1
                if dynamic_bsf_update and i > warmup_end + 2:
                    lo = max(0, i - bsf_lookback)
                    cnt = 0
                    for v in mp[lo:i]:
                        if not np.isnan(v):
                            cnt += 1
                    if cnt > 0:
                        recent = np.empty(cnt, dtype=np.float64)
                        idx = 0
                        for v in mp[lo:i]:
                            if not np.isnan(v):
                                recent[idx] = v
                                idx += 1
                        candidate = np.median(recent)
                        hist_q = np.quantile(bsf_hist[:h], 0.7)
                        if candidate > hist_q:
                            bsf = candidate
                        # else: keep current bsf
                    else:
                        bsf = lbsf
                else:
                    bsf = lbsf
                bsf_hist[h] = bsf
                h += 1
                handled = True
                break
            # else: d < bsf and j <= look_back_limit — loop continues

        if not handled:
            # Inner loop exhausted without break
            n_full_back += 1
            bsf = lbsf
            bsf_hist[h] = bsf
            h += 1

        mp[i] = lbsf

    return mp, n_skipped, n_full_back, bsf_hist[:h]


class Catch22MPModel(AnomalyModel):
    """Anomaly detection via Left Catch22 Matrix Profile.

    For each entity, computes the left matrix profile over the catch22 feature
    profile: each subsequence's nearest-neighbor distance to all previous
    subsequences using L2 norm.  High MP values indicate subsequences unlike
    anything seen before — potential anomalies.

    All integer "length" params below are in units of *subsequences* (rows of
    the feature profile), not raw timestamps.  If the feature profile was
    built with stride=s, one subsequence spans s raw timestamps.

    Args:
        exclude_zone: Trivial match exclusion radius.  Subsequences within this
            many positions of the query are skipped.  Pin this to cover at
            least the expected anomaly duration (in subsequences) to avoid
            self-matching within a long anomaly.
        early_abandon: If True, use the ORR algorithm (early stopping when a
            match closer than BSF is found).  If False, brute-force 1-NN.
        n_warmup: Number of initial scored subsequences computed with brute
            force to establish the BSF threshold.
        look_back_window: If set, limits backward search to this many steps.
        dynamic_bsf_update: If True, update BSF using median of recent MP
            values instead of the 1-NN distance on full-backtrack events.
        bsf_lookback: Size of the recent-MP window used when
            dynamic_bsf_update is True.  Ignored otherwise.
    """

    DAY_AGG_STATS = {"max", "p90", "mean"}

    def __init__(
        self,
        exclude_zone: int = 1,
        early_abandon: bool = True,
        n_warmup: int = 100,
        look_back_window: int | None = None,
        dynamic_bsf_update: bool = False,
        bsf_lookback: int = 1000,
    ):
        self.exclude_zone = exclude_zone
        self.early_abandon = early_abandon
        self.n_warmup = n_warmup
        self.look_back_window = look_back_window
        self.dynamic_bsf_update = dynamic_bsf_update
        self.bsf_lookback = bsf_lookback

    def fit_profile(
        self,
        dataset: TimeSeriesDataset,
        weights: dict[str, float] | None = None,
    ) -> tuple[TimeSeriesDataset, dict]:
        """Compute the left catch22 matrix profile per entity.

        Args:
            dataset: Feature profile from FeatureTransformer.transform().
                value_cols are the catch22 feature columns.
            weights: Dict mapping feature name to weight.  If None, equal weights.
                Features with weight ~0 are dropped.  Only features present in
                both weights and dataset.value_cols are used.

        Returns:
            Tuple of:
              - TimeSeriesDataset with value_cols=['left_c22_mp'] at timestamp level.
              - Debug dict: entity_id -> {n_skipped, n_full_back, bsf_history}.
        """
        entity_col = dataset.col_map["entity"]
        time_col = dataset.col_map["time"]
        value_cols = dataset.col_map["value_cols"]

        if weights is not None:
            use_cols = [c for c in value_cols if c in weights and not np.isclose(weights[c], 0.0)]
            weight_arr = np.array([weights[c] for c in use_cols])
        else:
            use_cols = value_cols
            weight_arr = None

        profile_rows = []
        debug: dict = {}

        for entity_id, grp in tqdm(
            dataset.df.groupby(entity_col), desc="Catch22MP fit_profile"
        ):
            grp = grp.sort_values(time_col)
            X = grp[use_cols].values.astype(np.float64)

            # Min-max scale per feature to [0, 1] so no single feature dominates
            # the L2 distance purely due to having a larger natural scale.
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            X = (X - X_min) / (X_max - X_min + 1e-12)

            if weight_arr is not None:
                X = X * (weight_arr / weight_arr.sum())

            mp_values, debug_info = self._left_c22_mp(X)
            debug[entity_id] = debug_info

            profile_rows.append(pd.DataFrame({
                entity_col: entity_id,
                time_col: grp[time_col].values,
                "left_c22_mp": mp_values,
            }))

        profile_df = pd.concat(profile_rows, ignore_index=True)
        mp_ds = TimeSeriesDataset(
            profile_df,
            {"entity": entity_col, "time": time_col, "value_cols": ["left_c22_mp"]},
        )
        return mp_ds, debug

    def score(
        self,
        mp_ds: TimeSeriesDataset,
        level: str = "day",
        day_agg_stat: str = "max",
    ) -> TimeSeriesDataset:
        """Convert a matrix profile to anomaly scores.

        Z-normalizes per entity so different entities are comparable, then
        optionally aggregates to day level.

        Args:
            mp_ds: Output of fit_profile — value_cols must include 'left_c22_mp'.
            level: "day" or "timestamp".
            day_agg_stat: How to aggregate timestamp-level z-scores to day
                level.  "max", "p90", or "mean".  Ignored when level="timestamp".

        Returns:
            TimeSeriesDataset with value_cols=['left_c22_mp', 'anomaly_score'].
        """
        if level not in ("day", "timestamp"):
            raise ValueError(f"level must be 'day' or 'timestamp', got {level!r}")
        if day_agg_stat not in self.DAY_AGG_STATS:
            raise ValueError(
                f"Unknown day_agg_stat: {day_agg_stat!r}. "
                f"Use one of {self.DAY_AGG_STATS}."
            )

        entity_col = mp_ds.col_map["entity"]
        time_col = mp_ds.col_map["time"]

        df = mp_ds.df.copy()
        grp_stats = df.groupby(entity_col)["left_c22_mp"].agg(
            _mp_mean="mean", _mp_std="std"
        )
        df = df.join(grp_stats, on=entity_col)
        df["anomaly_score"] = df["left_c22_mp"]
        # safe_std = df["_mp_std"].where(df["_mp_std"] > 0, 1.0)
        # df["anomaly_score"] = np.where(
        #     df["_mp_std"] > 0,
        #     (df["left_c22_mp"] - df["_mp_mean"]) / safe_std,
        #     0.0,
        # )
        df = df.drop(columns=["_mp_mean", "_mp_std"])

        if level == "timestamp":
            return TimeSeriesDataset(
                df[[entity_col, time_col, "left_c22_mp", "anomaly_score"]],
                {"entity": entity_col, "time": time_col,
                 "value_cols": ["left_c22_mp", "anomaly_score"]},
            )

        # Day-level aggregation
        df["day"] = pd.to_datetime(df[time_col]).dt.date
        agg_func = self._get_agg_func(day_agg_stat)
        day_df = (
            df.groupby([entity_col, "day"])
            .agg(left_c22_mp=("left_c22_mp", agg_func),
                 anomaly_score=("anomaly_score", agg_func))
            .reset_index()
        )
        return TimeSeriesDataset(
            day_df,
            {"entity": entity_col, "time": "day",
             "value_cols": ["left_c22_mp", "anomaly_score"]},
        )

    def _get_agg_func(self, stat: str):
        if stat == "max":
            return "max"
        elif stat == "p90":
            return lambda x: np.percentile(x, 90)
        elif stat == "mean":
            return "mean"

    def _left_c22_mp(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Thin wrapper over the Numba kernel.

        Returns (mp_values, debug_info) — subsequences without enough history
        get NaN in mp_values.
        """
        lbw = -1 if self.look_back_window is None else self.look_back_window
        mp, n_skipped, n_full_back, bsf_history = _left_c22_mp_kernel(
            np.ascontiguousarray(X),
            self.exclude_zone,
            self.n_warmup,
            self.early_abandon,
            lbw,
            self.dynamic_bsf_update,
            self.bsf_lookback,
        )
        return mp, {
            "n_skipped": int(n_skipped),
            "n_full_back": int(n_full_back),
            "bsf_history": bsf_history.tolist(),
        }
