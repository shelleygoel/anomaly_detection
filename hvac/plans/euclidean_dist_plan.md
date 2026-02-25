# Plan: Rolling Euclidean Distance + Threshold-Based Anomaly Flagging

## Context
The blog post notebook needs Euclidean distance-based anomaly detection. Implementing as a reusable utils module (`hvac/utils/euclidean_dist.py`) rather than inline notebook code, consistent with existing utils pattern (`visual.py`, `hvac_data_gen.py`).

## File Created
- `hvac/utils/euclidean_dist.py`

## Implementation

### Function 1: `rolling_euclidean_distance(series_a, series_b, window)`
Pure computation helper. Two aligned Series + window → rolling Euclidean distance Series.
```python
def rolling_euclidean_distance(series_a, series_b, window):
    return np.sqrt(((series_a - series_b) ** 2).rolling(window).sum())
```

### Function 2: `compute_pairwise_distances(df, smooth_window=10, dist_window=60)`
Takes raw HVAC dataset, returns DataFrame with rolling distances for all container-pair combinations.

**Logic:**
1. Loop over containers (`df.groupby('container_id')`)
2. Smooth TmpRet per unit (rolling mean, window=`smooth_window`)
3. Pivot to wide format (index=timestamp, columns=unit, values=TmpRet_smooth)
4. For each pair (0-1, 0-2, 1-2): call `rolling_euclidean_distance()`
5. Return long-format DataFrame: `container_id, timestamp_et, pair, distance`

### Function 3: `flag_anomalies(dist_df, strategy="mad", k=None)`
Takes distance DataFrame from Function 2. Returns **tuple of two DataFrames**: `(unit_df, thresholds_df)`.

**Threshold strategies:**
- `"mad"`: threshold = median + k × MAD (MAD = median(|x - median(x)|) × 1.4826). Robust to outliers. Default `k=3`.
- `"iqr"`: threshold = Q3 + k × IQR. Default `k=1.5`.

**Threshold computed per container** — all pairs within a container share one threshold (pooled from all pair distances).

**`thresholds_df`** (one row per container):
| container_id | threshold | strategy |

**`unit_df`** (unit-level anomaly flags):
| timestamp_et | container_id | unit | dist_0 | dist_1 | dist_2 | anomaly_flag | model |

- `dist_X` = distance from this unit to unit X; self-distance = 0.
  - e.g. unit=1: `dist_0`=pair_0_1, `dist_1`=0, `dist_2`=pair_1_2
- `anomaly_flag` = True only when **both** pairs involving this unit exceed threshold:
  - unit 0: pairs 0_1 AND 0_2 both above threshold
  - unit 1: pairs 0_1 AND 1_2 both above threshold
  - unit 2: pairs 0_2 AND 1_2 both above threshold
- `model` = `"euclidean_distance_{strategy}"`

**Implementation:**
1. Compute threshold per container (pool all pair distances)
2. Build `thresholds_df` from per-container thresholds
3. Pivot `dist_df` from long (pair rows) → wide (one row per timestamp×container) with pair distance columns
4. For each unit, map pair distances to `dist_0, dist_1, dist_2` (self=0), compute `anomaly_flag` from both-pairs logic
5. Concat unit rows, return `(unit_df, thresholds_df)`

## Verification
In notebook, import and call:
```python
from hvac.utils.euclidean_dist import compute_pairwise_distances, flag_anomalies
dist_df = compute_pairwise_distances(hvac_dataset)
unit_df, thresholds_df = flag_anomalies(dist_df, strategy="mad", k=3)
```
Confirm unit_df has expected columns, only the divergent unit is flagged, and thresholds_df has one row per container.
