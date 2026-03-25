# Model Evaluation Metrics: Event-AUC-PR + Event-AUC-ROC

## Why These Metrics

We only have **day-level labels** (an entire day is marked anomalous, not exact timestamps). This mirrors real-world operations where operators report daily issues, not precise anomaly boundaries.

**Requirements for the primary metric:**
- Works with coarse (day-level) labels
- Penalizes false alarms (reduces operator alert fatigue)
- Threshold-free (no need to pick a decision boundary)

**Why not VUS-PR?** VUS-PR rewards temporal proximity — its precision machinery is designed for exact anomaly boundaries. With day-level labels, that precision is unavailable and wasted.

**Why not Event-F1?** Threshold-dependent. Requires choosing a cutoff to convert scores into binary predictions, adding an arbitrary step that can skew results.

**Why not point-level AUC-ROC?** Doesn't match our evaluation granularity. We care about day-level detection, not point-level discrimination.

## Metric Mechanics

### Day-Level Aggregation

Both metrics operate on **day-aggregated** scores and labels:

```python
# Aggregate to day level
day_scores = df.groupby('day')['anomaly_score'].max()   # worst score per day
day_labels = df.groupby('day')['label'].max()            # 1 if any anomaly that day
```

Using `max()` for scores: if any point in the day looks anomalous, the day gets a high score.
Using `max()` for labels: if any anomaly occurs during the day, the day is labeled anomalous.

### Primary: Event-AUC-PR

```python
from sklearn.metrics import average_precision_score
event_auc_pr = average_precision_score(day_labels, day_scores)
```

- Area under the precision-recall curve on day-level data
- **Stricter metric** — directly penalizes false alarms via precision
- Honest for imbalanced data (few anomalous days vs many normal days)
- Threshold-free: evaluates the full ranking of day scores

### Secondary: Event-AUC-ROC

```python
from sklearn.metrics import roc_auc_score
event_auc_roc = roc_auc_score(day_labels, day_scores)
```

- Area under the ROC curve on day-level data
- Standard discrimination measure
- **More forgiving** — can be misleadingly optimistic with imbalanced classes
- Included for educational comparison: shows how metric choice affects perceived quality

### Why Show Both

Showing both metrics is itself educational:
- **AUC-ROC will be higher across the board** — it credits true negatives, which are plentiful with imbalanced data
- **Event-AUC-PR is the honest metric** — it focuses only on how well the detector handles positives
- The gap between them reveals how much the class imbalance flatters the detector

## Changes Needed

### Data Generator (`hvac_data_gen.py`)

- **Simplify labeling:** Remove point-wise `anomaly=True` from inject methods
- **New approach:** Derive `anomaly` column from `anomaly_config` at day level
- **Labeling rule:** Mark entire days as `1` for any (unit, day) pair with an anomaly occurring during that day

### Notebook (`01_model_selection_workflow.ipynb`)

**Evaluation section narrative:**
> "Operators report daily issues, not exact timestamps. We aggregate anomaly scores to the day level and use threshold-free metrics: Event-AUC-PR (strict, penalizes false alarms) and Event-AUC-ROC (more forgiving, for comparison)."

**Metrics to compute:**
- **Primary:** Event-AUC-PR via `average_precision_score(day_labels, day_scores)`
- **Secondary:** Event-AUC-ROC via `roc_auc_score(day_labels, day_scores)`
- **Drop:** VUS-PR (needs precise boundaries), Event-F1 (threshold-dependent)

**Results table format:**

| Anomaly Type | Method | Event-AUC-PR | Event-AUC-ROC |
|--------------|--------|--------------|---------------|
| Lag | Euclidean Dist | high | high |
| Lag | IForest | low | low |
| Frequency | Euclidean Dist | mid | mid |
| Frequency | IForest | mid | mid |
| Amplitude | Euclidean Dist | low | low |
| Amplitude | IForest | high | high |

## Key Files

| File | Role |
|------|------|
| `sklearn.metrics` | `average_precision_score`, `roc_auc_score` |
| `anomaly_detection/hvac/hvac_data_gen.py` | Modify labeling to day-level |
| `anomaly_detection/hvac/01_model_selection_workflow.ipynb` | Evaluation section |
