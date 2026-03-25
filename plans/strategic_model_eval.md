# Anomaly Detection Framework: Evaluation Harness + Debugger + Data Structure

## Context
Debugging and evaluating anomaly detection models is currently ad-hoc across notebooks. Need reusable, domain-agnostic classes that work for any time series anomaly detection problem — not just HVAC. Four components: `TimeSeriesDataset`, `AnomalyModel`, `Evaluation`, `ModelDebug`.

## File Structure
```
anomaly_detection/core/
  __init__.py
  dataset.py       # TimeSeriesDataset
  model.py         # AnomalyModel
  evaluation.py    # Evaluation
  debug.py         # ModelDebug
```

Why `core/` (not `hvac/utils/`): these classes are domain-agnostic. HVAC-specific code stays in `hvac/utils/`.

---

## Component 1: `TimeSeriesDataset` — `core/dataset.py`

Thin wrapper over DataFrame. Maps semantic roles to column names via `col_map`.

```python
class TimeSeriesDataset:
    def __init__(self, df, col_map):
        # col_map required keys: 'entity', 'time', 'value_cols'
        # col_map optional keys: 'label', 'label_type', 'sub_entity'

    def day_labels(self) -> pd.DataFrame:
        # Returns [entity_col, 'day', 'label', 'label_type']
        # max() aggregation, mirrors 02_debug_performance pattern
        # Raises if 'label' not in col_map

    def ts_labels(self) -> pd.DataFrame:
        # Returns [entity_col, time_col, 'label', 'label_type']
        # Timestamp-level: max label across sub_entities per timestamp
        # Raises if 'label' not in col_map

    def anomaly_types(self) -> np.ndarray

    def sample_and_visualize_cases(self, n_cases=1, entity_ids=None,
                                    label_type=None) -> go.Figure:
        # Generic version of plot_container_anomaly_timeseries
        # Samples n_cases random entities (or uses explicit entity_ids)
        # If label_type provided, samples only from entities with that label_type
        # Creates n_cases subplot rows, each with:
        #   - Primary y: one line per sub_entity for each value_col, sorted by time_col
        #   - Secondary y: anomaly markers (if 'label' in col_map)
        #     - Per sub_entity markers if label_type is unit-level (lag, frequency)
        #     - Single marker trace if label_type is global (amplitude)
        # Uses consistent colors across subplots for sub_entity groups
        # Returns Plotly figure
```

HVAC usage:
```python
col_map = {
    'entity': 'container_id', 'time': 'timestamp_et',
    'value_cols': ['TmpRet'],
    'label': 'anomaly', 'label_type': 'anomaly_type',
    'sub_entity': 'unit',
}
dataset = TimeSeriesDataset(hvac_df, col_map)

# Equivalent to plot_container_anomaly_timeseries(hvac_df, anomaly_type="lag", num_containers=2)
dataset.sample_and_visualize_cases(n_cases=2, label_type="lag")
```

---

## Component 2: `AnomalyModel` — `core/model.py`

Wraps model scores + metadata about features.

```python
class AnomalyModel:
    def __init__(self, scores_df, model_name, entity_col, time_or_day_col,
                 score_col='anomaly_score',
                 features_used=None, feature_descriptions=None)

    @classmethod
    def from_scores_df(cls, df, col_map, model_name=None, **kwargs):
        # Accepts existing eucl_dist.score_anomalies() output directly

    def day_scores(self, agg='max') -> pd.DataFrame
    def describe(self) -> str
```

Zero-friction adoption — existing scoring code unchanged, just wrap output:
```python
scores_df = eucl_dist.score_anomalies(dist_df)
model = AnomalyModel.from_scores_df(scores_df, col_map=dataset.col_map,
    features_used=['pairwise_euclidean_distance'],
    feature_descriptions={'pairwise_euclidean_distance': 'Rolling Euclidean distance between unit pairs'})
```

---

## Component 3: `Evaluation` — `core/evaluation.py`

Joins labels + scores, computes metrics, plots curves.

```python
class Evaluation:
    def __init__(self, dataset, model, level='day'):
        # Merges dataset.day_labels() or ts_labels() with model scores

    def auc_pr(self, anomaly_type=None) -> float
    def auc_roc(self, anomaly_type=None) -> float

    def metrics_table(self, anomaly_types=None) -> pd.DataFrame:
        # Returns [anomaly_type, auc_pr, auc_roc] per type

    def plot_pr_curve(self, anomaly_types=None) -> go.Figure:
        # PR curve subplots, one per anomaly type, with AUC annotation

    def plot_roc_curve(self, anomaly_types=None) -> go.Figure

    @staticmethod
    def compare(evaluations: list) -> pd.DataFrame:
        # Multi-model comparison table: rows=anomaly_type, cols=model_name

    @staticmethod
    def plot_pr_curves_compared(evaluations: list) -> go.Figure:
        # Overlay PR curves from multiple models, color-coded by model
```

Per-type filtering: keep all normals + target anomaly type (matches existing pattern).

---

## Component 4: `ModelDebug` — `core/debug.py`

Integrates with `TimeSeriesDataset` and `AnomalyModel` (replaces raw-array `model_debug_plan.md`).

```python
class ModelDebug:
    def __init__(self, dataset, model, level='day', seed=42):
        # Merges labels + scores internally

    def plot_score_separation(self, nbins=50) -> go.Figure:
        # Overlaid histograms: scores for label=0 vs label=1

    def plot_score_by_type(self, plot_type='box') -> go.Figure:
        # Box/violin of scores colored by label_type

    def plot_high_score_normals(self, n_samples=3, score_range=None) -> go.Figure:
        # Normal cases w/ high scores → raw TS + score subplot via dataset.plot_case()

    def plot_boundary_abnormals(self, n_samples=3, score_range=None) -> go.Figure:
        # Abnormal cases w/ low scores → same layout

    def _sample_and_plot(self, mask, n_samples, score_range,
                         default_quantile_range, title) -> go.Figure:
        # Shared: filter, sample entities, create 2*n row subplots (TS + score per case)
```

---

## Implementation Order

1. `core/__init__.py` — empty
2. `core/dataset.py` — `TimeSeriesDataset` (foundation)
3. `core/model.py` — `AnomalyModel` (standalone, lightweight)
4. `core/evaluation.py` — `Evaluation` (depends on 2+3)
5. `core/debug.py` — `ModelDebug` (depends on 2+3)
6. Refactor `02_debug_performance.ipynb` to use new classes

Steps 2 and 3 can be done in parallel; 4 and 5 can be done in parallel.

---

## Key Design Decisions

- **col_map pattern**: Maps semantic roles → column names. Avoids baking in HVAC column names.
- **Labels owned by Dataset, not Evaluation**: Dataset knows column mapping; Evaluation just consumes labels.
- **sample_and_visualize_cases**: Generic version of `plot_container_anomaly_timeseries` from `visual.py`. Samples entities, plots value_cols per sub_entity with anomaly overlays. ModelDebug's `_sample_and_plot` calls this internally for individual cases.
- **Existing code unchanged**: `euclidean_dist.py` stays as-is. `AnomalyModel.from_scores_df()` wraps its output.
- **No copy**: Dataset stores reference to df, not a copy.

---

## Verification

1. In `02_debug_performance.ipynb`, replace ad-hoc label/score/debug code with new classes
2. Confirm `Evaluation.metrics_table()` matches manually computed AUC-PR/ROC values
3. Confirm `ModelDebug` plots render correctly in notebook
4. Test `Evaluation.compare()` with euclidean distance vs IForest models side by side

---

## Unresolved Questions

1. **`plot_case()` anomaly window rendering**: Should anomaly windows be shaded rectangles (`add_vrect`) or marker overlays on secondary y-axis? Current notebook uses secondary y-axis markers. Shading is cleaner but needs start/end detection. **Proposal**: Use secondary y-axis markers (matches existing pattern), can upgrade later.

2. **`compare()` validation**: Should `Evaluation.compare()` verify all evaluations use the same dataset? **Proposal**: No enforcement — trust the user. Log a warning if entity counts differ.

3. **`Evaluation` level auto-detection**: User must pass `level='day'` or `level='timestamp'`. Should we auto-detect from model's `time_or_day_col` dtype (date vs datetime)? **Proposal**: Yes, auto-detect with manual override.

---

## How This Aligns with Real-World ML Workflows

### What aligns well
- Separation of data / model / evaluation / debugging — standard pattern in any ML pipeline
- Per-type evaluation — analogous to per-class metrics in classification
- PR/ROC curves with AUC — standard model selection tools
- Error analysis on failure cases (high-score normals, boundary abnormals) — mirrors confusion-matrix-driven debugging in classification

### What's deliberately deferred (deployment/productionization stage)
This plan covers **model selection and diagnostics**: "which model/feature combination works best for this anomaly type?" The following concerns belong to a subsequent deployment stage:

- **Train/test split**: Temporal holdout validation to avoid overconfident results. For model selection on synthetic data with known ground truth, this is less critical — but should be called out as a caveat.
- **Threshold selection**: PR/ROC curves show potential; shipping a model requires an operating point (e.g., target precision → find recall). Tabled for now.
- **Confusion matrix at operating point**: TP/FP/TN/FN counts at a chosen threshold — the most basic deployment diagnostic.
- **Slice-based evaluation**: Performance across subpopulations (time periods, operational modes, data quality) beyond anomaly type.
- **Experiment tracking**: MLflow or similar for persistent logging of model comparisons (see `evaluation_harness.md`).
- **Model as callable**: Making `AnomalyModel` wrap `.fit()` / `.score()` for systematic retraining and tuning.
