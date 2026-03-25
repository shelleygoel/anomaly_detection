#ModelDebug Class — Domain-Agnostic Anomaly Detection Diagnostics

## Context
Debugging anomaly detection models is currently ad-hoc (notebook `02_debug_performance.ipynb`). Need a reusable class that works for any anomaly detection model — not just HVAC/Euclidean distance. The class provides 4 diagnostic plots that answer: "why is my model underperforming?"

## New File
`/anomaly_detection/hvac/utils/model_debug.py`

## Class API

```python
class ModelDebug:
    def __init__(self, scores, labels, case_ids=None, anomaly_types=None,
                 timeseries_plotter=None, seed=42):
        # scores: continuous anomaly scores (any scale)
        # labels: binary 0/1 ground truth
        # case_ids: identifies each case (container_id, patient_id, etc.)
        # anomaly_types: multi-class strings ("lag", "frequency", "normal", etc.)
        # timeseries_plotter: callback with signature (fig, case_id, row) -> None
        # Builds internal self._df with columns: score, label, case_id, anomaly_type

    def plot_score_separation(self, nbins=50, opacity=0.6) -> go.Figure:
        # Plot 1: Overlaid histograms of scores for label=0 vs label=1

    def plot_feature_by_type(self, df, feature_col, type_col="anomaly_type",
                             plot_type="box") -> go.Figure:
        # Plot 2: Boxplot (or violin) of any feature column colored by anomaly type
        # Standalone — takes external DataFrame, not self._df

    def plot_high_score_normals(self, n_samples=3, score_range=None) -> go.Figure:
        # Plot 3: Sample normal cases from high-score region, plot raw timeseries
        # Auto-detect: top 25% of normal scores (Q75 to max)

    def plot_boundary_abnormals(self, n_samples=3, score_range=None) -> go.Figure:
        # Plot 4: Sample abnormal cases from low-score region, plot raw timeseries
        # Auto-detect: bottom 25% of abnormal scores (min to Q25)

    def _sample_and_plot(self, mask, n_samples, score_range,
                         default_quantile_range, title) -> go.Figure:
        # Shared logic for plots 3 & 4
```

## Key Design Decision: `timeseries_plotter` Callback

Plots 3 & 4 need raw timeseries but the class must stay domain-agnostic. Solution: user provides a callback that knows how to render a case.

**Contract:** `def plotter(fig: go.Figure, case_id: Any, row: int) -> None`
- Adds traces via `fig.add_trace(..., row=row, col=1)`
- Does NOT call `fig.show()` or modify layout

**HVAC example** (factory pattern, lives in `visual.py`):
```python
def make_hvac_plotter(hvac_dataset):
    unit_colors = {0: '#5470C6', 1: '#EE6666', 2: '#5DBCD2'}
    def _plot(fig, container_id, row):
        data = hvac_dataset[hvac_dataset['container_id'] == container_id].sort_values('timestamp_et')
        for unit in sorted(data['unit'].unique()):
            ud = data[data['unit'] == unit]
            fig.add_trace(
                go.Scatter(x=ud['timestamp_et'], y=ud['TmpRet'],
                           name=f'Unit {unit}', mode='lines',
                           line=dict(color=unit_colors.get(unit)),
                           showlegend=(row == 1), legendgroup=f'unit_{unit}'),
                row=row, col=1)
    return _plot
```

## Implementation Steps

1. Create `model_debug.py` with imports + class skeleton
2. `__init__` — validate array lengths, build `self._df`, store rng + plotter
3. `plot_score_separation` — two `go.Histogram` traces, `barmode="overlay"`
4. `plot_feature_by_type` — `px.box` or `px.violin` on user-provided df
5. `_sample_and_plot` — shared helper: filter by mask, determine score range (explicit or auto-quantile), sample case_ids, create subplots, call callback per case
6. `plot_high_score_normals` / `plot_boundary_abnormals` — thin wrappers
7. Add `make_hvac_plotter` factory to `visual.py`
8. Refactor `02_debug_performance.ipynb` to use `ModelDebug`

## Edge Cases to Handle
- Quantile range collapses (all scores identical): expand by epsilon or use full range
- No cases in auto-detected range: warn + widen to full group
- `timeseries_plotter` is None when plots 3/4 called: raise `ValueError`
- `case_ids` is None when plots 3/4 called: raise `ValueError`

## Files to Modify
- **Create:** `hvac/utils/model_debug.py`
- **Edit:** `hvac/utils/visual.py` (add `make_hvac_plotter`)
- **Edit:** `blog_posts/02_debug_performance.ipynb` (refactor to use ModelDebug)

## Verification
1. Run all 4 plots in `02_debug_performance.ipynb` using ModelDebug
2. Confirm plots render correctly in notebook
3. Test with `score_range=None` (auto) and explicit ranges

## Open Questions
1. Should `plot_feature_by_type` stay on the class or become a standalone function? (Leaning: keep on class for API cohesion)
2. Should `make_hvac_plotter` live in `visual.py` or `model_debug.py`? (Leaning: `visual.py` to keep model_debug domain-agnostic)
3. Day-level vs timestamp-level: class treats `case_id` opaquely — callback handles resolution. Sufficient?
