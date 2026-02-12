# Post 1: End-to-End Anomaly Detection Model Selection Workflow

## Context

**Goal:** Create a Jupyter notebook that demonstrates a complete workflow for choosing the best anomaly detection model for a given problem. Use HVAC sensor data as the running example, but the workflow is general-purpose for any time series anomaly detection task.

**Audience:** Anyone working with time series anomaly detection.

**Key takeaway:** Different anomaly types require different detection approaches. A systematic evaluation workflow helps you pick the right tool.

**Evaluation Metrics:** See [metric_evaluation_plan.md](metric_evaluation_plan.md)

**Two methods compared:**
- **Euclidean Distance** - pairwise rolling distance between correlated sensors (threshold-based)
- **Isolation Forest** - multi-dimensional feature-space isolation (via TSB-AD)

**Dataset Used**
- Synthetically generated dataset of BESS containers w/ each container having 3 liquid cooling HVAC units.
- multiday per container
- Return air temperature - Uni-variate per HVAC unit/ 

**Three anomaly types:**
| Type | Description | Which method wins |
|------|-------------|-------------------|
| **Lag** | One unit responds late (time-shifted) | Euclidean Distance (clear inter-unit divergence) |
| **Frequency** | One unit cycles at wrong rate | Both detect it, different mechanisms |
| **Amplitude** | ALL units' peak-to-trough range shrinks | IForest (inter-unit distance stays normal since all units are affected equally; IForest catches unusual absolute values) |


---

## What Exists Today

- **`hvac_data_gen.py`** (`anomaly_detection/hvac/hvac_data_gen.py`) - Generator with `lag` and `frequency` anomaly types. Needs new `amplitude` anomaly type.
- **`hvac_data_eda.ipynb`** - Has rolling Euclidean distance computation and IForest via TSB-AD. Reuse these patterns.
- **TSB-AD library** - `run_Unsupervise_AD('IForest', data)` for IForest. Evaluation uses `sklearn.metrics` (Event-AUC-PR, Event-AUC-ROC) on day-aggregated data.

---

## Implementation Plan

### Step 1: Add amplitude anomaly to `hvac_data_gen.py`

Add `inject_anomaly_amplitude()` method to `HVACDataGenerator`:
- Takes a `scale_factor` param (e.g., 0.3 = 30% of normal range)
- For the anomaly window: scale deviations from base_temp by `scale_factor` and add back to base_temp
- Apply to ALL 3 units in a container (this is what makes it a "global" anomaly)

Modify `generate_container_data()` to support `amplitude` type in the anomaly config, applying it to all units.

Also update `_generate_random_anomaly_config()` to include amplitude as a possible random anomaly type.

**File:** `anomaly_detection/hvac/hvac_data_gen.py`

### Step 2: Generate a synthethic dataset
Use `HVADataGenerator` to generate a synthetic dataset of 10 containers across 2 weeks which is injected with anomalies of all three types.
 5% unit-days for each anomaly type. (A unit-day is data of a single HVAC unit for a day)
Store the data in `hvac/datasets` folder.   

### Step 3: Create Data Visulization helper methods in `hvac/utils/visual.py`
- Method to visualize % of total unit-days grouped by anomaly types.
- Method to visualize TempRet of a container across multiple days - colored by HVAC unit 
- TODO: Plan more methods for visualizing flagged anomalies per container 

**File:** `anomaly_detection/hvac/01_model_selection_workflow.ipynb`

#### Section 1: Setup & Data Generation
- Imports (hvac_data_gen, TSB-AD, plotly/matplotlib, numpy, pandas)
- Load and visualize synthetic dataset
- Use Visualization helper methods from `hvac/utils/visual.py`


#### Section 2: Method 1 - Rolling Euclidean Distance
- Create a helper method to calculate rolling euclidean distance which takes raw hvac data as input
- Compute for all 3 unit pairs per container
- Choose a threshold
  - mean + k*std
  - Q1 + k * IQR
- Apply threshold to get binary predictions
- Flag anomalies
- Show the distance time series overlaid with ground truth anomaly windows

#### Section 3: Method 2 - Isolation Forest
Use TSB-AD wrapper (from `model_wrapper.py`):
```python
from TSB_AD.model_wrapper import run_Unsupervise_AD
scores = run_Unsupervise_AD('IForest', data)
```
- **Features:** raw temperatures of all 3 units (3-column matrix per container)
- Run IForest per container
- Show anomaly scores overlaid with ground truth

#### Section 4: Evaluation

See [metrics_plan.md](metrics_plan.md) for full evaluation strategy.

**Day-level aggregation** — scores and labels are aggregated to the day level before computing metrics:
```python
from sklearn.metrics import average_precision_score, roc_auc_score

# Aggregate to day level
day_scores = df.groupby('day')['anomaly_score'].max()
day_labels = df.groupby('day')['label'].max()

# Compute metrics
event_auc_pr = average_precision_score(day_labels, day_scores)
event_auc_roc = roc_auc_score(day_labels, day_scores)
```

**Primary metric: Event-AUC-PR** — area under precision-recall curve on day-level data. Strict: directly penalizes false alarms. Honest for imbalanced data (few anomalous days vs many normal days). Threshold-free.

**Secondary metric: Event-AUC-ROC** — area under ROC curve on day-level data. More forgiving — can overstate quality with imbalanced classes. Included for educational comparison.

Compute both metrics for both methods on each container (6 evaluations total).

#### Section 5: Comparison & Insights
- **Results table:** rows = (container/anomaly type), columns = (method), cells = Event-AUC-PR and Event-AUC-ROC
- **Bar chart:** grouped by anomaly type, comparing the two methods
- **Key findings:**
  - Lag anomaly → Euclidean distance wins (divergence between units is the direct signal)
  - Frequency anomaly → both methods work (different mechanisms)
  - Amplitude anomaly → IForest wins (all units affected equally, so inter-unit distance doesn't change, but IForest sees unusual absolute feature values)
- **Educational point:** "Notice AUC-ROC is higher across the board — it's more forgiving with imbalanced data. Event-AUC-PR is the honest metric."
- **Conclusion:** "No single method dominates. The right choice depends on your anomaly type. Use metrics like Event-AUC-PR to make data-driven decisions."

---

## Key Files

| Action | File |
|--------|------|
| **Modify** | `anomaly_detection/hvac/hvac_data_gen.py` — add `inject_anomaly_amplitude()` method |
| **Create** | `anomaly_detection/hvac/01_model_selection_workflow.ipynb` — the blog notebook |
| **Create** | `hvac/utils/visual.py` - visualization utilities |


## Existing Code to Reuse

- `HVACDataGenerator` class and its `generate_container_data()` method (`hvac_data_gen.py`)
- Rolling Euclidean distance pattern from `hvac_data_eda.ipynb` 
- IForest via `run_Unsupervise_AD('IForest', data)` from `TSB_AD/model_wrapper.py:56`
- Evaluation via `get_metrics(scores, labels)` from `TSB_AD/evaluation/metrics.py`
- Plotly visualization patterns from `hvac_data_eda.ipynb`

---

## Verification

1. Run the notebook top-to-bottom in the `TSB-AD` conda environment
2. 1. Verify: the amplitude anomaly is visually convincing (reduced oscillation range across all units)
3. Verify: lag container shows Euclidean distance >> IForest in Event-AUC-PR, Event-AUC-ROC
4. Verify: amplitude container shows IForest >> Euclidean distance in Event-AUC-PR, Event-AUC-ROC
5. Verify: all visualizations render (plotly line charts, bar charts, results table)

