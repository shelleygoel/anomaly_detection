# Statistical Modeling Blog - Anomaly Detection Project

## CRITICAL RESTRICTION

**DO NOT MODIFY ANY CODE IN THE `TSB-AD/` DIRECTORY.**

The TSB-AD library is a third-party dependency that should be used as-is. You may:
- Read files in TSB-AD for reference
- Import and use functions from TSB-AD
- Document how to use TSB-AD

You may ONLY modify files within the `anomaly_detection/` directory:
- Notebooks (`.ipynb` files)
- Python scripts
- Documentation files (`.md` files)
- Any other files under `anomaly_detection/`

## Project Overview

This is a blog/tutorial project focused on time series anomaly detection using HVAC sensor data from battery energy storage systems (BESS). The goal is to create educational content that demonstrates systematic approaches to choosing the right anomaly detection method for different problem types.

## Project Structure

```
/Users/shelleygoel/Code/01_statistical_mod_blog/
├── TSB-AD/                              # TSB-AD library (Time Series Benchmark for Anomaly Detection)
│   ├── explorations/
│   │   └── hvac_data_gen.py            # Synthetic HVAC data generator
│   ├── TSB_AD/
│   │   ├── model_wrapper.py            # run_Unsupervise_AD() - IForest, etc.
│   │   └── evaluation/
│   │       └── metrics.py              # get_metrics() - point-level metrics (not used as primary eval)
│   └── ...
├── anomaly_detection/
│   ├── hvac/
│   │   ├── hvac_data_eda.ipynb         # EDA notebook with Euclidean distance & IForest examples
│   │   └── 01_model_selection_workflow.ipynb  # Main blog post (to be created)
│   └── hvac/plans/
│       ├── IMPLEMENTATION_PLAN.md
│       └── evaluation_plan.md
└── data/
```

## Key Components

### 1. HVAC Data Generator (`TSB-AD/explorations/hvac_data_gen.py`)

**Purpose:** Generate realistic synthetic HVAC temperature data for testing anomaly detection methods.

**Current Features:**
- Simulates 3 HVAC units per container (BESS container)
- Realistic charge/discharge cycles with sharp rise and gradual decline
- Two anomaly types:
  - **Lag anomaly** (`inject_anomaly_lag`): One unit responds with a time delay
  - **Frequency anomaly** (`inject_anomaly_frequency`): One unit cycles at wrong rate
- Configurable anomalies via `anomaly_config` parameter

**Planned Addition:**
- **Amplitude anomaly** (`inject_anomaly_amplitude`): ALL units have reduced oscillation range
  - This is a "global" anomaly affecting all units equally
  - Makes inter-unit distance methods fail (they rely on divergence between units)
  - Should be detected by methods looking at absolute feature values (e.g., Isolation Forest)

### 2. TSB-AD Library Integration

**Location:** `/Users/shelleygoel/Code/01_statistical_mod_blog/TSB-AD/`

**Key Functions:**
- `TSB_AD.model_wrapper.run_Unsupervise_AD('IForest', data)` - Run anomaly detection models
- `TSB_AD.evaluation.metrics.get_metrics(scores, labels)` - Comprehensive evaluation metrics

**Metrics Used:**
- **Event-AUC-PR** (Primary): `sklearn.metrics.average_precision_score` on day-aggregated scores/labels. Strict, penalizes false alarms, honest for imbalanced data.
- **Event-AUC-ROC** (Secondary): `sklearn.metrics.roc_auc_score` on day-aggregated scores/labels. More forgiving — can overstate quality with imbalanced classes. Included for educational comparison.
- Day-level aggregation: `max()` score per day, `max()` label per day.
- TSB-AD's `get_metrics()` also available for point-level metrics (Standard-F1, PA-F1, Event-based-F1, R-based-F1, Affiliation-F, VUS-PR, AUC-ROC) but not used as primary evaluation.

### 3. Existing EDA Notebook (`anomaly_detection/hvac/hvac_data_eda.ipynb`)

Contains reusable patterns:
- **Rolling Euclidean Distance** (cell `612d8c7f`):
  ```python
  df_pivot[f'dist_{i}_{j}'] = np.sqrt(
      ((df_pivot[i] - df_pivot[j])**2).rolling(window_size).sum()
  )
  ```
- **Isolation Forest via TSB-AD**: Using both distance features and raw temperatures
- **Plotly visualizations**: Multi-panel time series plots

## Current Work: Blog Post #1 - Model Selection Workflow

**File:** `anomaly_detection/hvac/01_model_selection_workflow.ipynb` (to be created)

**Objective:** Demonstrate that different anomaly types require different detection methods through a systematic comparison.

**Methods Compared:**
1. **Euclidean Distance** - Pairwise rolling distance between correlated sensors
2. **Isolation Forest** - Multi-dimensional feature-space isolation

**Test Cases:**
| Container | Anomaly Type | Expected Winner | Why |
|-----------|--------------|-----------------|-----|
| 0 | Lag (unit 1, days 2-4) | Euclidean Distance | Direct signal of inter-unit divergence |
| 1 | Frequency (unit 2, days 2-4) | Both methods | Different mechanisms, both effective |
| 2 | Amplitude (ALL units, days 3-5) | Isolation Forest | Inter-unit distance unchanged; IForest sees unusual absolute values |

**Evaluation:** Event-AUC-PR (primary), Event-AUC-ROC (secondary) — day-level aggregated

See `hvac/plans/IMPLEMENTATION_PLAN.md` for detailed implementation steps.

## Development Environment

**Conda Environment:** `TSB-AD`
- Run notebooks using this environment
- Contains all TSB-AD dependencies

**Python Version:** Compatible with TSB-AD requirements

## Code Style & Conventions

### Data Generation
- Use `HVACDataGenerator` class from `hvac_data_gen.py`
- Always set `seed` for reproducibility
- Anomaly configs follow format:
  ```python
  [{
      'unit': 0-2,
      'type': 'lag' | 'frequency' | 'amplitude',
      'start_day': int,
      'start_hour': int,
      'duration_hours': int,
      'params': {...}  # Type-specific parameters
  }]
  ```

### Visualization
- Use Plotly for interactive plots (primary)
- Matplotlib for static plots when needed
- Multi-panel layouts with `plotly.subplots.make_subplots`
- Color scheme for HVAC units: `['#5470C6', '#EE6666', '#5DBCD2']` (blue, red, cyan)

### Anomaly Detection
- Use TSB-AD wrappers when available
- Always compute multiple metrics for comparison
- Store results in pandas DataFrames for easy tabulation

## Key Implementation Details

### Rolling Euclidean Distance Pattern
```python
# 1. Smooth data
df['TmpRet_smooth'] = df.groupby(['container_id', 'HVACNum'])['TmpRet'].transform(
    lambda x: x.rolling(window=10).mean()
)

# 2. Pivot to wide format
df_pivot = df.pivot(index='timestamp_et', columns='HVACNum', values='TmpRet_smooth')

# 3. Compute pairwise distances
window_size = 60
for i in [0, 1, 2]:
    for j in range(i+1, 3):
        df_pivot[f'dist_{i}_{j}'] = np.sqrt(
            ((df_pivot[i] - df_pivot[j])**2).rolling(window_size).sum()
        )

# 4. Aggregate to single anomaly score (max or mean of all distances)
```

### Isolation Forest Pattern
```python
from TSB_AD.model_wrapper import run_Unsupervise_AD

# Features: raw temperatures of all 3 units (N x 3 matrix)
features = df_pivot[[0, 1, 2]].values
scores = run_Unsupervise_AD('IForest', features, contamination=0.2)
```

### Evaluation Pattern
```python
from sklearn.metrics import average_precision_score, roc_auc_score

# Aggregate scores and labels to day level
day_scores = df.groupby('day')['anomaly_score'].max()
day_labels = df.groupby('day')['label'].max()

# Compute metrics
event_auc_pr = average_precision_score(day_labels, day_scores)
event_auc_roc = roc_auc_score(day_labels, day_scores)
```

## Working with the Codebase

### Adding New Anomaly Types
1. Add `inject_anomaly_<type>()` method to `HVACDataGenerator`
2. Update `generate_container_data()` to handle new type in anomaly_config
3. Update `_generate_random_anomaly_config()` to include new type
4. Test with explicit config before using random generation

### Creating New Notebooks
- Place in `anomaly_detection/hvac/` directory
- Use numbered prefixes: `01_`, `02_`, etc.
- Start with markdown cell explaining goal and context
- Import from TSB-AD using: `sys.path.append('/Users/shelleygoel/Code/01_statistical_mod_blog/TSB-AD')`
- Or use relative imports if possible

## File Paths Reference

**HVAC Data Generator:**
`/Users/shelleygoel/Code/01_statistical_mod_blog/TSB-AD/explorations/hvac_data_gen.py`

**TSB-AD Model Wrapper:**
`/Users/shelleygoel/Code/01_statistical_mod_blog/TSB-AD/TSB_AD/model_wrapper.py`

**TSB-AD Metrics:**
`/Users/shelleygoel/Code/01_statistical_mod_blog/TSB-AD/TSB_AD/evaluation/metrics.py`

**EDA Notebook:**
`/Users/shelleygoel/Code/01_statistical_mod_blog/anomaly_detection/hvac/hvac_data_eda.ipynb`

**Implementation Plan:**
`/Users/shelleygoel/Code/01_statistical_mod_blog/anomaly_detection/hvac/plans/IMPLEMENTATION_PLAN.md`

**Evaluation Plan:**
`/Users/shelleygoel/Code/01_statistical_mod_blog/anomaly_detection/hvac/plans/evaluation_plan.md`

## Common Tasks

### Generate Test Data
```python
import sys
sys.path.append('/Users/shelleygoel/Code/01_statistical_mod_blog/TSB-AD/explorations')
import hvac_data_gen as hvdg
from datetime import datetime

generator = hvdg.HVACDataGenerator(seed=42)

# Single container with specific anomaly
anomaly_config = [{
    'unit': 1,
    'type': 'lag',
    'start_day': 2,
    'start_hour': 8,
    'duration_hours': 48,
    'params': {'lag_minutes': 180}
}]

df = generator.generate_container_data(
    container_id=0,
    start_time=datetime(2026, 1, 15),
    duration_days=5,
    anomaly_config=anomaly_config
)
```

### Run Anomaly Detection
```python
from TSB_AD.model_wrapper import run_Unsupervise_AD
from sklearn.metrics import average_precision_score, roc_auc_score

# Prepare data (ensure no NaN values)
features = df_pivot[[0, 1, 2]].dropna().values

# Run detection
scores = run_Unsupervise_AD('IForest', features)

# Aggregate to day level and evaluate
day_scores = df.groupby('day')['anomaly_score'].max()
day_labels = df.groupby('day')['label'].max()

print(f"Event-AUC-PR: {average_precision_score(day_labels, day_scores):.3f}")
print(f"Event-AUC-ROC: {roc_auc_score(day_labels, day_scores):.3f}")
```

## Notes

- **Amplitude anomaly is the key novelty** - it demonstrates when inter-unit methods fail
- The blog post aims to be educational and generalizable beyond HVAC data
- Emphasis on systematic evaluation using proper metrics (Event-AUC-PR for day-level evaluation)
- All anomalies are synthetic with known ground truth for clean evaluation

## Plan Mode

- Make the plan extremely concise. Sacrifice grammar for the sake of concision.
- At the end of each plan, give me a list of unresolved questions to answer, if any.
EOF