# BESS Failure Detection: Simplified Plan

## Goal
Build an evaluation harness with synthetic data so that when real data and models arrive, you can immediately measure what works.

---

## Phase 1: Synthetic Data + Data Foundation (Week 1)

### Directory Structure
```
project/
├── data/
│   ├── synthetic/
│   │   ├── raw/           # time-series CSVs
│   │   └── metadata.csv   # system_id, operational_mode, season, system_id
│   ├── real/              # same structure, empty for now
│   └── labels.csv         # system_id, label, failure_type, lead_time_days, source
├── src/
│   ├── data_loader.py     # load and merge time-series + labels
│   ├── generate.py        # synthetic data generation
│   ├── evaluation.py      # metrics and scoring
│   └── baselines.py       # simple baseline models
└── notebooks/
    └── analysis.ipynb
```

### Synthetic Data Generation (`src/generate.py`)
Generate cases with known ground truth covering:
- Normal operation across modes/seasons
- Edge cases: noisy normals, subtle anomalies, sensor dropouts

Output: time-series CSVs + labels.csv + metadata.csv — your "test suite."

### Data Schema
- **labels.csv**: system_id, label (normal/anomaly), failure_type, source (synthetic/real), label_version
- **metadata.csv**: system_id, source(synthetic/real), operational_mode, season
- **Time-series**: CSV per system with timestamp + sensor columns

### Data Loader (`src/data_loader.py`)
- `load_timeseries(system_id)` → DataFrame
- `load_labels(source=None)` → DataFrame
- `load_dataset(source=None)` → merged time-series + labels
- Basic validation: missing files, schema checks
- `visualize_labels(labels_df)` → bar charts of label distribution by label, failure_type, operational_mode

---

## Phase 2: Evaluation Harness (Week 1-2)

### Evaluation Module (`src/evaluation.py`)
- Input: list of (system_id, score) pairs + ground truth labels
- Metrics: See metrics from [metrics_plan](metrics_plan.md)
- Breakdown by: failure_type, operational_mode, lead_time bucket
- Visualize metrics

### Smoke Test Baseline (`src/baselines.py`)
- `random_scorer(system_ids)` → random scores
- `threshold_scorer(system_ids, sensor, threshold)` → z-score or simple threshold
- PCA reconstruction error (optional, only if quick)

### Validation Checklist
Run baselines through evaluation and confirm:
- [ ] Random scores → ~0.5 AUC
- [ ] Simple threshold → better than random
- [ ] Metrics break down correctly by strata
- [ ] Pipeline handles edge cases (all normal, all anomaly, missing data)

---

## Phase 3: Real Data Integration (When Available)

### Labeling Workflow
- Streamlit app (`src/labeling_app.py`): select case, view time-series plot, enter label + failure_type + confidence + lead_time + notes, save to labels.csv
- Run with `streamlit run src/labeling_app.py`

### Stratified Sampling
- Only build this when you have enough real cases that you need to prioritize which to label
- Until then, label everything or sample randomly

### Combine and Evaluate
- `load_dataset(source='all')` merges synthetic + real
- Run same evaluation — compare model performance on synthetic vs real
- This is the payoff: harness is already built, just plug in new data

---

## Phase 4: Experiment Tracking (Before Deployment)

### Why Now
Before deploying a model, you need to answer: "which model, trained on which data, with which parameters, produced these results?" If you can't, you can't deploy with confidence or reproduce issues.

### Use MLflow
- `pip install mlflow`, run `mlflow ui` for local dashboard
- Log per run: model params, data source, label version, metrics (AUC, F1, precision, recall)
- Use tags for metadata: `mlflow.set_tag("data_source", "synthetic+real")`
- Store model artifacts and evaluation plots as needed
- Built-in run comparison and metric visualization

```python
import mlflow

with mlflow.start_run(run_name="pca_baseline"):
    mlflow.set_tag("data_source", "synthetic")
    mlflow.set_tag("label_version", "v1")
    mlflow.log_params({"model": "pca", "n_components": 5})
    mlflow.log_metrics({"auc": 0.82, "f1": 0.75})
```

---

## Deferred (Build When Needed)

| Component | Trigger to build |
|---|---|
| Stratified sampler | You have 500+ unlabeled real cases |
| Full labeling CLI | Labeling more than ~50 cases manually or Multiple labelers or labeling campaigns |

---

## Key Principle

Everything serves the feedback loop:
```
synthetic data + labels → evaluation harness → smoke-test baseline → trust the harness
    → build real models → get real data → slot it in → iterate with confidence
```