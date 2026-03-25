# Post 02: Threshold Selection via Conformal Prediction

## Context

Post 01 answers "which anomaly detection method?" (Euclidean distance vs IForest vs ...). But the thresholding step — deciding when a score is "anomalous enough" — is usually ad-hoc (MAD z-score, fixed percentile, contamination parameter). Conformal prediction provides calibrated thresholds with finite-sample guarantees, no distribution assumptions needed.

## Problem Statement

Given an anomaly score function (any method), how do you set a threshold that guarantees a maximum false alarm rate on new data?

## Approach: Conformal Anomaly Detection

### Core Idea
- Hold out a **calibration set** of known-clean data
- For each new test point, compute a **conformal p-value**: fraction of calibration scores ≥ the new score
- Flag as anomalous if p-value < α (desired significance level)
- **Guarantee:** false alarm rate ≤ α if calibration data is exchangeable with future data

### Implementation Steps

1. **Generate data** — reuse same HVAC containers from Post 01 (lag, frequency, amplitude anomalies)
2. **Compute base anomaly scores** — Euclidean distance and IForest (same as Post 01)
3. **Split known-clean period** into calibration set (e.g., day 0–1 which have no injected anomalies)
4. **Compute conformal p-values** for each test-period point:
   ```python
   def conformal_pvalue(new_score, cal_scores):
       return (np.sum(cal_scores >= new_score) + 1) / (len(cal_scores) + 1)
   ```
5. **Compare thresholding methods** on same data:
   - MAD z-score threshold (current approach)
   - Fixed percentile (e.g., 95th of training scores)
   - Conformal p-value at α = 0.05
6. **Evaluate** — Event-AUC-PR and actual false alarm rate on clean test days
7. **Visualize** — time series with flagged anomalies overlaid, comparing threshold methods

### Key Comparisons to Show

| Method | Assumption | Guarantee | Tuning |
|--------|-----------|-----------|--------|
| MAD z-score | Score distribution is symmetric/Gaussian-like | None | z threshold |
| Percentile | Enough training data to estimate tail | Asymptotic | percentile |
| Conformal | Exchangeability (i.i.d. or weaker) | Finite-sample FAR ≤ α | α |

### Blog Narrative

1. "You picked the right method — now what threshold do you use?"
2. Show that MAD z-score / percentile thresholds are fragile (change with score distribution shape)
3. Introduce conformal p-values as principled alternative
4. Demo: same base scores, different thresholding → different false alarm rates
5. Conformal gives you interpretable p-values ("< 5% chance this is normal behavior")
6. Takeaway: the scoring function and the thresholding step are separate design choices

### Practical Considerations to Discuss

- **Calibration set size**: more calibration data → finer p-values (minimum ~20 points for α=0.05)
- **Non-stationarity**: if normal behavior drifts, recalibrate periodically
- **Multiple testing**: if flagging many points, consider Bonferroni or BH correction
- **Online vs batch**: can update calibration set as new clean data arrives

## Dependencies

- Post 01 complete (reuse data generation + scoring code)
- Same conda env (TSB-AD)

## Unresolved Questions

- How much calibration data is needed for HVAC data at 1-min resolution? (day 0-1 = ~2880 points — likely plenty)
- Should we show Bonferroni correction or keep it simple for a blog post?
- Include comparison to TSB-AD's built-in threshold methods?
