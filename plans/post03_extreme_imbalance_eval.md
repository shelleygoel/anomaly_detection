# Post 03: Evaluation & Deployment Under Extreme Class Imbalance

## Context

Real-world anomaly detection often has anomaly rates of 0.01% or less. At this prevalence:
- Standard metrics break down or mislead
- Not enough positive labels to estimate recall
- Test sets may contain zero anomalies
- Production monitoring becomes the main challenge

## Part 1: Why Standard Metrics Fail at 0.01%

### The Numbers
- 1M data points → ~100 anomalies
- A model that flags nothing gets 99.99% accuracy
- A test split might have 0-5 anomalies — too few for reliable metric estimates

### Metric-by-Metric Breakdown

| Metric | Problem at 0.01% |
|--------|-----------------|
| Accuracy | Useless — always near 100% |
| F1-score | Heavily influenced by threshold choice; unstable with few positives |
| AUC-ROC | Misleadingly high — FPR denominator (TN+FP) is huge, so even many FPs barely move FPR |
| AUC-PR | Honest but noisy — few positives means PR curve is jagged |
| Recall@K | Requires knowing K in advance |

### Demo Plan
- Generate HVAC data with 0.01% anomaly rate (e.g., 30 days, anomaly on 1 hour)
- Show all metrics for a mediocre model → AUC-ROC looks great, AUC-PR reveals the truth
- Visualize ROC vs PR curves side by side

## Part 2: Metrics That Work

### 2a. Precision-Recall AUC (Primary)
- Baseline = prevalence (0.01%), so any improvement is meaningful
- Report "lift over random": AUC-PR / prevalence
- Honest about false alarm cost

### 2b. Precision@K (Practical for alert review)
- "If I review top 10 alerts per day, how many are real?"
- Directly maps to analyst workload
- Compute: sort by score descending, precision in top K

### 2c. Cost-Weighted Metrics
- Assign business costs: cost_FN (missed anomaly) vs cost_FP (false alert)
- Total cost = cost_FN × FN + cost_FP × FP
- Optimal threshold minimizes total cost
- Demo: show how optimal threshold shifts with cost ratio
- Key insight: if missing an anomaly costs 1000x investigating a false alarm, the optimal threshold is very low (flag aggressively)

### 2d. Time-to-Detection
- For sequential anomalies: how quickly after onset is it flagged?
- More relevant than point-level metrics for monitoring systems
- Compute: time between anomaly start and first alert

### 2e. Event-Level Evaluation (Already Using)
- Aggregate to coarser time units (day-level, event-level)
- Reduces noise from point-level evaluation
- Already in Post 01 — extend with bootstrapped confidence intervals

### 2f. Bootstrap Confidence Intervals
- With ~100 anomalies, metric estimates have wide CIs
- Bootstrap the anomaly set to quantify uncertainty
- Report: "AUC-PR = 0.45 [0.31, 0.58] (95% CI)"
- Prevents overconfidence in metric values

## Part 3: Evaluation Strategy When Labels Are Scarce

### 3a. Synthetic Anomaly Injection
- Inject known anomalies into clean data at controlled rates
- Evaluate detection performance on injected anomalies
- Already doing this in HVAC project — generalize the pattern
- Caveat: synthetic anomalies may not cover real-world failure modes

### 3b. Cross-Validation on the Few Labels You Have
- Leave-one-anomaly-out CV: train on all data, evaluate each known anomaly separately
- Gives N estimates (where N = number of known anomalies)
- More robust than single train/test split

### 3c. Score Distribution Monitoring (Label-Free)
- Don't need labels to monitor model health
- Track score distribution on normal data over time
- If distribution shifts → model may be degrading or data has changed
- KS-test or KL-divergence between current vs calibration period scores

## Part 4: Deployment Patterns

### 4a. Tiered Alerting
```
High confidence:   p-value < 0.001  → auto-escalate
Medium confidence: p-value < 0.01   → analyst queue
Low confidence:    p-value < 0.05   → log for review
```
- Reduces alert fatigue
- Conformal p-values (Post 02) plug directly in

### 4b. Active Learning Loop
1. Model scores all data
2. Analyst reviews top-K alerts daily
3. Analyst labels: true anomaly or false alarm
4. Labels feed back to improve model or recalibrate threshold
5. Over time: precision@K improves, workload stays constant

### 4c. Cold Start Problem
- Day 1: no anomaly labels at all
- Strategy: unsupervised model (IForest) + conservative threshold
- Build calibration set from first clean period
- Gradually collect labels through active learning
- Switch to semi-supervised model once you have ~50+ labeled anomalies

### 4d. Monitoring in Production
- Track daily: # alerts, score distribution stats, precision@K (when labels available)
- Alert on: score distribution drift, sudden spike in alert volume, precision@K dropping
- Recalibrate conformal threshold periodically (weekly/monthly)

## Blog Narrative

1. "Your model looks great — 0.998 AUC-ROC!" → Show why this is misleading
2. Switch to AUC-PR → "Oh, it's actually 0.12"
3. "But even AUC-PR is noisy with so few positives" → Bootstrap CIs
4. "What metric should I report to stakeholders?" → Precision@K, cost-weighted
5. "How do I deploy with almost no labels?" → Cold start → active learning → monitoring
6. Takeaway: at extreme imbalance, evaluation strategy matters more than model choice

## Implementation Outline

### Notebook Cells
1. Generate HVAC data: 30 days, anomaly on ~1 hour (0.01% rate)
2. Run Euclidean distance + IForest
3. Compute all metrics, show ROC vs PR curves
4. Bootstrap CIs on AUC-PR
5. Cost-weighted analysis: vary cost ratio, show threshold shift
6. Precision@K analysis
7. Score distribution monitoring demo
8. Summary table comparing metrics

### Reusable Functions to Create
- `bootstrap_metric(scores, labels, metric_fn, n_boot=1000)` → point estimate + CI
- `precision_at_k(scores, labels, k)` → precision for top-K alerts
- `cost_weighted_threshold(scores, labels, cost_fn, cost_fp)` → optimal threshold
- `score_distribution_monitor(cal_scores, new_scores)` → KS stat + p-value

## Dependencies
- Post 01 (data generation, base scoring methods)
- Post 02 (conformal p-values for tiered alerting)

## Unresolved Questions
- How to simulate realistic 0.01% rate with HVAC data? (30 days with 1-hour anomaly ≈ 0.14%, not quite 0.01%. Could do 90 days with 1-hour anomaly ≈ 0.046%)
- Include semi-supervised methods (e.g., label propagation) or keep it to unsupervised?
- How deep to go on active learning? Full implementation or conceptual?
- Should this be one post or split into "evaluation" and "deployment" posts?
