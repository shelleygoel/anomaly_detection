# catch22 Feature Selection Pipeline — Pseudocode

Reference: Lubba et al. (2019), "catch22: CAnonical Time-series CHaracteristics"

## Inputs

- `D`: set of M = 93 classification tasks from UCR/UEA repository
- `F`: pool of 4791 hctsa features (after prefiltering)

---

## Step 1: Score each feature individually (Sec 2.4)

```
for each feature i in F:                          # 4791 features
    for each task j in D:                          # 93 tasks
        fit univariate decision tree using only feature i
            with stratified N_CV-fold cross-validation
        a[i,j] = mean class-balanced accuracy across folds    # Eq. 3
```

## Step 2: Normalize and combine scores across tasks (Sec 2.4)

```
for each task j:
    a_bar[j] = mean(a[i,j] for all i in F)        # mean accuracy across all features on task j

for each feature i:
    for each task j:
        a_n[i,j] = a[i,j] / a_bar[j]             # Eq. 4: normalized accuracy (relative to other features)

    a_nc[i] = mean(a_n[i,j] for all j in D)       # Eq. 5: combined normalized accuracy across all tasks
```

## Step 3: Statistical prefiltering — remove chance-level features (Sec 2.5)

```
for each feature i:
    for each task j:
        repeat 1000 times:
            shuffle class labels
            fit same univariate decision tree
            record shuffled accuracy
        fit Gaussian to shuffled accuracy distribution
        p[i,j] = P(shuffled accuracy >= a[i,j])   # one-sided p-value

    combine p-values across tasks using Fisher's method -> p_combined[i]
    apply Holm-Bonferroni correction across all features

remove features where corrected p_combined >= 0.05
    -> 145 features removed (3%), 4646 remain
```

## Step 4: Performance filtering — select top performers (Sec 2.6, Fig 3B)

```
threshold = mean(a_nc) + std(a_nc)                # one std dev above mean
top_features = {i : a_nc[i] >= threshold}
    -> 710 features survive
```

Note: the choice of threshold is not sensitive — Fig 3A shows that the final
accuracy after clustering is robust to starting from different numbers of top
features (beta = 100, 200, ..., 1000).

## Step 5: Redundancy minimization — cluster and pick representatives (Sec 2.6, Fig 3C)

```
# Build performance-correlation distance matrix between top features
for each pair (i, k) in top_features:
    r[i,k] = pearson_correlation(a_n[i, :], a_n[k, :])   # correlation of M-dimensional performance vectors
    d[i,k] = 1 - r[i,k]                                   # distance

# Cluster
apply hierarchical clustering (complete linkage) with distance threshold gamma = 0.2
    -> all features within a cluster have pairwise r > 0.8
    -> yields 22 clusters

# Pick one representative per cluster
for each cluster c:
    select feature with highest a_nc[i]
    (or manually pick a simpler/more interpretable feature if accuracy is similar)
    -> 22 canonical features = catch22
```

## Step 6 (Fig 3A): Verify that ~22 features is enough (sensitivity analysis)

This is not part of the pipeline itself — it justifies the choice of ~22 features.

```
for N = 1 to 50:                                   # number of final features
    for beta in [100, 200, 300, ..., 1000]:        # number of top features to start from
        take top-beta features by a_nc
        cut dendrogram at level giving N clusters
        pick best feature per cluster -> N features
        a_tot_subset = mean of a[i,j] across selected features, tasks, CV folds   # Eq. 6
    
    accuracy_drop[N] = (a_tot_full - mean(a_tot_subset across betas)) / a_tot_full
    error_bars[N] = std(a_tot_subset across betas)

# Result: accuracy drop saturates under 10% at N ~ 20-30
# Tight error bars confirm insensitivity to beta
```

## Evaluation (Sec 2.7)

```
# Within-pipeline comparisons (catch22 vs full set):
a_tot = (1/M) * sum over tasks j of (1/N_CV,j) * sum over folds k of a[j,k]    # Eq. 6, class-balanced

# Comparison to external methods (DTW, COTE, etc.):
a_ub_tot = (1/M) * sum over tasks j of a_ub[j]     # Eq. 7, unbalanced (to match UCR repository metrics)
```

---

## Summary of key parameters

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| N_CV | 2-10 per task | Cross-validation folds (Eq. 2) |
| p-value threshold | 0.05 | Statistical prefiltering cutoff |
| a_th | mean + 1 std | Performance filtering threshold |
| gamma | 0.2 | Clustering distance threshold |
| beta | 100-1000 | Top features input to clustering (robust to choice) |
