from typing import List, Optional

import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
import re
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, balanced_accuracy_score
from joblib import Parallel, delayed

CATCH22 = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "SB_BinaryStats_mean_longstretch1",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    ## missing features from the dataset provided from paper's authors
    # "CO_f1ecac",
    # "CO_FirstMin_ac",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
    "CO_trev_1_num",
    "CO_HistogramAMI_even_2_5",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "FC_LocalSimple_mean1_tauresrat",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0.01",
]

## Transformations etc.
def normalize_features(tasks_df: pd.DataFrame):
    feature_cols = [c for c in tasks_df.columns if c not in ('label', 'dataset_name')]

    def _normalize(group):
        feat = group[feature_cols].copy()
        min_vals = feat.min()
        range_vals = feat.max() - min_vals
        result = group.copy()
        result[feature_cols] = (feat - min_vals) / range_vals
        # result['dataset_name'] = group.name
        return result

    return tasks_df.groupby('dataset_name').apply(_normalize).copy()

def filter_invalid_features(tasks_df: pd.DataFrame):
    feature_cols = [c for c in tasks_df.columns if c not in ('label', 'dataset_name')]
    def _has_invalid(col):
        return col.isna().any() | np.isinf(col).any()

    invalid_dataset_count = (
        tasks_df.groupby('dataset_name')[feature_cols]
        .apply(lambda grp: grp.apply(_has_invalid))
        .sum()
    )

    num_datasets = tasks_df["dataset_name"].nunique()
    survivors = invalid_dataset_count[invalid_dataset_count < 0.8 * num_datasets]

    # Quality: % of datasets where the feature is fully valid
    quality = 1 - survivors / num_datasets
    print(f"Median: {quality.median():.1%}")
    print(f"Worst:  {quality.min():.1%}")
    print(f"≥99%:   {(quality >= 0.99).mean():.1%} of survivors")


    in_survivors = [f for f in CATCH22 if f in survivors.index]
    missing = [f for f in CATCH22 if f not in survivors.index]
    not_in_features = [f for f in CATCH22 if f not in feature_cols]

    print(f"Catch22 In survivors: {len(in_survivors)}/22")
    print(f"Missing from survivors: {missing}")
    print(f"Not in feature set at all: {not_in_features}")

    # all tasks with raw features removed
    # normalized features and features with invalid values prefiltered 
    return tasks_df[list(survivors.index) + ["label", "dataset_name"]]

# Decision Tree classification based feature scoring
def calc_per_feat_score(
    task_df: pd.DataFrame, nfolds: pd.DataFrame, feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    if not feature_cols:
        feature_cols = [col for col in task_df.columns if col not in ["label", "dataset_name"]]

    actual_scores = {}

    def _score_feature(data, y, feature, cv, scorer):
        X = data[[feature]]
        # print(X.shape)
        # TODO: replace this with custom tree
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)
        return cross_val_score(clf, X, y, cv=cv, scoring=scorer).mean()

    datasets = list(task_df.groupby(["dataset_name"]))
    total = len(datasets)
    for i, ((dname,), data) in enumerate(datasets, 1):
        print(f"[{i}/{total}] {dname=}")
        n_cv = nfolds.loc[nfolds["dataset_name"] == dname, "nfolds"].values[0]
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=23)
        y = data["label"]
        actual_scores[dname] = Parallel(n_jobs=-1)(
            delayed(_score_feature)(data, y, feature, cv, make_scorer(balanced_accuracy_score))
            for feature in feature_cols
        )

    return pd.DataFrame.from_dict(actual_scores, orient="index", columns=feature_cols).rename_axis(
        "dataset_name"
    )

def normalize_and_combine_perf_scores(perf_scores: pd.DataFrame) -> tuple:
    perf_scores_n = perf_scores.div(perf_scores.mean(axis=1), axis=0)
    perf_scores_n_c = perf_scores_n.mean()
    return perf_scores_n, perf_scores_n_c

def num_of_folds(tasks: pd.DataFrame):
    num_ts = tasks.groupby(["dataset_name", "label"]).size().reset_index(name="sample_size")
    num_folds = num_ts.groupby(["dataset_name"])["sample_size"].min().reset_index(name="nfolds")
    num_folds["nfolds"] = num_folds["nfolds"].clip(2, 10)
    return num_folds


def sample_imbalanced_datasets(df, n=5, imbalance_ratio=0.3, seed=42):
    """Sample n datasets where minority class is <= imbalance_ratio of total samples."""
    counts = df.groupby(["dataset_name", "label"]).size().reset_index(name="count")
    totals = counts.groupby("dataset_name")["count"].transform("sum")
    counts["frac"] = counts["count"] / totals
    minority_frac = counts.groupby("dataset_name")["frac"].min()
    imbalanced = minority_frac[minority_frac <= imbalance_ratio]
    print(f"{len(imbalanced)} datasets with minority class <= {imbalance_ratio:.0%} of samples")
    sampled = imbalanced.sample(n=min(n, len(imbalanced)), random_state=seed)
    return sampled 

def accuracy_class_balanced(y_true: np.array, y_predict: np.array):
    y_true = np.asarray(y_true)
    y_predict = np.asarray(y_predict)
    classes, counts = np.unique(y_true, return_counts=True)
    # map each sample to its class weight via index lookup (classes are sorted)
    w = 1.0 / counts[np.searchsorted(classes, y_true)]
    accuracy = np.sum((y_true == y_predict) * w) / np.sum(w)
    return accuracy


### --------- Statitical Filtering of randomly performant features ---------------------------#####
def calc_score_null_dist(
    task_df: pd.DataFrame,
    nfolds: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_permutations: int = 1000,
    seed: int = 42,
) -> List[pd.DataFrame]:
    rng = np.random.RandomState(seed)
    null_scores = []

    for i in range(n_permutations):
        print(f"[{i+1}/{n_permutations}]")
        shuffled_df = task_df.copy()
        shuffled_df["label"] = (
            shuffled_df.groupby("dataset_name")["label"]
            .transform(rng.permutation)
        )
        scores_i = calc_per_feat_score(shuffled_df, nfolds, feature_cols=features)
        null_scores.append(scores_i)

    return null_scores


def calc_p_values(
    actual_scores: pd.DataFrame,
    null_scores: List[pd.DataFrame],
) -> pd.DataFrame:
    from scipy.stats import gaussian_kde

    n_permutations = len(null_scores)
    # Stack null distributions: for each (dataset, feature) collect scores across permutations
    null_stack = np.stack([ns.values for ns in null_scores], axis=0)  # (n_perms, n_datasets, n_features)

    p_values = pd.DataFrame(
        index=actual_scores.index, columns=actual_scores.columns, dtype=float
    )
    for i, dname in enumerate(actual_scores.index):
        for j, feat in enumerate(actual_scores.columns):
            null_vals = null_stack[:, i, j]
            actual = actual_scores.loc[dname, feat]
            if np.all(np.isnan(null_vals)) or np.isnan(actual):
                p_values.loc[dname, feat] = np.nan
                continue
            kde = gaussian_kde(null_vals[~np.isnan(null_vals)])
            p = kde.integrate_box_1d(actual, np.inf)
            p_values.loc[dname, feat] = max(p, 1 / (n_permutations + 1))

    return p_values

### ----------------------------------- Plotting Functionalities ------------------------------------- ###
def plot_null_distribution(
    feature: str,
    actual_scores: pd.DataFrame,
    null_scores: List[pd.DataFrame],
    p_values: pd.DataFrame,
):
    from scipy.stats import gaussian_kde

    null_stack = np.stack([ns[feature].values for ns in null_scores], axis=0)  # (n_perms, n_datasets)
    datasets = actual_scores.index.tolist()
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 1, figsize=(5, 2 * n_datasets), squeeze=False)

    for i, dname in enumerate(datasets):
        ax = axes[i, 0]
        null_vals = null_stack[:, i]
        null_vals = null_vals[~np.isnan(null_vals)]
        actual = actual_scores.loc[dname, feature]
        p_value = p_values.loc[dname, feature]

        kde = gaussian_kde(null_vals)
        x_grid = np.linspace(
            null_vals.min() - 0.01, max(null_vals.max(), actual) + 0.01, 500
        )
        ax.fill_between(x_grid, kde(x_grid), alpha=0.3, label="Null KDE")
        ax.fill_between(
            x_grid[x_grid >= actual],
            kde(x_grid[x_grid >= actual]),
            alpha=0.5, color="red",
            label=f"p-value = {p_value:.4f}",
        )
        ax.axvline(actual, color="red", linestyle="--", label=f"Actual: {actual:.3f}")
        ax.set_xlabel("Accuracy")
        ax.set_title(f"{feature} — {dname}")
        ax.legend()

    fig.tight_layout()
    return fig


def plot_score_densities_per_dataset(perf_scores: pd.DataFrame, colorscale: str = "Turbo", width: float = 2):
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde

    fig = go.Figure()
    datasets = perf_scores.index.tolist()
    colors = px.colors.sample_colorscale(colorscale, np.linspace(0, 1, len(datasets)))

    for ds, color in zip(datasets, colors):
        vals = perf_scores.loc[ds].dropna().values
        kde = gaussian_kde(vals)
        x_range = np.linspace(vals.min(), vals.max(), 500)
        y = kde(x_range)
        fig.add_trace(go.Scatter(
            x=x_range, y=y,
            mode="lines",
            line=dict(color=color, width=width),
            name=ds,
            showlegend=True,
        ))
    return fig

def plot_combined_accuracy(perf_scores_n_c: pd.Series, highlight_features: list = None):
    import plotly.graph_objects as go

    if highlight_features is None:
        highlight_features = CATCH22

    sorted_scores = perf_scores_n_c.sort_values().reset_index()
    sorted_scores.columns = ['feature', 'score']
    sorted_scores['is_highlight'] = sorted_scores['feature'].isin(highlight_features)

    other = sorted_scores[~sorted_scores['is_highlight']]
    highlighted = sorted_scores[sorted_scores['is_highlight']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=other.index, y=other['score'], mode='markers',
        marker=dict(size=4, color='lightgrey'),
        text=other['feature'], hoverinfo='text+y', name='Other'
    ))
    fig.add_trace(go.Scatter(
        x=highlighted.index, y=highlighted['score'], mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        text=highlighted['feature'], hoverinfo='text+y', name='Highlighted'
    ))

    fig.update_layout(
        width=1200, height=500,
        xaxis=dict(
            title='Feature Rank',
            tickvals=highlighted.index.tolist(),
            ticktext=highlighted['feature'].tolist(),
            tickangle=90,
            tickfont=dict(size=10),
        ),
        yaxis_title='Combined Accuracy',
        hovermode='closest',
    )
    return fig


if __name__ == "__main__":
    from sklearn.metrics import balanced_accuracy_score

    assert accuracy_class_balanced([1, 1, 2], [2, 2, 2]) == balanced_accuracy_score([1, 1, 2], [2, 2, 2])
    accuracy_class_balanced([1, 1, 2], [2, 2, 2])
