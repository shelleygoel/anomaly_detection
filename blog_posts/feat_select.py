from typing import List, Optional

import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
import re
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer
from joblib import Parallel, delayed

CATCH22 = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "SB_BinaryStats_mean_longstretch1",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "CO_f1ecac",
    "CO_FirstMin_ac",
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


def num_of_folds(tasks: pd.DataFrame):
    num_ts = tasks.groupby(["dataset_name", "label"]).size().reset_index(name="sample_size")
    num_folds = num_ts.groupby(["dataset_name"])["sample_size"].min().reset_index(name="nfolds")
    num_folds["nfolds"] = num_folds["nfolds"].clip(2, 10)
    return num_folds


def accuracy_class_balanced(y_true: np.array, y_predict: np.array):
    y_true = np.asarray(y_true)
    y_predict = np.asarray(y_predict)
    classes, counts = np.unique(y_true, return_counts=True)
    # map each sample to its class weight via index lookup (classes are sorted)
    w = 1.0 / counts[np.searchsorted(classes, y_true)]
    accuracy = np.sum((y_true == y_predict) * w) / np.sum(w)
    return accuracy


def cal_feature_score(
    task_df: pd.DataFrame, nfolds: pd.DataFrame, feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    if not feature_cols:
        feature_cols = [col for col in task_df.columns if col not in ["label", "dataset_name"]]
    scorer = make_scorer(accuracy_class_balanced)

    actual_scores = {}

    def _score_feature(data, y, feature, cv, scorer):
        X = data[[feature]]
        # TODO: replace this with custom tree
        stump = tree.DecisionTreeClassifier()
        return cross_val_score(stump, X, y, cv=cv, scoring=scorer).mean()

    for (dname,), data in task_df.groupby(["dataset_name"]):
        n_cv = nfolds.loc[nfolds["dataset_name"] == dname, "nfolds"].values[0]
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        y = data["label"]
        actual_scores[dname] = Parallel(n_jobs=-1)(
            delayed(_score_feature)(data, y, feature, cv, scorer) for feature in feature_cols
        )

    return pd.DataFrame.from_dict(actual_scores, orient="index", columns=feature_cols).rename_axis(
        "dataset_name"
    )


def calc_score_null_dist(task_df: pd.DataFrame, feature: str, n_permutations=1000, nfolds=None):
    scorer = make_scorer(accuracy_class_balanced)

    null_dist = {}
    actual_scores = {}

    for (dname,), data in task_df.groupby(["dataset_name"]):
        null_scores = []
        n_cv = nfolds.loc[nfolds["dataset_name"] == dname, "nfolds"].values[0]
        X = data[[feature]]
        y = data["label"]
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        stump = tree.DecisionTreeClassifier(max_depth=1)
        scores = cross_val_score(stump, X, y, cv=cv, scoring=scorer)
        actual_scores[dname] = scores.mean()

        for i in range(n_permutations):
            shuffled_y = np.random.permutation(y)

            cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
            stump = tree.DecisionTreeClassifier(max_depth=1)
            scores = cross_val_score(stump, X, shuffled_y, cv=cv, scoring=scorer)
            null_scores.append(scores.mean())

            # print(f"Folds: {n_cv}, Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
            # print(f"Per-fold: {scores}")
        null_dist[dname] = null_scores
    return actual_scores, null_dist


def calc_combined_p_values(feature: str, actual_scores: dict, null_dist: dict, n_permutations=1000):
    from scipy.stats import gaussian_kde

    all_p_values = {}
    for dname, actual_score in actual_scores.items():
        kde = gaussian_kde(null_dist[dname])
        # p-value = P(score >= actual_score) under null distribution
        p_value = kde.integrate_box_1d(actual_score, np.inf)
        p_value = np.maximum(p_value, 1 / (n_permutations + 1))
        all_p_values[(feature, dname)] = p_value

        fig, ax = plt.subplots(figsize=(5, 2))
        x_grid = np.linspace(
            min(null_dist[dname]) - 0.01, max(max(null_dist[dname]), actual_score) + 0.01, 500
        )
        ax.fill_between(x_grid, kde(x_grid), alpha=0.3, label="Null KDE")
        ax.fill_between(
            x_grid[x_grid >= actual_score],
            kde(x_grid[x_grid >= actual_score]),
            alpha=0.5,
            color="red",
            label=f"p-value = {p_value:.4f}",
        )
        ax.axvline(actual_score, color="red", linestyle="--", label=f"Actual: {actual_score:.3f}")
        ax.set_xlabel("Accuracy")
        ax.set_title(f"Null Distribution with p-value for {feature} for {dname=}")
        ax.legend()
    return all_p_values


if __name__ == "__main__":
    from sklearn.metrics import balanced_accuracy_score

    assert accuracy_class_balanced([1, 1, 2], [2, 2, 2]) == balanced_accuracy_score([1, 1, 2], [2, 2, 2])
    accuracy_class_balanced([1, 1, 2], [2, 2, 2])
