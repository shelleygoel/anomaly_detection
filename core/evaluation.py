"""Evaluation class for anomaly detection models.

Joins day-level scores with day-level labels, computes AUC-PR/ROC metrics,
and plots PR/ROC curves broken down by anomaly type.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from core.dataset import TimeSeriesDataset


class Evaluation:
    """Evaluate anomaly detection scores against ground-truth labels."""

    def __init__(self, level: str = "day"):
        #TODO: Add range based event level evaluation
        #currently only day level evaluation supported
        if level != "day":
            raise ValueError("Only day level evaluation supported")
        self.level = level


    def auc_pr(
        self,
        scores: TimeSeriesDataset,
        dataset: TimeSeriesDataset,
        anomaly_type: str | None = None,
    ) -> float:
        labeled_scores = self._merge_scores_w_labels(scores, dataset)
        label_type_col = dataset.col_map["label_type"]

        if anomaly_type is not None:
            scores_subset, binary_labels = self._filter_by_type(labeled_scores, anomaly_type, label_type_col)
        else:
            # compute global PR -
            # all anomaly types are weighted equally
            scores_subset = labeled_scores
            label_col = dataset.col_map["label"]
            binary_labels = labeled_scores[label_col].astype(int)

        return average_precision_score(binary_labels, scores_subset["anomaly_score"])

    def auc_roc(
        self,
        scores: TimeSeriesDataset,
        dataset: TimeSeriesDataset,
        anomaly_type: str | None = None,
    ) -> float:
        labeled_scores = self._merge_scores_w_labels(scores, dataset)
        label_type_col = dataset.col_map["label_type"]

        if anomaly_type is not None:
            scores_subset, binary_labels = self._filter_by_type(labeled_scores, anomaly_type, label_type_col)
        else:
            scores_subset = labeled_scores
            label_col = dataset.col_map["label"]
            binary_labels = labeled_scores[label_col].astype(int)

        return roc_auc_score(binary_labels, scores_subset["anomaly_score"])

    def metrics_table(
        self,
        scores: TimeSeriesDataset,
        dataset: TimeSeriesDataset,
        anomaly_types: list[str] | None = None,
    ) -> pd.DataFrame:
        """Returns DataFrame with columns [anomaly_type, auc_pr, auc_roc]."""
        if anomaly_types is None:
            anomaly_types = self._get_anomaly_types(dataset)

        rows = []
        for atype in anomaly_types:
            rows.append({
                "anomaly_type": atype,
                "auc_pr": self.auc_pr(scores, dataset, anomaly_type=atype),
                "auc_roc": self.auc_roc(scores, dataset, anomaly_type=atype),
            })

        rows.append({
            "anomaly_type": "overall",
            "auc_pr": self.auc_pr(scores, dataset),
            "auc_roc": self.auc_roc(scores, dataset),
        })

        return pd.DataFrame(rows)

    
    def plot_pr_curve(
        self,
        scores: TimeSeriesDataset,
        dataset: TimeSeriesDataset,
        model_name: str,
        anomaly_types: list[str] | None = None,
    ) -> go.Figure:
        """PR curve subplots — one column per anomaly type."""
        if anomaly_types is None:
            anomaly_types = self._get_anomaly_types(dataset)

        labeled_scores = self._merge_scores_w_labels(scores, dataset)
        label_type_col = dataset.col_map["label_type"]

        fig = make_subplots(
            rows=1,
            cols=len(anomaly_types),
            subplot_titles=[t.capitalize() for t in anomaly_types],
            shared_yaxes=True,
        )

        self._add_pr_traces(fig, labeled_scores, label_type_col, anomaly_types, model_name)

        fig.update_xaxes(title_text="Recall", range=[0, 1])
        fig.update_yaxes(title_text="Precision", range=[0, 1], col=1)
        fig.update_layout(
            title_text=f"PR Curves — {model_name}",
            height=600,
            width=600 * len(anomaly_types),
        )
        return fig

    def plot_roc_curve(
        self,
        scores: TimeSeriesDataset,
        dataset: TimeSeriesDataset,
        model_name: str,
        anomaly_types: list[str] | None = None,
    ) -> go.Figure:
        """ROC curve subplots — one column per anomaly type."""
        if anomaly_types is None:
            anomaly_types = self._get_anomaly_types(dataset)

        labeled_scores = self._merge_scores_w_labels(scores, dataset)
        label_type_col = dataset.col_map["label_type"]

        fig = make_subplots(
            rows=1,
            cols=len(anomaly_types),
            subplot_titles=[t.capitalize() for t in anomaly_types],
            shared_yaxes=True,
        )

        for col, atype in enumerate(anomaly_types, 1):
            scores_atype, binary_labels = self._filter_by_type(labeled_scores, atype, label_type_col)
            if binary_labels.sum() == 0:
                continue

            fpr, tpr, _ = roc_curve(binary_labels, scores_atype["anomaly_score"])
            auc = roc_auc_score(binary_labels, scores_atype["anomaly_score"])

            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{atype} (AUC={auc:.3f})",
                ),
                row=1, col=col,
            )

            # Diagonal baseline
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dot", color="gray"),
                    showlegend=False,
                ),
                row=1, col=col,
            )

        fig.update_xaxes(title_text="FPR", range=[0, 1])
        fig.update_yaxes(title_text="TPR", range=[0, 1], col=1)
        fig.update_layout(
            title_text=f"ROC Curves — {model_name}",
            height=600,
            width=600 * len(anomaly_types),
        )
        return fig

    def compare(
        self,
        scores_dict: dict[str, TimeSeriesDataset],
        dataset: TimeSeriesDataset,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Multi-model comparison table. Rows=anomaly_type, columns per model.

        Runs `metrics_table` per model in parallel via a thread pool (sklearn
        AUC computations release the GIL).
        """
        model_names = list(scores_dict.keys())
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            tables = list(pool.map(
                lambda name: self.metrics_table(scores_dict[name], dataset),
                model_names,
            ))

        metric_tables = []
        for model_name, table in zip(model_names, tables):
            table = table.rename(columns={
                "auc_pr": f"{model_name}_auc_pr",
                "auc_roc": f"{model_name}_auc_roc",
            })
            metric_tables.append(table.set_index("anomaly_type"))

        return pd.concat(metric_tables, axis=1).reset_index()

    def false_positives(
        self,
        scores: TimeSeriesDataset,
        dataset: TimeSeriesDataset,
        threshold: float,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Day-level false positives at a given score threshold.

        Returns rows where the day's ground-truth label is normal but the
        model's anomaly_score is >= threshold, sorted by anomaly_score desc.

        Args:
            scores: Day-level scored TimeSeriesDataset (value_cols=['anomaly_score']).
            dataset: Labeled dataset used to look up label / label_type per day.
            threshold: Score cutoff — rows with anomaly_score >= threshold are
                treated as predicted anomalies.
            top_k: If set, cap output at the top_k highest-scoring false positives.

        Returns:
            DataFrame with columns [entity, 'day', 'anomaly_score', label_type].
        """
        entity_col = dataset.col_map["entity"]
        label_col = dataset.col_map["label"]
        label_type_col = dataset.col_map["label_type"]

        merged = self._merge_scores_w_labels(scores, dataset)
        fp = merged[
            (merged[label_col] == 0) & (merged["anomaly_score"] >= threshold)
        ].sort_values("anomaly_score", ascending=False)
        if top_k is not None:
            fp = fp.head(top_k)
        return fp[[entity_col, "day", "anomaly_score", label_type_col]].reset_index(drop=True)

    def plot_pr_curves_compared(
        self,
        scores_dict: dict[str, TimeSeriesDataset],
        dataset: TimeSeriesDataset,
        max_workers: int | None = None,
    ) -> go.Figure:
        """Overlay PR curves from multiple models. One subplot per anomaly type.

        Per-model merge + PR/AP computation runs in parallel; figure mutation
        stays on the main thread.
        """
        anomaly_types = self._get_anomaly_types(dataset)
        label_type_col = dataset.col_map["label_type"]
        colors = ["#5470C6", "#EE6666", "#5DBCD2", "#FAC858", "#91CC75"]

        fig = make_subplots(
            rows=1,
            cols=len(anomaly_types),
            subplot_titles=[t.capitalize() for t in anomaly_types],
            shared_yaxes=True,
        )

        model_names = list(scores_dict.keys())
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pr_data_per_model = list(pool.map(
                lambda name: self._compute_pr_data(
                    scores_dict[name], dataset, anomaly_types, label_type_col,
                ),
                model_names,
            ))

        for model_idx, (model_name, pr_data) in enumerate(zip(model_names, pr_data_per_model)):
            color = colors[model_idx % len(colors)]
            show_baseline = (model_idx == 0)
            self._render_pr_traces(
                fig, pr_data, anomaly_types, model_name,
                color=color, show_baseline=show_baseline,
            )

        fig.update_xaxes(title_text="Recall", range=[0, 1])
        fig.update_yaxes(title_text="Precision", range=[0, 1], col=1)
        fig.update_layout(
            title_text="PR Curves — Model Comparison",
            height=600,
            width=600 * len(anomaly_types),
        )
        return fig

    # --- Private helpers ---

    def _merge_scores_w_labels(self, scores: TimeSeriesDataset, dataset: TimeSeriesDataset) -> pd.DataFrame:
        entity_col = dataset.col_map["entity"]
        day_labels = dataset.day_labels()
        merged = scores.df.merge(day_labels, on=[entity_col, "day"])
        # Drop days that couldn't be scored (e.g. pre-history days in MP)
        return merged.dropna(subset=["anomaly_score"])

    def _get_anomaly_types(self, dataset: TimeSeriesDataset) -> list[str]:
        return [t for t in dataset.anomaly_types() if t != "normal"]

    def _filter_by_type(
        self, labeled_scores: pd.DataFrame, anomaly_type: str, label_type_col: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        mask = labeled_scores[label_type_col].isin([anomaly_type, "normal"])
        subset_df = labeled_scores[mask]
        binary_labels = (subset_df[label_type_col] == anomaly_type).astype(int)
        return subset_df, binary_labels

    def _compute_pr_data(
        self,
        scores: TimeSeriesDataset,
        dataset: TimeSeriesDataset,
        anomaly_types: list[str],
        label_type_col: str,
    ) -> list[dict | None]:
        """Thread-safe: merge + PR/AP per anomaly type. No figure mutation."""
        labeled_scores = self._merge_scores_w_labels(scores, dataset)
        out: list[dict | None] = []
        for atype in anomaly_types:
            scores_atype, binary_labels = self._filter_by_type(labeled_scores, atype, label_type_col)
            if binary_labels.sum() == 0:
                out.append(None)
                continue
            prec, rec, thr = precision_recall_curve(binary_labels, scores_atype["anomaly_score"])
            thr_padded = np.concatenate([thr, [np.nan]])
            ap = average_precision_score(binary_labels, scores_atype["anomaly_score"])
            out.append({
                "prec": prec, "rec": rec, "thresholds": thr_padded,
                "ap": ap, "baseline": binary_labels.mean(),
            })
        return out

    def _render_pr_traces(
        self,
        fig: go.Figure,
        pr_data: list[dict | None],
        anomaly_types: list[str],
        model_name: str,
        color: str | None = None,
        show_baseline: bool = True,
        showlegend: bool = True,
    ) -> None:
        """Main-thread only: add precomputed PR traces to figure."""
        for col, (atype, data) in enumerate(zip(anomaly_types, pr_data), 1):
            if data is None:
                print(f"No cases found with anomaly type = {atype}")
                continue

            trace_kwargs = dict(
                x=data["rec"], y=data["prec"], mode="lines",
                name=f"{model_name} (AP={data['ap']:.3f})",
                legendgroup=model_name,
                showlegend=(showlegend and col == 1),
                customdata=data["thresholds"],
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "Recall: %{x:.3f}<br>"
                    "Precision: %{y:.3f}<br>"
                    "Threshold: %{customdata:.4f}<extra></extra>"
                ),
            )
            if color is not None:
                trace_kwargs["line"] = dict(color=color)

            fig.add_trace(go.Scatter(**trace_kwargs), row=1, col=col)

            if show_baseline:
                fig.add_hline(y=data["baseline"], line_dash="dot", line_color="gray", row=1, col=col)

    def _add_pr_traces(
        self,
        fig: go.Figure,
        labeled_scores: pd.DataFrame,
        label_type_col: str,
        anomaly_types: list[str],
        model_name: str,
        color: str | None = None,
        show_baseline: bool = True,
        showlegend: bool = True,
    ) -> None:
        """Compute + render PR traces for a single pre-merged labeled_scores frame."""
        pr_data = []
        for atype in anomaly_types:
            scores_atype, binary_labels = self._filter_by_type(labeled_scores, atype, label_type_col)
            if binary_labels.sum() == 0:
                pr_data.append(None)
                continue
            prec, rec, thr = precision_recall_curve(binary_labels, scores_atype["anomaly_score"])
            thr_padded = np.concatenate([thr, [np.nan]])
            ap = average_precision_score(binary_labels, scores_atype["anomaly_score"])
            pr_data.append({
                "prec": prec, "rec": rec, "thresholds": thr_padded,
                "ap": ap, "baseline": binary_labels.mean(),
            })
        self._render_pr_traces(
            fig, pr_data, anomaly_types, model_name,
            color=color, show_baseline=show_baseline, showlegend=showlegend,
        )
