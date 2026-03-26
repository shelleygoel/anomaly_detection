"""Evaluation class for anomaly detection models.

Joins day-level scores with day-level labels, computes AUC-PR/ROC metrics,
and plots PR/ROC curves broken down by anomaly type.
"""

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
            height=400,
            width=400 * len(anomaly_types),
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
            height=400,
            width=400 * len(anomaly_types),
        )
        return fig

    def compare(
        self,
        scores_dict: dict[str, TimeSeriesDataset],
        dataset: TimeSeriesDataset,
    ) -> pd.DataFrame:
        """Multi-model comparison table. Rows=anomaly_type, columns per model."""
        metric_tables = []
        for model_name, scores in scores_dict.items():
            table = self.metrics_table(scores, dataset)
            table = table.rename(columns={
                "auc_pr": f"{model_name}_auc_pr",
                "auc_roc": f"{model_name}_auc_roc",
            })
            metric_tables.append(table.set_index("anomaly_type"))

        return pd.concat(metric_tables, axis=1).reset_index()

    def plot_pr_curves_compared(
        self,
        scores_dict: dict[str, TimeSeriesDataset],
        dataset: TimeSeriesDataset,
    ) -> go.Figure:
        """Overlay PR curves from multiple models. One subplot per anomaly type."""
        anomaly_types = self._get_anomaly_types(dataset)
        label_type_col = dataset.col_map["label_type"]
        colors = ["#5470C6", "#EE6666", "#5DBCD2", "#FAC858", "#91CC75"]

        fig = make_subplots(
            rows=1,
            cols=len(anomaly_types),
            subplot_titles=[t.capitalize() for t in anomaly_types],
            shared_yaxes=True,
        )

        for model_idx, (model_name, scores) in enumerate(scores_dict.items()):
            labeled_scores = self._merge_scores_w_labels(scores, dataset)
            color = colors[model_idx % len(colors)]
            show_baseline = (model_idx == 0)

            self._add_pr_traces(
                fig, labeled_scores, label_type_col, anomaly_types,
                model_name, color=color, show_baseline=show_baseline,
            )

        fig.update_xaxes(title_text="Recall", range=[0, 1])
        fig.update_yaxes(title_text="Precision", range=[0, 1], col=1)
        fig.update_layout(
            title_text="PR Curves — Model Comparison",
            height=400,
            width=400 * len(anomaly_types),
        )
        return fig

    # --- Private helpers ---

    def _merge_scores_w_labels(self, scores: TimeSeriesDataset, dataset: TimeSeriesDataset) -> pd.DataFrame:
        entity_col = dataset.col_map["entity"]
        day_labels = dataset.day_labels()
        return scores.df.merge(day_labels, on=[entity_col, "day"])

    def _get_anomaly_types(self, dataset: TimeSeriesDataset) -> list[str]:
        return [t for t in dataset.anomaly_types() if t != "normal"]

    def _filter_by_type(
        self, labeled_scores: pd.DataFrame, anomaly_type: str, label_type_col: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        mask = labeled_scores[label_type_col].isin([anomaly_type, "normal"])
        subset_df = labeled_scores[mask]
        binary_labels = (subset_df[label_type_col] == anomaly_type).astype(int)
        return subset_df, binary_labels

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
        """Add PR curve traces to an existing figure (one trace per anomaly type column)."""
        for col, atype in enumerate(anomaly_types, 1):
            scores_atype, binary_labels = self._filter_by_type(labeled_scores, atype, label_type_col)
            if binary_labels.sum() == 0:
                print(f"No cases found with anomaly type = {atype}")
                continue

            prec, rec, _ = precision_recall_curve(binary_labels, scores_atype["anomaly_score"])
            ap = average_precision_score(binary_labels, scores_atype["anomaly_score"])

            trace_kwargs = dict(
                x=rec, y=prec, mode="lines",
                name=f"{model_name} (AP={ap:.3f})",
                legendgroup=model_name,
                showlegend=(showlegend and col == 1),
            )
            if color is not None:
                trace_kwargs["line"] = dict(color=color)

            fig.add_trace(go.Scatter(**trace_kwargs), row=1, col=col)

            if show_baseline:
                baseline = binary_labels.mean()
                fig.add_hline(y=baseline, line_dash="dot", line_color="gray", row=1, col=col)
