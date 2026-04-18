"""Feature transformer: rolling Catch22 profiles on sub_entity series and pairwise diffs."""

import itertools
from enum import Enum

import numpy as np
import pandas as pd
import pycatch22
import plotly.graph_objects as go
import plotly.express as px
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from core.dataset import TimeSeriesDataset


class FeatureCategory(str, Enum):
    C22_RAW = "c22_raw"
    C22_DIFF = "c22_diff"


# ---------------------------------------------------------------------------
# Module-level helpers (top-level so joblib can pickle them)
# ---------------------------------------------------------------------------

def _rolling_c22(
    series: np.ndarray, window_size: int, stride: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Slide a window over *series* and compute 22 catch22 features per position.

    With stride > 1, windows are taken every `stride` positions — reduces
    compute by ~stride× at the cost of temporal resolution.
    """
    n_windows = (len(series) - window_size) // stride + 1
    features = np.empty((n_windows, 22))
    names = None
    for i in range(n_windows):
        start = i * stride
        result = pycatch22.catch22_all(series[start : start + window_size])
        features[i, :] = result["values"]
        if names is None:
            names = result["names"]
    return features, names


def _process_entity(
    entity_df: pd.DataFrame,
    entity_id,
    entity_col: str,
    time_col: str,
    sub_entity_col: str,
    raw_data_columns: list[str],
    categories: list[FeatureCategory],
    window_size: int,
    stride: int = 1,
) -> tuple[pd.DataFrame, list[tuple[list[str], FeatureCategory]]]:
    """Compute all c22 features for one entity. Called once per entity by joblib.

    Returns
    -------
    result_df
        Wide DataFrame: one row per window position, columns = all c22 features
        plus entity_col and time_col.
    column_categories
        [(column_names, FeatureCategory), ...] — one entry per input series.
        The caller uses the first entity's value to build feature_map.
    """
    # --- pivot: rows=timestamps, one column per sub_entity ---
    wide = entity_df.pivot_table(
        index=time_col, columns=sub_entity_col, values=raw_data_columns[0],
    ).sort_index()
    timestamps = wide.index.values

    if len(raw_data_columns) > 1:
        wide_per_col = {
            rc: entity_df.pivot_table(index=time_col, columns=sub_entity_col, values=rc).sort_index()
            for rc in raw_data_columns
        }
    else:
        wide_per_col = {raw_data_columns[0]: wide}

    sub_entities = sorted(wide.columns)
    pairs = list(itertools.combinations(sub_entities, 2))

    # --- build named 1-D series to featurize ---
    # Each entry: (column_prefix, values_1d, category)
    named_series: list[tuple[str, np.ndarray, FeatureCategory]] = []
    for raw_col, wide_df in wide_per_col.items():
        if FeatureCategory.C22_RAW in categories:
            for sub_ent in sub_entities:
                named_series.append(
                    (f"{raw_col}__{sub_ent}", wide_df[sub_ent].values, FeatureCategory.C22_RAW)
                )
        if FeatureCategory.C22_DIFF in categories:
            for a, b in pairs:
                named_series.append(
                    (f"{raw_col}__{a}_{b}", wide_df[a].values - wide_df[b].values, FeatureCategory.C22_DIFF)
                )

    # --- rolling c22 on every series, collect columns ---
    all_feature_arrays = []
    all_column_names: list[str] = []
    column_categories: list[tuple[list[str], FeatureCategory]] = []

    for prefix, values, category in named_series:
        feat_array, c22_names = _rolling_c22(values, window_size, stride)
        all_feature_arrays.append(feat_array)
        cols = [f"{prefix}__{n}" for n in c22_names]
        all_column_names.extend(cols)
        column_categories.append((cols, category))

    # --- assemble DataFrame ---
    feature_matrix = np.concatenate(all_feature_arrays, axis=1)
    n_windows = feature_matrix.shape[0]

    result_df = pd.DataFrame(feature_matrix, columns=all_column_names)
    result_df[entity_col] = entity_id
    # Each row's timestamp is the start of its window; strided windows skip timestamps accordingly.
    result_df[time_col] = timestamps[::stride][:n_windows]
    return result_df, column_categories


def _format_feature_label(feat_col: str) -> str:
    """'TmpRet__0__CO_f1ecac' -> 'c22_raw: 0 / CO_f1ecac'"""
    parts = feat_col.split("__")
    if len(parts) != 3:
        return feat_col
    _raw_col, source, c22_name = parts
    cat = FeatureCategory.C22_DIFF if "_" in str(source) else FeatureCategory.C22_RAW
    return f"{cat.value}: {source} / {c22_name}"


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class FeatureTransformer:
    """Compute rolling Catch22 feature profiles from a TimeSeriesDataset.

    Args:
        raw_data_columns: Which value columns to compute features on.
        window_size: Rolling window length for catch22.
        stride: Step between consecutive windows. Defaults to window_size // 4
            (4x fewer c22 calls than stride=1, minor loss of temporal resolution
            since adjacent windows at stride=1 are near-duplicates).
        n_jobs: Parallelism across entities (-1 = all cores).
    """

    def __init__(
        self,
        raw_data_columns: list[str],
        window_size: int = 720,
        stride: int | None = None,
        n_jobs: int = -1,
    ):
        self.raw_data_columns = raw_data_columns
        self.window_size = window_size
        self.stride = stride if stride is not None else max(1, window_size // 4)
        self.n_jobs = n_jobs
        self._feature_map: dict[FeatureCategory, list[str]] | None = None

    @property
    def feature_map(self) -> dict[FeatureCategory, list[str]]:
        """Feature columns grouped by category. Available after transform()."""
        if self._feature_map is None:
            raise RuntimeError("feature_map not available — call transform() first")
        return self._feature_map

    def transform(
        self,
        dataset: TimeSeriesDataset,
        categories: list[FeatureCategory] | None = None,
    ) -> TimeSeriesDataset:
        """Compute rolling c22 feature profiles for all entities (parallelized).

        Args:
            dataset: Must have 'sub_entity' in col_map.
            categories: Which categories to compute. None = all.

        Returns:
            TimeSeriesDataset — one row per (entity, timestamp), no sub_entity.
        """
        dataset._require("sub_entity")
        if categories is None:
            categories = list(FeatureCategory)

        entity_col = dataset.col_map["entity"]
        time_col = dataset.col_map["time"]
        sub_entity_col = dataset.col_map["sub_entity"]

        entity_groups = [
            (eid, dataset.df[dataset.df[entity_col] == eid])
            for eid in sorted(dataset.df[entity_col].unique())
        ]

        results = list(tqdm(
            Parallel(n_jobs=self.n_jobs, return_as="generator")(
                delayed(_process_entity)(
                    grp, eid,
                    entity_col, time_col, sub_entity_col,
                    self.raw_data_columns, categories, self.window_size,
                    self.stride,
                )
                for eid, grp in entity_groups
            ),
            total=len(entity_groups),
            desc="FeatureTransformer",
        ))

        # Build feature_map from first entity
        _, first_column_categories = results[0]
        feature_map: dict[FeatureCategory, list[str]] = {c: [] for c in categories}
        feature_col_names: list[str] = []
        for col_names, category in first_column_categories:
            feature_col_names.extend(col_names)
            feature_map[category].extend(col_names)
        self._feature_map = feature_map

        result_df = pd.concat([df for df, _ in results], ignore_index=True)
        return TimeSeriesDataset(result_df, {
            "entity": entity_col,
            "time": time_col,
            "value_cols": feature_col_names,
        })

    # -------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------

    def visualize(
        self,
        original_dataset: TimeSeriesDataset,
        feature_dataset: TimeSeriesDataset,
        entity_id,
        feature_names: list[str] | None = None,
    ) -> go.Figure:
        """Plot raw TS (top) + selected feature traces (bottom) for one entity."""
        entity_col = original_dataset.col_map["entity"]
        time_col = original_dataset.col_map["time"]
        sub_entity_col = original_dataset.col_map["sub_entity"]
        value_cols = original_dataset.col_map["value_cols"]
        label_col = original_dataset.col_map.get("label")

        raw_e = original_dataset.df[original_dataset.df[entity_col] == entity_id].sort_values(time_col)
        feat_e = feature_dataset.df[feature_dataset.df[entity_col] == entity_id].sort_values(time_col)

        if feature_names is None:
            feature_names = feature_dataset.col_map["value_cols"][:5]

        n_raw = len(value_cols)
        n_feat = len(feature_names)
        feat_labels = [_format_feature_label(f) for f in feature_names]

        fig = make_subplots(
            rows=n_raw + n_feat, cols=1, shared_xaxes=True,
            vertical_spacing=max(0.005, 0.15 / (n_raw + n_feat)),
            subplot_titles=value_cols + feat_labels,
            row_heights=[3] * n_raw + [1] * n_feat,
            specs=[[{"secondary_y": True}]] * n_raw + [[{"secondary_y": False}]] * n_feat,
        )

        colors = px.colors.qualitative.Plotly
        sub_entities = sorted(raw_e[sub_entity_col].unique())
        sub_colors = {s: colors[i % len(colors)] for i, s in enumerate(sub_entities)}

        # Raw TS rows
        for row, vcol in enumerate(value_cols, start=1):
            for se in sub_entities:
                se_data = raw_e[raw_e[sub_entity_col] == se]
                fig.add_trace(
                    go.Scatter(
                        x=se_data[time_col], y=se_data[vcol],
                        name=f"{sub_entity_col} {se}", mode="lines",
                        line=dict(color=sub_colors[se]),
                        showlegend=(row == 1), legendgroup=f"sub_{se}",
                    ),
                    row=row, col=1, secondary_y=False,
                )
            if label_col is not None:
                anom = raw_e[raw_e[label_col].astype(bool)]
                if len(anom) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=anom[time_col], y=anom[label_col].astype(int),
                            name="Anomaly", mode="markers",
                            marker=dict(size=3, color="red"),
                            showlegend=(row == 1), legendgroup="anomaly",
                        ),
                        row=row, col=1, secondary_y=True,
                    )

        # Feature rows
        feat_timestamps = feat_e[time_col].values
        for i, (fname, flabel) in enumerate(zip(feature_names, feat_labels)):
            fig.add_trace(
                go.Scatter(
                    x=feat_timestamps, y=feat_e[fname].values,
                    mode="lines", name=flabel, showlegend=False,
                ),
                row=n_raw + i + 1, col=1,
            )

        entity_label = original_dataset._get_entity_label_type(raw_e) if label_col else "unknown"
        fig.update_layout(
            height=250 * n_raw + 80 * n_feat + 100, width=1200,
            title_text=f"Entity {entity_id} — {entity_label.capitalize()} — Raw TS + Features",
            hovermode="x unified",
        )
        fig.update_annotations(font_size=9)
        return fig
