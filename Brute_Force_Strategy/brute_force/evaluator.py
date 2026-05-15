"""Evaluation: precision curves, heatmaps, and combo-level metric aggregation.

Metric definition
-----------------
``precision_at_threshold(ranked_genes, pct)`` mirrors the heatmap in predict.py:

    k = ceil(N_targets * pct / 100)
    precision = (targets found in top-k ranked genes) / k

The x-axis therefore represents *prediction depth as a fraction of the known
target count*, not a fraction of all genes.  This directly answers the
biologically relevant question: "of my top-k predictions, how many are real?"

All functions are pure (no I/O, no side-effects) so they can be tested
independently and called from parallel workers without locks.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

import numpy as np
import pandas as pd

from ._types import ComboResult, RankedGene, PrecisionCurve, TrainResult

logger = logging.getLogger(__name__)

# Percentage thresholds at which precision is evaluated.
PRECISION_THRESHOLDS: List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# np.trapezoid is the NumPy 2.0 name; np.trapz was the legacy alias (removed in 2.0).
try:
    _trapezoid = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:
    _trapezoid = np.trapz  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Precision curve
# ---------------------------------------------------------------------------

def precision_at_threshold(
    ranked_genes: List[RankedGene],
    pct: int,
) -> float:
    """Precision at k, where k = ceil(N_targets * pct / 100).

    Returns the fraction of the top-k ranked genes that are known targets.
    Returns 0.0 when there are no targets or no genes.
    """
    n_targets = sum(1 for g in ranked_genes if g.is_target)
    if n_targets == 0 or not ranked_genes:
        return 0.0

    cutoff = max(1, math.ceil(n_targets * pct / 100))
    cutoff = min(cutoff, len(ranked_genes))
    found  = sum(1 for g in ranked_genes[:cutoff] if g.is_target)
    return found / cutoff


def precision_curve(ranked_genes: List[RankedGene]) -> PrecisionCurve:
    """Compute precision@k for every threshold in PRECISION_THRESHOLDS."""
    return {
        f"precision_at_{pct}pct": precision_at_threshold(ranked_genes, pct)
        for pct in PRECISION_THRESHOLDS
    }


def auc_precision_curve(curve: PrecisionCurve) -> float:
    """Area under the precision curve (trapezoid, normalised to [0, 1]).

    Uses the evenly-spaced PRECISION_THRESHOLDS as x-axis values.
    """
    values = [curve[f"precision_at_{pct}pct"] for pct in PRECISION_THRESHOLDS]
    return float(_trapezoid(values) / (len(values) - 1))


# ---------------------------------------------------------------------------
# Heatmap DataFrame
# ---------------------------------------------------------------------------

def build_heatmap_df(results: List[TrainResult]) -> pd.DataFrame:
    """Build the precision heatmap DataFrame for a fixed training dataset.

    Parameters
    ----------
    results:
        All TrainResult objects with the same ``train_dataset``, one per
        test dataset (including self-test).

    Returns
    -------
    DataFrame with:
        - index  = test dataset names
        - columns = ["precision_at_10pct", ..., "precision_at_100pct"]
    All values in [0, 1].
    """
    rows = {}
    for r in results:
        curve = precision_curve(r.ranked_genes)
        rows[r.test_dataset] = curve
    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df[[f"precision_at_{p}pct" for p in PRECISION_THRESHOLDS]]
    df.index.name = "test_dataset"
    return df


# ---------------------------------------------------------------------------
# Combo-level summary aggregation
# ---------------------------------------------------------------------------

def aggregate_combo_summary(
    all_results: List[TrainResult],
) -> Dict:
    """Compute aggregate metrics across all (train_dataset, test_dataset) pairs.

    Returns a dict suitable for writing to combo_summary.json.  Contains:
    - mean_precision_at_Xpct for every threshold
    - mean_auc
    - per_dataset_summary: per training dataset -> mean precision across test sets
    - mean_feature_importance: averaged importance across training datasets
    """
    if not all_results:
        return {}

    all_curves: List[PrecisionCurve] = []
    per_train_curves: Dict[str, List[PrecisionCurve]] = {}
    all_importances: List[Dict[str, float]] = []

    for r in all_results:
        curve = precision_curve(r.ranked_genes)
        all_curves.append(curve)
        per_train_curves.setdefault(r.train_dataset, []).append(curve)
        all_importances.append(r.feature_importances)

    # Global means across every (train, test) pair.
    summary: Dict = {}
    for pct in PRECISION_THRESHOLDS:
        key = f"precision_at_{pct}pct"
        summary[f"mean_{key}"] = float(
            np.mean([c[key] for c in all_curves])
        )

    summary["mean_auc"] = float(
        np.mean([auc_precision_curve(c) for c in all_curves])
    )

    # Per-training-dataset summaries (mean across test datasets).
    per_dataset: Dict[str, Dict] = {}
    for td, curves in per_train_curves.items():
        ds_summary: Dict = {}
        for pct in PRECISION_THRESHOLDS:
            key = f"precision_at_{pct}pct"
            ds_summary[f"mean_{key}"] = float(np.mean([c[key] for c in curves]))
        ds_summary["mean_auc"] = float(
            np.mean([auc_precision_curve(c) for c in curves])
        )
        per_dataset[td] = ds_summary
    summary["per_dataset_summary"] = per_dataset

    # Mean feature importances across all training runs.
    if all_importances:
        feature_names = list(all_importances[0].keys())
        mean_imp = {
            f: float(np.mean([imp.get(f, 0.0) for imp in all_importances]))
            for f in feature_names
        }
        summary["mean_feature_importance"] = dict(
            sorted(mean_imp.items(), key=lambda kv: kv[1], reverse=True)
        )

    return summary
