"""
Train a Random Forest model for RBP target prediction.

Uses the 18-feature schema from feature_extraction.py.
Supports configurable normalization, editing thresholds, editing type,
and priority feature boosting via biased feature subsampling.

Two modes:
  1. Single dataset  (--expt, --bg, --targets)
  2. Batch LOO       (--metadata TSV file)

=============================================================================
BATCH MODE  (--metadata)
=============================================================================
Reads a metadata TSV describing N datasets.  For each dataset D:
  - trains a model on D alone (using D's targets_file for labels)
  - saves the model to  outdir/Trained_<label>/
  - records all other datasets as the intended test set

After all models are trained, generates cross-model comparison plots in
outdir/comparison/.  Use cross_dataset_heatmap.py --loo-dir to run each
model against the datasets it was NOT trained on.

Metadata TSV format (tab-separated, header required):
    label           expt_files                      ctrl_files                      targets_file        background_files
    Ataxin-2        data/Atx_1.txt;data/Atx_2.txt   data/Adar_1.txt;data/Adar_2.txt targets/atx.txt     gDNA_1.txt;gDNA_2.txt
    Thor            data/Thor_1.txt;Thor_2.txt       data/BG_1.txt;BG_2.txt          targets/thor.txt

  - expt_files and ctrl_files: semicolon-separated replicate paths
    (ctrl_files was called bg_files in older versions — both accepted)
  - targets_file:  path to known target genes
  - background_files: optional — background control files per dataset
    (e.g. gDNA or no-enzyme control) for site-level noise filtering

=============================================================================
SINGLE DATASET MODE  (--expt / --ctrl / --targets)
=============================================================================
python train_model.py \\
    --expt  Expt_1.txt Expt_2.txt \\
    --ctrl  Ctrl_1.txt Ctrl_2.txt \\
    --targets known_targets.txt    \\
    --background gDNA_1.txt        \\  # optional
    --outdir outputs/model

=============================================================================
Outputs (single):
    rf_model.pkl           - trained Random Forest model
    model_config.json      - training parameters
    gene_features.tsv      - computed features for all genes
    predictions.tsv        - ranked gene predictions
    feature_importance.tsv - feature importance (duplicates aggregated)
    topk_metrics.tsv       - precision at top-K thresholds
    plots/                 - per-model diagnostic plots

Outputs (batch LOO):
    LOO_<label>/           - all of the above for each held-out dataset
    comparison/            - cross-model comparison plots + training_summary.tsv
"""

from __future__ import annotations
import argparse
import json
import logging
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from scipy.stats import pearsonr

# Optional gradient-boosting libraries — gracefully degrade if missing
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

from feature_extraction import (
    load_editing_sites,
    build_features,
    FEATURE_COLUMNS,
    FEATURE_SETS,
    VALID_FEATURE_SETS,
    get_feature_columns,
    learn_background_filter,
    save_background_filter,
    load_background_filter,
)

warnings.filterwarnings("ignore")


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: str | Path | None = None, prefix: str = "train") -> Path | None:
    if log_dir is None:
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"{prefix}_{timestamp}.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    class LogPrinter:
        def __init__(self, logger):
            self._logger = logger
        def write(self, msg):
            if msg and msg.strip():
                self._logger.info(msg.rstrip())
        def flush(self):
            pass

    sys.stdout = LogPrinter(root)
    logging.info(f"Log file: {log_path}")
    logging.info(f"Started:  {datetime.now().isoformat()}")
    logging.info("")
    return log_path


# ============================================================================
# Feature Weight System
# ============================================================================

_WEIGHT_BASE = 10


def load_feature_weights_file(path: str) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature weights file not found: {path}")
    with open(path) as f:
        first_data = f.readline()
    sep = "\t" if "\t" in first_data else ","
    df = pd.read_csv(path, sep=sep, comment="#", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    if "feature" not in df.columns or "weight" not in df.columns:
        raise ValueError("Feature weights file must have 'feature' and 'weight' columns.")
    weights = {}
    for _, row in df.iterrows():
        feat = row["feature"].strip()
        try:
            w = float(row["weight"].strip())
        except ValueError:
            raise ValueError(f"Invalid weight for '{feat}': '{row['weight']}'")
        weights[feat] = w
    return weights


def apply_feature_weights(X, feature_cols, feature_weights):
    if not feature_weights:
        mapping = {c: c for c in feature_cols}
        return X, list(feature_cols), mapping
    unknown = set(feature_weights.keys()) - set(feature_cols)
    if unknown:
        available = ", ".join(feature_cols)
        raise ValueError(
            f"Unknown feature(s) in feature weights: {sorted(unknown)}. Available: {available}"
        )
    base = _WEIGHT_BASE
    out_blocks, out_cols, expansion_map, dropped, modified = [], [], {}, [], []
    for i, col_name in enumerate(feature_cols):
        w = feature_weights.get(col_name, 1.0)
        n_copies = max(0, round(w * base))
        if n_copies == 0:
            dropped.append(col_name)
            continue
        if w != 1.0:
            modified.append(f"{col_name}: {w}")
        for c in range(n_copies):
            cname = col_name if c == 0 else f"{col_name}__dup{c}"
            out_blocks.append(X[:, i : i + 1])
            out_cols.append(cname)
            expansion_map[cname] = col_name
    X_out = np.hstack(out_blocks)
    print(f"\n  Feature weights applied (base={base}):")
    if dropped:
        print(f"    Dropped (weight=0): {', '.join(dropped)}")
    if modified:
        print(f"    Modified: {', '.join(modified)}")
    print(f"    Matrix: {len(feature_cols)} features -> {len(out_cols)} columns")
    return X_out, out_cols, expansion_map


def aggregate_importance(importance_values, expanded_cols, expansion_mapping,
                         all_original_cols=None):
    raw_imp = dict(zip(expanded_cols, importance_values))
    aggregated = {}
    for exp_col, orig_col in expansion_mapping.items():
        aggregated[orig_col] = aggregated.get(orig_col, 0.0) + raw_imp.get(exp_col, 0.0)
    if all_original_cols:
        for col in all_original_cols:
            if col not in aggregated:
                aggregated[col] = 0.0
    imp_df = pd.DataFrame([
        {"feature": k, "importance": v} for k, v in aggregated.items()
    ]).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


# ============================================================================
# Helpers
# ============================================================================

def _savefig(fig, path_stem: Path):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_targets(targets_file: str) -> set[str]:
    p = Path(targets_file)
    try:
        df = pd.read_csv(p)
        name_col = "Name" if "Name" in df.columns else df.columns[0]
        return {
            g.strip().lower() for g in df[name_col].dropna().astype(str)
            if g.strip() and g.strip().lower() not in ("name", "reg")
        }
    except Exception:
        targets = set()
        with open(p) as f:
            for line in f:
                g = line.strip().lower()
                if g and g not in ("name", "reg"):
                    targets.add(g)
        return targets


def parse_files(s: str) -> list[str]:
    return [f.strip() for f in s.split(";") if f.strip()]


def load_metadata(path: Path) -> pd.DataFrame:
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                first = line
                break
    sep = "\t" if "\t" in first else ","
    df = pd.read_csv(path, sep=sep, comment="#", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {}
    # Accept dataset_name as alias for label
    if "dataset_name" in df.columns and "label" not in df.columns:
        rename_map["dataset_name"] = "label"
    # Accept bg_files as alias for ctrl_files (backward compat)
    if "bg_files" in df.columns and "ctrl_files" not in df.columns:
        rename_map["bg_files"] = "ctrl_files"
    # Accept validation_targets as alias for targets_file
    if "validation_targets" in df.columns and "targets_file" not in df.columns:
        rename_map["validation_targets"] = "targets_file"
    if rename_map:
        df = df.rename(columns=rename_map)
    required = {"label", "expt_files", "ctrl_files", "targets_file"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata file missing columns: {missing}")
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df.reset_index(drop=True)


def load_dataset_features(row: pd.Series, normalization: str, editing_type: str,
                           min_edit_pct: float, min_reads: int,
                           bg_filter_model: object = None,
                           feature_set: str = "full22",
                           min_total_reads: int = 0,
                           min_fold_change: float = 0.0):
    expt_dfs = []
    for p in parse_files(row["expt_files"]):
        df = load_editing_sites(Path(p), editing_type=editing_type,
                                min_edit_pct=min_edit_pct, min_reads=min_reads)
        expt_dfs.append(df)
    ctrl_dfs = []
    for p in parse_files(row["ctrl_files"]):
        df = load_editing_sites(Path(p), editing_type=editing_type,
                                min_edit_pct=min_edit_pct, min_reads=min_reads)
        ctrl_dfs.append(df)
    gene_names = sorted(set().union(*(df["Name"].unique() for df in expt_dfs + ctrl_dfs)))
    features_df = build_features(expt_dfs, ctrl_dfs, gene_names,
                                 normalization=normalization,
                                 bg_filter_model=bg_filter_model,
                                 feature_set=feature_set,
                                 min_total_reads=min_total_reads,
                                 min_fold_change=min_fold_change)
    targets_set = load_targets(row["targets_file"])
    return features_df, targets_set


# ============================================================================
# Per-model diagnostic plots
# ============================================================================

def generate_per_model_plots(
    predictions_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    cv_scores: list[float],
    cv_oof_probs: np.ndarray,
    cv_oof_labels: np.ndarray,
    final_probs: np.ndarray,
    outdir: Path,
    label: str = "model",
):
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    y       = cv_oof_labels
    y_score = cv_oof_probs
    has_two_classes = len(np.unique(y)) == 2

    # ── 1. ROC curve ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    if has_two_classes:
        fpr, tpr, _ = roc_curve(y, y_score)
        auc_val = roc_auc_score(y, y_score)
        ax.plot(fpr, tpr, lw=2, color="#2196F3",
                label=f"CV out-of-fold  (AUC={auc_val:.3f})")
        fpr_f, tpr_f, _ = roc_curve(y, final_probs)
        auc_f = roc_auc_score(y, final_probs)
        ax.plot(fpr_f, tpr_f, lw=2, color="#FF5722", ls="--",
                label=f"Final model (train)  (AUC={auc_f:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, plots_dir / "roc_curve")

    # ── 2. Precision–Recall curve ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    if has_two_classes:
        prec_cv, rec_cv, _ = precision_recall_curve(y, y_score)
        ap_cv = average_precision_score(y, y_score)
        ax.plot(rec_cv, prec_cv, lw=2, color="#2196F3",
                label=f"CV out-of-fold  (AP={ap_cv:.3f})")
        prec_f, rec_f, _ = precision_recall_curve(y, final_probs)
        ap_f = average_precision_score(y, final_probs)
        ax.plot(rec_f, prec_f, lw=2, color="#FF5722", ls="--",
                label=f"Final model (train)  (AP={ap_f:.3f})")
        ax.axhline(y.mean(), color="grey", ls=":", lw=1.5,
                   label=f"Baseline ({y.mean():.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision–Recall Curve — {label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, plots_dir / "precision_recall_curve")

    # ── 3. Feature importance ──────────────────────────────────────────────
    top_n = min(len(FEATURE_COLUMNS), len(importance_df))
    top_imp = importance_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45 + 1.5)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, top_n))[::-1]
    ax.barh(range(top_n), top_imp["importance"].values[::-1], color=colors, alpha=0.9)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_imp["feature"].values[::-1], fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=11)
    ax.set_title(f"Feature Importance — {label}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, plots_dir / "feature_importance")

    # ── 4. CV fold scores ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fold_cols = plt.cm.Blues(np.linspace(0.4, 0.85, len(cv_scores)))
    ax.bar(range(1, len(cv_scores) + 1), cv_scores, color=fold_cols, alpha=0.9)
    mean_auc = np.nanmean(cv_scores)
    std_auc  = np.nanstd(cv_scores)
    ax.axhline(mean_auc, color="#E53935", lw=2, ls="--",
               label=f"Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    ax.fill_between([0.5, len(cv_scores) + 0.5],
                    mean_auc - std_auc, mean_auc + std_auc,
                    color="#E53935", alpha=0.12)
    ax.set_xticks(range(1, len(cv_scores) + 1))
    ax.set_xticklabels([f"Fold {i}" for i in range(1, len(cv_scores) + 1)])
    ax.set_ylabel("ROC-AUC", fontsize=11)
    lo = max(0.0, mean_auc - 0.25)
    hi = min(1.05, mean_auc + 0.25)
    ax.set_ylim(lo, hi)
    ax.set_title(f"CV Fold ROC-AUC Scores — {label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, plots_dir / "cv_fold_scores")

    # ── 5. Score distribution ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    scores_pos = final_probs[y == 1]
    scores_neg = final_probs[y == 0]
    bins = np.linspace(0, 1, 40)
    ax.hist(scores_neg, bins=bins, alpha=0.6, color="#64B5F6",
            label="Non-targets", density=True)
    ax.hist(scores_pos, bins=bins, alpha=0.7, color="#EF5350",
            label="Targets", density=True)
    ax.set_xlabel("RF Prediction Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Score Distribution — {label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, plots_dir / "score_distribution")

    # ── 6. Top-K precision ─────────────────────────────────────────────────
    if len(metrics_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(metrics_df["k"], metrics_df["precision"] * 100,
                marker="o", lw=2, color="#7B1FA2", markersize=7)
        ax.fill_between(metrics_df["k"], 0, metrics_df["precision"] * 100,
                        alpha=0.12, color="#7B1FA2")
        ax.set_xlabel("Top-K genes", fontsize=12)
        ax.set_ylabel("Precision (%)", fontsize=12)
        ax.set_title(f"Top-K Precision — {label}", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _savefig(fig, plots_dir / "topk_precision")

    print(f"  Per-model plots saved: {plots_dir}")


# ============================================================================
# Cross-model comparison plots
# ============================================================================

def generate_model_comparison_plots(model_results: list[dict], outdir: Path):
    """
    Generate cross-model comparison plots after LOO batch training.

    Each entry in model_results is a dict:
        label, cv_scores, cv_mean_auc, cv_std_auc,
        importance_df, metrics_df, n_genes, n_targets
    """
    if len(model_results) < 2:
        print("  (Skipping cross-model plots: need at least 2 models)")
        return

    comp_dir = outdir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    labels   = [r["label"] for r in model_results]
    n_models = len(labels)
    palette  = plt.cm.tab10(np.linspace(0, 0.9, n_models))

    all_features = list(FEATURE_COLUMNS)

    # Build feature importance matrix (models × features)
    imp_rows = []
    for r in model_results:
        imp = r["importance_df"].set_index("feature")["importance"]
        imp_rows.append([float(imp.get(f, 0.0)) for f in all_features])
    imp_df = pd.DataFrame(imp_rows, index=labels, columns=all_features)
    imp_df.to_csv(comp_dir / "feature_importance_matrix.tsv", sep="\t")

    # ── 1. Feature importance heatmap ─────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(max(14, len(all_features) * 0.85 + 3),
                 max(4, n_models * 0.75 + 2))
    )
    sns.heatmap(imp_df, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.4, ax=ax, cbar_kws={"label": "MDI Importance"})
    ax.set_title("Feature Importance Across LOO Models",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("LOO Model (held-out dataset)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    _savefig(fig, comp_dir / "feature_importance_heatmap")

    # ── 2. Feature rank stability heatmap ─────────────────────────────────
    rank_df = imp_df.rank(axis=1, ascending=False)
    fig, ax = plt.subplots(
        figsize=(max(14, len(all_features) * 0.85 + 3),
                 max(4, n_models * 0.75 + 2))
    )
    sns.heatmap(rank_df, annot=True, fmt=".0f", cmap="RdYlGn_r",
                vmin=1, vmax=len(all_features),
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "Rank (1 = most important)"})
    ax.set_title("Feature Importance Rank Stability Across LOO Models",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("LOO Model", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    _savefig(fig, comp_dir / "feature_rank_stability_heatmap")

    # ── 3. CV AUC comparison bar chart ────────────────────────────────────
    means = [r["cv_mean_auc"] for r in model_results]
    stds  = [r["cv_std_auc"]  for r in model_results]
    sorted_idx = np.argsort(means)[::-1]
    s_labels = [labels[i] for i in sorted_idx]
    s_means  = [means[i]  for i in sorted_idx]
    s_stds   = [stds[i]   for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(4, n_models * 0.65 + 2)))
    bar_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, n_models))
    bars = ax.barh(range(n_models), s_means, xerr=s_stds,
                   color=bar_colors, alpha=0.9, capsize=4,
                   error_kw={"lw": 1.5})
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(s_labels, fontsize=9)
    ax.invert_yaxis()
    for i, (m, s) in enumerate(zip(s_means, s_stds)):
        ax.text(m + s + 0.005, i, f"{m:.3f}±{s:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean CV ROC-AUC (5-fold ± SD)", fontsize=11)
    ax.set_title("CV AUC Across LOO Models", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.2)
    ax.axvline(0.5, color="grey", ls="--", lw=1, alpha=0.6, label="Random (0.5)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, comp_dir / "cv_auc_comparison")

    # ── 4. CV fold score distributions (box + jitter) ─────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.2 + 2), 5))
    all_folds = [r["cv_scores"] for r in model_results]
    bp = ax.boxplot(all_folds, patch_artist=True, notch=False,
                    medianprops={"color": "black", "lw": 2})
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    rng = np.random.default_rng(42)
    for i, (folds, color) in enumerate(zip(all_folds, palette)):
        jitter = rng.uniform(-0.15, 0.15, len(folds))
        ax.scatter([i + 1 + j for j in jitter], folds,
                   color=color, s=35, zorder=5, alpha=0.85)
    ax.set_xticks(range(1, n_models + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("CV Fold ROC-AUC", fontsize=11)
    ax.set_title("CV Fold Score Distributions Across LOO Models",
                 fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, comp_dir / "cv_score_distributions")

    # ── 5. Top-K precision comparison ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for r, color in zip(model_results, palette):
        mdf = r.get("metrics_df")
        if mdf is None or len(mdf) == 0:
            continue
        ax.plot(mdf["k"], mdf["precision"] * 100,
                marker="o", lw=2, color=color,
                label=r["label"], markersize=5)
    ax.set_xlabel("Top-K genes", fontsize=12)
    ax.set_ylabel("Precision (%)", fontsize=12)
    ax.set_title("Top-K Precision Across LOO Models",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, comp_dir / "topk_precision_comparison")

    # ── 6. Model similarity (Pearson r of feature importance vectors) ──────
    n = len(labels)
    sim_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            vi = imp_df.iloc[i].values
            vj = imp_df.iloc[j].values
            if np.std(vi) > 1e-12 and np.std(vj) > 1e-12:
                r_val, _ = pearsonr(vi, vj)
            else:
                r_val = 0.0
            sim_matrix[i, j] = r_val
            sim_matrix[j, i] = r_val

    # Mask upper triangle (keep lower + diagonal)
    mask = np.zeros_like(sim_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    fig, ax = plt.subplots(figsize=(max(5, n * 0.9 + 2), max(5, n * 0.9 + 2)))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, mask=mask,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Model Similarity (Feature Importance Pearson r)",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    _savefig(fig, comp_dir / "model_similarity_heatmap")

    # ── 7. Training data summary ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, n_models * 0.65 + 2)))
    n_genes_list   = [r["n_genes"]   for r in model_results]
    n_targets_list = [r["n_targets"] for r in model_results]
    bar_c = plt.cm.Blues(np.linspace(0.4, 0.85, n_models))

    for ax, vals, title, xlabel in zip(
        axes,
        [n_genes_list, n_targets_list],
        ["Training Gene-Entries per LOO Model", "Training Targets per LOO Model"],
        ["Number of gene-entries", "Number of targets"],
    ):
        ax.barh(range(n_models), vals, color=bar_c, alpha=0.9)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.01, i, f"{v:,}", va="center", fontsize=8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, comp_dir / "training_data_summary")

    # ── 8. Consolidated feature RANK summary (avg rank, SD, re-rank) ──────
    rank_df = imp_df.rank(axis=1, ascending=False)  # per-model rank of each feature

    avg_rank = rank_df.mean(axis=0)
    std_rank = rank_df.std(axis=0)

    # Re-rank: sort by avg_rank ascending (best avg rank = re-rank 1)
    rank_summary = (
        pd.DataFrame({"feature": avg_rank.index,
                      "avg_rank": avg_rank.values,
                      "std_rank": std_rank.values})
        .sort_values("avg_rank")
        .reset_index(drop=True)
    )
    rank_summary.insert(0, "consolidated_rank", range(1, len(rank_summary) + 1))

    # Also attach each model's individual rank for reference
    for lbl in labels:
        rank_summary[f"rank_{lbl}"] = rank_summary["feature"].map(
            rank_df.loc[lbl].to_dict()
        )

    rank_summary.to_csv(comp_dir / "feature_consolidated_rank.tsv", sep="\t", index=False)

    # ── Plot: consolidated rank bar chart with SD + re-rank annotation ────
    n_feat = len(rank_summary)
    bar_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_feat))

    fig, ax = plt.subplots(figsize=(11, max(5, n_feat * 0.48 + 2)))
    bars = ax.barh(range(n_feat), rank_summary["avg_rank"].values,
                   xerr=rank_summary["std_rank"].values,
                   color=bar_colors, alpha=0.88, capsize=4,
                   error_kw={"lw": 1.5, "ecolor": "dimgrey"})

    # y-axis: "Re-rank #N  feature_name"
    ytick_labels = [
        f"#{row.consolidated_rank}  {row.feature}"
        for _, row in rank_summary.iterrows()
    ]
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.invert_yaxis()

    # Annotate bars with avg ± SD
    for i, (avg, sd) in enumerate(
        zip(rank_summary["avg_rank"], rank_summary["std_rank"])
    ):
        ax.text(avg + sd + 0.15, i,
                f"avg {avg:.1f} ± {sd:.1f}", va="center", fontsize=7.5)

    ax.set_xlabel("Average Per-Model Importance Rank  (1 = most important)", fontsize=11)
    ax.set_title(
        "Consolidated Feature Importance Ranking Across All Models\n"
        "Y-axis = re-rank by avg rank  |  bar = avg rank  |  error = SD",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(n_feat / 2, color="grey", ls="--", lw=1, alpha=0.5,
               label=f"Midpoint (rank {n_feat / 2:.0f})")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, comp_dir / "feature_consolidated_rank_barplot")

    # ── Dot-strip chart: per-model rank dots + avg line per feature ────────
    fig, ax = plt.subplots(figsize=(11, max(5, n_feat * 0.48 + 2)))
    for i, (_, row) in enumerate(rank_summary.iterrows()):
        per_model_ranks = [
            rank_df.loc[lbl, row["feature"]]
            for lbl in labels
            if row["feature"] in rank_df.columns
        ]
        jitter = np.random.default_rng(i).uniform(-0.25, 0.25, len(per_model_ranks))
        ax.scatter(per_model_ranks, [i + j for j in jitter],
                   color=bar_colors[i], s=28, alpha=0.75, zorder=3)
        # Draw avg ± SD as a horizontal line + whiskers
        ax.plot([row["avg_rank"] - row["std_rank"],
                 row["avg_rank"] + row["std_rank"]],
                [i, i], color="black", lw=1.5, zorder=4)
        ax.scatter([row["avg_rank"]], [i],
                   color="black", s=55, zorder=5, marker="D")

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Per-Model Importance Rank  (1 = most important)", fontsize=11)
    ax.set_title(
        "Feature Rank Variability Across Models\n"
        "Dots = individual model ranks  |  ◆ = mean  |  line = ±SD",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(n_feat / 2, color="grey", ls="--", lw=1, alpha=0.4)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, comp_dir / "feature_rank_dotstrip")

    # ── 9. Consolidated feature MDI WEIGHT summary (avg, SD, re-rank) ─────
    avg_imp = imp_df.mean(axis=0)
    std_imp = imp_df.std(axis=0)

    # Re-rank: sort by avg_importance descending (highest avg MDI = re-rank 1)
    imp_summary = (
        pd.DataFrame({"feature": avg_imp.index,
                      "avg_importance": avg_imp.values,
                      "std_importance": std_imp.values})
        .sort_values("avg_importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_summary.insert(0, "consolidated_rank", range(1, len(imp_summary) + 1))

    # Attach each model's individual MDI value
    for lbl in labels:
        imp_summary[f"mdi_{lbl}"] = imp_summary["feature"].map(
            imp_df.loc[lbl].to_dict()
        )

    imp_summary.to_csv(comp_dir / "feature_consolidated_importance.tsv", sep="\t", index=False)

    # ── Plot: consolidated MDI bar chart with SD + re-rank annotation ─────
    bar_colors2 = plt.cm.YlOrRd(np.linspace(0.85, 0.15, n_feat))

    fig, ax = plt.subplots(figsize=(11, max(5, n_feat * 0.48 + 2)))
    ax.barh(range(n_feat), imp_summary["avg_importance"].values,
            xerr=imp_summary["std_importance"].values,
            color=bar_colors2, alpha=0.88, capsize=4,
            error_kw={"lw": 1.5, "ecolor": "dimgrey"})

    ytick_labels_imp = [
        f"#{row.consolidated_rank}  {row.feature}"
        for _, row in imp_summary.iterrows()
    ]
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(ytick_labels_imp, fontsize=9)
    ax.invert_yaxis()

    for i, (avg, sd) in enumerate(
        zip(imp_summary["avg_importance"], imp_summary["std_importance"])
    ):
        ax.text(avg + sd + 0.001, i,
                f"avg {avg:.3f} ± {sd:.3f}", va="center", fontsize=7.5)

    ax.set_xlabel("Average MDI Feature Importance", fontsize=11)
    ax.set_title(
        "Consolidated Feature Importance Weight (MDI) Across All Models\n"
        "Y-axis = re-rank by avg MDI  |  bar = avg MDI  |  error = SD",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, comp_dir / "feature_consolidated_importance_barplot")

    # ── Dot-strip chart: per-model MDI dots + avg line per feature ─────────
    fig, ax = plt.subplots(figsize=(11, max(5, n_feat * 0.48 + 2)))
    for i, (_, row) in enumerate(imp_summary.iterrows()):
        per_model_vals = [
            float(imp_df.loc[lbl, row["feature"]])
            for lbl in labels
            if row["feature"] in imp_df.columns
        ]
        jitter = np.random.default_rng(i + 100).uniform(-0.25, 0.25, len(per_model_vals))
        ax.scatter(per_model_vals, [i + j for j in jitter],
                   color=bar_colors2[i], s=28, alpha=0.75, zorder=3)
        ax.plot([row["avg_importance"] - row["std_importance"],
                 row["avg_importance"] + row["std_importance"]],
                [i, i], color="black", lw=1.5, zorder=4)
        ax.scatter([row["avg_importance"]], [i],
                   color="black", s=55, zorder=5, marker="D")

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(ytick_labels_imp, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Per-Model MDI Feature Importance", fontsize=11)
    ax.set_title(
        "Feature MDI Weight Variability Across Models\n"
        "Dots = individual model MDI  |  ◆ = mean  |  line = ±SD",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, comp_dir / "feature_importance_dotstrip")

    # ── Side-by-side rank vs MDI re-rank comparison table ──────────────────
    # Shows whether a feature's rank by avg-rank agrees with rank by avg-MDI
    rank_rerank  = rank_summary[["consolidated_rank", "feature"]].rename(
        columns={"consolidated_rank": "rank_by_avg_rank"}
    )
    imp_rerank   = imp_summary[["consolidated_rank", "feature"]].rename(
        columns={"consolidated_rank": "rank_by_avg_mdi"}
    )
    combined_rerank = rank_rerank.merge(imp_rerank, on="feature")
    combined_rerank["rank_agreement"] = (
        combined_rerank["rank_by_avg_rank"] == combined_rerank["rank_by_avg_mdi"]
    )
    combined_rerank["rank_diff"] = (
        combined_rerank["rank_by_avg_rank"] - combined_rerank["rank_by_avg_mdi"]
    ).abs()
    combined_rerank = combined_rerank.sort_values("rank_by_avg_rank").reset_index(drop=True)
    combined_rerank.to_csv(comp_dir / "feature_rerank_comparison.tsv", sep="\t", index=False)

    # ── 10. Summary table ──────────────────────────────────────────────────
    summary_rows = []
    for r in model_results:
        row = {
            "model_label":       r["label"],
            "training_dataset":  r.get("training_dataset", r["label"]),
            "n_training_genes":  r["n_genes"],
            "n_targets":         r["n_targets"],
            "cv_mean_auc":       r["cv_mean_auc"],
            "cv_std_auc":        r["cv_std_auc"],
        }
        mdf = r.get("metrics_df")
        if mdf is not None and len(mdf) > 0:
            for k in [10, 50, 100, 200]:
                sub = mdf[mdf["k"] == k]
                row[f"prec_top{k}"] = float(sub["precision"].iloc[0]) if len(sub) else float("nan")
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        comp_dir / "training_summary.tsv", sep="\t", index=False
    )
    print(f"\n  Cross-model comparison plots saved: {comp_dir}")


# ============================================================================
# Model Registry (multi-algorithm support)
# ============================================================================
#
# Each entry maps a CLI-friendly short name to a *factory* function that
# returns a fresh, untrained estimator. Linear models are wrapped in a
# Pipeline with StandardScaler because their coefficients are scale-sensitive
# and the feature-weight expansion makes columns highly collinear.
#
# Linear/logistic regression: classification by ranking the predicted score.
# AUC, average precision and top-K precision are all rank-based, so a plain
# regressor (LinearRegression) is fine for those metrics — its predictions
# are used directly as scores.
# ============================================================================


class _LinearRegressionScorer(LinearRegression):
    """LinearRegression with a sklearn-classifier-style score interface.

    Treats the regression output as the positive-class score. Adds:
      * `predict_proba(X)` returning a 2-column array (pseudo-probabilities)
        clipped to [0, 1] via min-max scaling on training scores;
      * `classes_` attribute so the rest of the pipeline doesn't choke.
    """

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        # Cache score range for later min-max scaling in predict_proba.
        scores = super().predict(X)
        self._score_min = float(np.min(scores))
        self._score_max = float(np.max(scores))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        scores = super().predict(X)
        rng = self._score_max - self._score_min
        if rng <= 0:
            p = np.full_like(scores, 0.5, dtype=float)
        else:
            p = (scores - self._score_min) / rng
            p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def _make_random_forest():
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10,
        min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1,
    )


def _make_decision_tree():
    # Single tree counterpart to RF — useful as an interpretability baseline
    # and for seeing how much ensembling actually buys you.
    return DecisionTreeClassifier(
        max_depth=10, min_samples_split=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42,
    )


def _make_naive_bayes():
    # GaussianNB scales each feature independently via its variance estimate,
    # so we don't wrap it in a StandardScaler.
    return GaussianNB()


def _make_knn():
    # Distance-based: must scale features. Distance-weighted voting helps when
    # neighbours are at very different distances (typical when classes overlap
    # in feature space but cluster locally near the decision boundary).
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance",
                                     n_jobs=-1)),
    ])


def _make_logistic():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced",
            solver="liblinear", random_state=42,
        )),
    ])


def _make_linear():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", _LinearRegressionScorer()),
    ])


def _make_svm():
    # RBF kernel SVM. probability=False so we use decision_function() for
    # scoring (much faster than enabling Platt scaling internally). All
    # downstream metrics are rank-based, so decision-function scores work
    # fine for ROC/PR/top-K/calibration.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=1.0, kernel="rbf", gamma="scale",
                    class_weight="balanced", random_state=42,
                    probability=False)),
    ])


def _make_gradient_boosting():
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )


def _make_xgboost():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42, n_jobs=-1,
        verbosity=0,
    )


def _make_lightgbm():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=-1, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1,
    )


# CLI short name -> (display label, factory, available?)
# All models are default-on. XGB/LGBM gracefully degrade if their libraries
# aren't installed (a warning is printed and they're skipped).
MODEL_REGISTRY: dict[str, tuple[str, callable, bool]] = {
    "rf":       ("Random Forest",        _make_random_forest,     True),
    "dt":       ("Decision Tree",        _make_decision_tree,     True),
    "nb":       ("Naive Bayes",          _make_naive_bayes,       True),
    "knn":      ("K-Nearest Neighbors",  _make_knn,               True),
    "logreg":   ("Logistic Regression",  _make_logistic,          True),
    "linreg":   ("Linear Regression",    _make_linear,            True),
    "svm":      ("SVM (RBF)",            _make_svm,               True),
    "gb":       ("Gradient Boosting",    _make_gradient_boosting, True),
    "xgb":      ("XGBoost",              _make_xgboost,           _XGB_AVAILABLE),
    "lgbm":     ("LightGBM",             _make_lightgbm,          _LGBM_AVAILABLE),
}


def resolve_model_selection(requested: list[str] | None,
                            dropped: list[str] | None) -> list[str]:
    """Resolve --models / --drop-models CLI args into a final list of model names.

    Args:
        requested: list of model names (or ['all']) — what the user asked for.
                   None means "all". 'all' may be combined with other names.
        dropped: list of model names to remove from the resolved set.

    Returns:
        Ordered list of model short-names that are both selected and available.
    """
    all_names = list(MODEL_REGISTRY.keys())

    if requested is None:
        selected = all_names[:]
    else:
        selected = []
        for name in requested:
            n = name.strip().lower()
            if n == "all":
                for d in all_names:
                    if d not in selected:
                        selected.append(d)
                continue
            if n not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model '{name}'. Valid: {', '.join(all_names)}"
                )
            if n not in selected:
                selected.append(n)

    if dropped:
        drop_set = {n.strip().lower() for n in dropped}
        unknown = drop_set - set(MODEL_REGISTRY.keys())
        if unknown:
            raise ValueError(
                f"Unknown model(s) in --drop-models: {sorted(unknown)}. "
                f"Valid: {', '.join(all_names)}"
            )
        selected = [n for n in selected if n not in drop_set]

    # Drop unavailable optional models, warning the user
    final = []
    for n in selected:
        _, _, available = MODEL_REGISTRY[n]
        if not available:
            print(f"  ! Skipping '{n}' ({MODEL_REGISTRY[n][0]}): library not installed")
            continue
        final.append(n)

    if not final:
        raise ValueError("No models selected after applying filters and availability checks.")
    return final


def _extract_importance(estimator, feature_names: list[str]) -> np.ndarray:
    """Pull a 1-D importance vector out of any supported model.

    Tree models use feature_importances_; linear models use |coef_|.
    Pipelines are unwrapped to their final step.
    """
    est = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator

    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_, dtype=float)
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float).ravel()
        return np.abs(coef)
    # Fallback: zeros (don't crash; just won't show on the heatmap)
    return np.zeros(len(feature_names), dtype=float)


def _get_scores(estimator, X) -> np.ndarray:
    """Return a 1-D positive-class score, regardless of model type."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def train_and_evaluate(
    features_df,
    targets_set,
    outdir,
    feature_weights=None,
    label="model",
    make_plots=True,
    model_name: str = "rf",
    save_as_rf_model: bool = True,
    X_test=None,
    y_test=None,
    bootstrap_n: int = 0,
    bootstrap_balanced: bool = False,
    bootstrap_random_state: int = 42,
):
    """Train one model, cross-validate, compute metrics.

    Returns (preds_df, importance_df, metrics_df, cv_info).

    Args:
        model_name: short name from MODEL_REGISTRY (default "rf"). Selects which
                    classifier/regressor to train.
        save_as_rf_model: if True, also save this model as 'rf_model.pkl' for
                    backward-compatibility with predict.py. The multi-model
                    orchestrator uses this only for the best-AUC model.
        X_test, y_test: optional held-out test set. When provided, the model is
                    additionally scored on this set and metrics are stored under
                    cv_info["test_metrics"].
        bootstrap_n: if > 0, after the standard test scoring, also compute
                    bootstrap mean and 95% CI for every test metric.
        bootstrap_balanced: if True, each bootstrap resample contains
                    min(n_pos, n_neg) positives and the same number of
                    negatives, both sampled with replacement. Use this when
                    the test set is class-imbalanced and the unbalanced metrics
                    are saturating (e.g. P@K hitting 1.0).
        bootstrap_random_state: seed for the bootstrap sampler.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         f"Valid: {list(MODEL_REGISTRY.keys())}")
    display_name, factory, available = MODEL_REGISTRY[model_name]
    if not available:
        raise RuntimeError(f"Model '{model_name}' ({display_name}) is unavailable "
                           f"(required library not installed).")

    features_df = features_df.copy()
    features_df["is_target"] = features_df["Name"].isin(targets_set).astype(int)

    n_targets     = int(features_df["is_target"].sum())
    n_non_targets = int((features_df["is_target"] == 0).sum())
    print(f"\n  [{display_name}] Labelled genes: {n_targets} targets, {n_non_targets} non-targets")

    feature_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
    X_orig     = features_df[feature_cols].fillna(0).values
    y          = features_df["is_target"].values
    gene_names = features_df["Name"].values

    X, expanded_cols, expansion_mapping = apply_feature_weights(
        X_orig, feature_cols, feature_weights
    )
    print(f"  [{display_name}] Training matrix: {X.shape[0]} genes x {X.shape[1]} features")

    # 5-fold CV
    print(f"  [{display_name}] 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores    = []
    cv_oof_probs = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        clf = factory()
        clf.fit(X[train_idx], y[train_idx])
        probs = _get_scores(clf, X[val_idx])
        cv_oof_probs[val_idx] = probs
        auc = roc_auc_score(y[val_idx], probs) if len(np.unique(y[val_idx])) == 2 else float("nan")
        cv_scores.append(auc)
        print(f"    [{display_name}] Fold {fold}: ROC-AUC = {auc:.4f}")

    mean_auc = float(np.nanmean(cv_scores))
    std_auc  = float(np.nanstd(cv_scores))
    print(f"    [{display_name}] Mean CV ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    # Final model on all data
    print(f"  [{display_name}] Training final model on all data...")
    final_clf = factory()
    final_clf.fit(X, y)
    final_probs = _get_scores(final_clf, X)

    model_filename = f"{model_name}_model.pkl"
    with open(outdir / model_filename, "wb") as f:
        pickle.dump(final_clf, f)
    print(f"  [{display_name}] Saved model: {outdir / model_filename}")

    # Backward-compat: also save as rf_model.pkl so predict.py works unchanged
    if save_as_rf_model:
        with open(outdir / "rf_model.pkl", "wb") as f:
            pickle.dump(final_clf, f)
        print(f"  [{display_name}] Also saved as rf_model.pkl (predict.py compatibility)")

    # Predictions
    predictions_df = pd.DataFrame({
        "Name":              gene_names,
        "is_target":         y,
        "cv_probability":    cv_oof_probs,
        "final_probability": final_probs,
    }).sort_values("final_probability", ascending=False)
    predictions_df["rank"] = range(1, len(predictions_df) + 1)
    pred_path = outdir / f"predictions_{model_name}.tsv"
    predictions_df.to_csv(pred_path, sep="\t", index=False)
    print(f"  [{display_name}] Saved predictions: {pred_path}")

    # Feature importance
    raw_importance = _extract_importance(final_clf, expanded_cols)
    importance_df = aggregate_importance(
        raw_importance, expanded_cols, expansion_mapping,
        all_original_cols=feature_cols,
    )
    imp_path = outdir / f"feature_importance_{model_name}.tsv"
    importance_df.to_csv(imp_path, sep="\t", index=False)
    print(f"  [{display_name}] Saved feature importance: {imp_path}")

    print(f"  [{display_name}] Top features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:30s}: {row['importance']:.4f}")

    # Top-K precision
    print(f"  [{display_name}] Top-K precision:")
    topk_rows = []
    for k in [10, 20, 50, 100, 150, 200, 250, 500, 1000]:
        if k > len(predictions_df):
            continue
        topk      = predictions_df.head(k)
        n_found   = int(topk["is_target"].sum())
        precision = n_found / k
        topk_rows.append({"k": k, "n_targets": n_found, "precision": precision})
        print(f"    Top-{k:4d}: {n_found:3d} targets ({precision * 100:5.1f}%)")
    metrics_df = pd.DataFrame(topk_rows)
    metrics_path = outdir / f"topk_metrics_{model_name}.tsv"
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    print(f"  [{display_name}] Saved metrics: {metrics_path}")

    if make_plots:
        # Per-model plots go into a subdirectory so multiple models don't collide
        per_model_dir = outdir / f"plots_{model_name}"
        per_model_dir.mkdir(parents=True, exist_ok=True)
        generate_per_model_plots(
            predictions_df=predictions_df,
            importance_df=importance_df,
            metrics_df=metrics_df,
            cv_scores=cv_scores,
            cv_oof_probs=cv_oof_probs,
            cv_oof_labels=y,
            final_probs=final_probs,
            outdir=per_model_dir,
            label=f"{label} | {display_name}",
        )

    # ── Held-out test set evaluation (optional) ───────────────────────────────
    test_metrics: dict = {}
    test_scores = None
    test_labels = None
    if X_test is not None and y_test is not None and len(y_test) > 0:
        test_probs = _get_scores(final_clf, X_test)
        test_scores = np.asarray(test_probs, dtype=float)
        test_labels = np.asarray(y_test)
        if len(np.unique(y_test)) == 2:
            test_metrics["auc"] = float(roc_auc_score(y_test, test_probs))
            test_metrics["average_precision"] = float(
                average_precision_score(y_test, test_probs)
            )
        else:
            test_metrics["auc"] = float("nan")
            test_metrics["average_precision"] = float("nan")
        # Top-K precision on the test set
        test_order = np.argsort(-test_probs)
        y_sorted = y_test[test_order]
        for k in (10, 20, 50, 100, 200):
            if k <= len(y_test):
                test_metrics[f"precision_at_{k}"] = float(np.mean(y_sorted[:k]))
            else:
                test_metrics[f"precision_at_{k}"] = float("nan")
        test_metrics["n_test"] = int(len(y_test))
        test_metrics["n_test_positives"] = int(np.sum(y_test))
        print(f"  [{display_name}] Held-out test: AUC={test_metrics['auc']:.4f}, "
              f"AP={test_metrics['average_precision']:.4f}, "
              f"P@10={test_metrics['precision_at_10']:.3f}, "
              f"n={test_metrics['n_test']} ({test_metrics['n_test_positives']} pos)")

        # ── Bootstrap test metrics (optional) ─────────────────────────────────
        # When --test-bootstrap N is set, draw N resamples of the test set,
        # recompute every metric on each, and store mean / 95% CI.
        # If bootstrap_balanced is True, each resample is a balanced
        # pos+neg subsample (sampled with replacement) — this fixes the
        # class-imbalance saturation problem where P@K ceilings out at 1.0
        # because there are too many positives.
        if bootstrap_n and bootstrap_n > 0:
            rng = np.random.default_rng(bootstrap_random_state)
            pos_idx = np.where(test_labels == 1)[0]
            neg_idx = np.where(test_labels == 0)[0]
            n_pos = len(pos_idx)
            n_neg = len(neg_idx)

            # Bootstrap requires both classes present; if not, skip
            if n_pos < 2 or n_neg < 2:
                print(f"  [{display_name}] Bootstrap skipped: "
                      f"need ≥2 of each class (got {n_pos} pos, {n_neg} neg)")
            else:
                if bootstrap_balanced:
                    # Each sample: k positives + k negatives, with replacement.
                    # k = min(n_pos, n_neg) so we use all of the smaller class.
                    sample_per_class = min(n_pos, n_neg)
                else:
                    # Standard bootstrap: same size as full test set, sampled
                    # with replacement (preserves the original class ratio).
                    pass

                # Per-metric arrays of bootstrap values
                metric_keys = [
                    "auc", "average_precision",
                    "precision_at_10", "precision_at_20", "precision_at_50",
                    "precision_at_100", "precision_at_200",
                ]
                boot: dict[str, list[float]] = {k: [] for k in metric_keys}

                for b in range(bootstrap_n):
                    if bootstrap_balanced:
                        # Sample equal counts from each class with replacement
                        b_pos = rng.choice(pos_idx, size=sample_per_class, replace=True)
                        b_neg = rng.choice(neg_idx, size=sample_per_class, replace=True)
                        sample_idx = np.concatenate([b_pos, b_neg])
                    else:
                        sample_idx = rng.choice(
                            len(test_labels), size=len(test_labels), replace=True,
                        )
                    s_labels = test_labels[sample_idx]
                    s_scores = test_scores[sample_idx]

                    # Both classes must be present in this sample
                    if len(np.unique(s_labels)) < 2:
                        continue

                    boot["auc"].append(float(roc_auc_score(s_labels, s_scores)))
                    boot["average_precision"].append(
                        float(average_precision_score(s_labels, s_scores))
                    )
                    s_order = np.argsort(-s_scores)
                    s_sorted = s_labels[s_order]
                    for k in (10, 20, 50, 100, 200):
                        if k <= len(s_labels):
                            boot[f"precision_at_{k}"].append(float(np.mean(s_sorted[:k])))

                # Summarise: mean + 95% CI
                for k in metric_keys:
                    vals = np.asarray(boot[k], dtype=float)
                    if len(vals) == 0:
                        continue
                    test_metrics[f"{k}_mean"] = float(np.mean(vals))
                    test_metrics[f"{k}_ci_lo"] = float(np.percentile(vals, 2.5))
                    test_metrics[f"{k}_ci_hi"] = float(np.percentile(vals, 97.5))
                test_metrics["bootstrap_n"] = int(bootstrap_n)
                test_metrics["bootstrap_balanced"] = bool(bootstrap_balanced)
                test_metrics["bootstrap_sample_per_class"] = (
                    int(sample_per_class) if bootstrap_balanced else 0
                )

                mode = "balanced" if bootstrap_balanced else "standard"
                print(f"  [{display_name}] Bootstrap ({mode}, n={bootstrap_n}): "
                      f"AUC={test_metrics['auc_mean']:.4f} "
                      f"[{test_metrics['auc_ci_lo']:.3f}, {test_metrics['auc_ci_hi']:.3f}], "
                      f"AP={test_metrics['average_precision_mean']:.4f}, "
                      f"P@10={test_metrics['precision_at_10_mean']:.3f}")

    cv_info = {
        "model_name":    model_name,
        "display_name":  display_name,
        "cv_scores":     cv_scores,
        "cv_mean_auc":   mean_auc,
        "cv_std_auc":    std_auc,
        "cv_oof_probs":  cv_oof_probs,
        "cv_oof_labels": y,
        "final_probs":   final_probs,
        "test_metrics":  test_metrics,
        # Held-out test set scores + labels for the score distribution panel
        "test_scores":   test_scores,
        "test_labels":   test_labels,
        # Exposed for downstream diagnostics
        "X":              X,
        "expanded_cols":  expanded_cols,
        "expansion_mapping": expansion_mapping,
        "feature_cols":   feature_cols,
    }
    return predictions_df, importance_df, metrics_df, cv_info


def train_all_models(
    features_df,
    targets_set,
    outdir,
    model_names: list[str],
    feature_weights=None,
    label: str = "model",
    make_plots: bool = True,
    test_fraction: float = 0.0,
    test_random_state: int = 42,
    bootstrap_n: int = 0,
    bootstrap_balanced: bool = False,
):
    """Train every model in model_names, then produce cross-algorithm comparison plots.

    A single stratified train/test split is performed once at the top so that
    every model is evaluated on the same held-out rows. Each model is trained
    only on the training portion (with 5-fold CV inside it for diagnostics),
    then scored on the held-out test set.

    The best model by CV mean AUC is also saved as rf_model.pkl so predict.py
    works unchanged.

    Args:
        test_fraction: fraction of rows to hold out for the test set
                       (default 0.20). Pass 0.0 to skip the split entirely.
        test_random_state: seed for the stratified split (default 42).
        bootstrap_n: if > 0, compute bootstrap mean and 95% CI for every
                       test metric (per-model).
        bootstrap_balanced: if True, each bootstrap resample uses balanced
                       pos/neg sampling — fixes metric saturation when the
                       test set is heavily class-imbalanced.

    Returns: list of dicts (one per model) with all per-model artefacts.
    """
    print(f"\n{'='*70}")
    print(f"MULTI-MODEL TRAINING ({len(model_names)} models)")
    print(f"{'='*70}")
    print("  Models:", ", ".join(MODEL_REGISTRY[n][0] for n in model_names))

    # ── Stratified train/test split (done once, shared across all models) ────
    df = features_df.copy()
    df["is_target"] = df["Name"].isin(targets_set).astype(int)
    n_pos = int(df["is_target"].sum())
    n_neg = int((df["is_target"] == 0).sum())

    train_df = df
    test_X = test_y = None
    if 0 < test_fraction < 1.0 and n_pos >= 5 and n_neg >= 5:
        feature_cols_all = [c for c in FEATURE_COLUMNS if c in df.columns]
        train_df, test_df = train_test_split(
            df,
            test_size=test_fraction,
            stratify=df["is_target"],
            random_state=test_random_state,
        )
        # Build the test design matrix once. Apply the same feature-weight
        # expansion as the training matrix so columns line up.
        X_test_orig = test_df[feature_cols_all].fillna(0).values
        test_X, _, _ = apply_feature_weights(
            X_test_orig, feature_cols_all, feature_weights
        )
        test_y = test_df["is_target"].values
        n_train = len(train_df)
        n_test = len(test_df)
        print(f"  Held-out test set: {n_test} rows "
              f"({int(test_y.sum())} positives), "
              f"training on remaining {n_train} rows")
        # Save the test split row IDs for reproducibility
        test_df[["Name", "is_target"]].to_csv(
            outdir / "test_set_genes.tsv", sep="\t", index=False
        )
    else:
        if test_fraction <= 0:
            print("  Skipping train/test split (test_fraction = 0)")
        else:
            print(f"  Skipping train/test split (insufficient data: "
                  f"{n_pos} pos, {n_neg} neg)")

    train_df_for_call = train_df.drop(columns=["is_target"])

    results = []
    for n in model_names:
        # Don't write rf_model.pkl yet — we'll do it once at the end for the best model
        preds, importance, metrics, cv_info = train_and_evaluate(
            train_df_for_call, targets_set, outdir,
            feature_weights=feature_weights,
            label=label,
            make_plots=make_plots,
            model_name=n,
            save_as_rf_model=False,
            X_test=test_X,
            y_test=test_y,
            bootstrap_n=bootstrap_n,
            bootstrap_balanced=bootstrap_balanced,
        )
        results.append({
            "model_name":    n,
            "display_name":  MODEL_REGISTRY[n][0],
            "predictions":   preds,
            "importance":    importance,
            "metrics":       metrics,
            "cv_info":       cv_info,
        })

    # ── Pick best by CV mean AUC and save as rf_model.pkl ────────────────────
    best = max(results, key=lambda r: r["cv_info"]["cv_mean_auc"])
    best_pkl = outdir / f"{best['model_name']}_model.pkl"
    with open(best_pkl, "rb") as src, open(outdir / "rf_model.pkl", "wb") as dst:
        dst.write(src.read())
    print(f"\n  Best model by CV AUC: {best['display_name']} "
          f"(AUC={best['cv_info']['cv_mean_auc']:.4f})")
    print(f"  Copied to rf_model.pkl for predict.py compatibility")

    # ── Cross-algorithm comparison plots & summary table ─────────────────────
    generate_algorithm_comparison_plots(results, outdir)

    # ── Always-on overfitting diagnostics (train vs CV) ──────────────────────
    generate_overfitting_diagnostics(results, outdir)

    # ── Combined publication-ready figure (always last) ──────────────────────
    generate_combined_publication_figure(results, outdir)

    return results


def generate_algorithm_comparison_plots(results: list[dict], outdir: Path):
    """Plot ROC, PR, AUC bars, top-K precision, and calibration across algorithms.

    Also writes a single TSV summary of headline metrics for all models.
    """
    if not results:
        return

    comp_dir = outdir / "model_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(results)
    palette  = plt.cm.tab10(np.linspace(0, 0.9, max(n_models, 3)))

    # ── 1. ROC overlay (out-of-fold CV) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, r in enumerate(results):
        y_true  = r["cv_info"]["cv_oof_labels"]
        y_score = r["cv_info"]["cv_oof_probs"]
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val     = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color=palette[i], lw=2,
                label=f"{r['display_name']} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Out-of-Fold Cross-Validation")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    _savefig(fig, comp_dir / "roc_curves")

    # ── 2. PR overlay ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, r in enumerate(results):
        y_true  = r["cv_info"]["cv_oof_labels"]
        y_score = r["cv_info"]["cv_oof_probs"]
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color=palette[i], lw=2,
                label=f"{r['display_name']} (AP = {ap:.3f})")
    pos_rate = float(np.mean(results[0]["cv_info"]["cv_oof_labels"]))
    ax.axhline(pos_rate, color="k", ls="--", lw=1, alpha=0.4,
               label=f"Baseline (positive rate = {pos_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Out-of-Fold CV")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    _savefig(fig, comp_dir / "pr_curves")

    # ── 3. AUC bar chart with CV error bars ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(7, 1.0 * n_models + 2), 6))
    names    = [r["display_name"] for r in results]
    means    = [r["cv_info"]["cv_mean_auc"] for r in results]
    stds     = [r["cv_info"]["cv_std_auc"]  for r in results]
    bars = ax.bar(range(n_models), means, yerr=stds, capsize=6,
                  color=[palette[i] for i in range(n_models)],
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Cross-Validated ROC-AUC")
    ax.set_title("Model Comparison — CV ROC-AUC (mean +/- std across folds)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, comp_dir / "auc_bar_chart")

    # ── 4. Top-K precision lines ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, r in enumerate(results):
        m = r["metrics"]
        if m.empty:
            continue
        ax.plot(m["k"], m["precision"], marker="o", color=palette[i], lw=2,
                label=r["display_name"])
    ax.set_xlabel("Top-K (genes)")
    ax.set_ylabel("Precision")
    ax.set_title("Top-K Precision — Final Model on All Data")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    _savefig(fig, comp_dir / "topk_precision")

    # ── 5. Calibration plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    n_bins = 10
    for i, r in enumerate(results):
        y_true  = r["cv_info"]["cv_oof_labels"]
        y_score = r["cv_info"]["cv_oof_probs"]
        # Min-max scale to [0,1] for fair calibration plotting across model types
        s_min, s_max = float(np.min(y_score)), float(np.max(y_score))
        if s_max > s_min:
            y_norm = (y_score - s_min) / (s_max - s_min)
        else:
            y_norm = np.full_like(y_score, 0.5, dtype=float)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.digitize(y_norm, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        means_pred, means_true = [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            means_pred.append(float(np.mean(y_norm[mask])))
            means_true.append(float(np.mean(y_true[mask])))
        ax.plot(means_pred, means_true, marker="o", color=palette[i], lw=2,
                label=r["display_name"])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Mean predicted score (min-max scaled)")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration — Out-of-Fold CV (scores rescaled per model)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _savefig(fig, comp_dir / "calibration")

    # ── 7. Summary metrics table ─────────────────────────────────────────────
    summary_rows = []
    for r in results:
        cv = r["cv_info"]
        y_true  = cv["cv_oof_labels"]
        y_score = cv["cv_oof_probs"]
        ap = (average_precision_score(y_true, y_score)
              if len(np.unique(y_true)) == 2 else float("nan"))
        m = r["metrics"].set_index("k")["precision"] if not r["metrics"].empty else pd.Series(dtype=float)
        tm = cv.get("test_metrics", {}) or {}
        summary_rows.append({
            "model":            r["model_name"],
            "display_name":     r["display_name"],
            "cv_mean_auc":      cv["cv_mean_auc"],
            "cv_std_auc":       cv["cv_std_auc"],
            "cv_average_precision": ap,
            "precision_at_10":  float(m.get(10,  np.nan)),
            "precision_at_50":  float(m.get(50,  np.nan)),
            "precision_at_100": float(m.get(100, np.nan)),
            "precision_at_200": float(m.get(200, np.nan)),
            "precision_at_500": float(m.get(500, np.nan)),
            # Held-out test-set metrics (NaN when no test split was made)
            "test_auc":              float(tm.get("auc", np.nan)),
            "test_average_precision": float(tm.get("average_precision", np.nan)),
            "test_precision_at_10":  float(tm.get("precision_at_10", np.nan)),
            "test_precision_at_50":  float(tm.get("precision_at_50", np.nan)),
            "test_precision_at_100": float(tm.get("precision_at_100", np.nan)),
            "test_n":                int(tm.get("n_test", 0)),
            "test_n_positives":      int(tm.get("n_test_positives", 0)),
            # Bootstrap mean and 95% CI per metric (NaN when bootstrap not run)
            "test_auc_bs_mean":               float(tm.get("auc_mean", np.nan)),
            "test_auc_bs_ci_lo":              float(tm.get("auc_ci_lo", np.nan)),
            "test_auc_bs_ci_hi":              float(tm.get("auc_ci_hi", np.nan)),
            "test_ap_bs_mean":                float(tm.get("average_precision_mean", np.nan)),
            "test_ap_bs_ci_lo":               float(tm.get("average_precision_ci_lo", np.nan)),
            "test_ap_bs_ci_hi":               float(tm.get("average_precision_ci_hi", np.nan)),
            "test_p_at_10_bs_mean":           float(tm.get("precision_at_10_mean", np.nan)),
            "test_p_at_10_bs_ci_lo":          float(tm.get("precision_at_10_ci_lo", np.nan)),
            "test_p_at_10_bs_ci_hi":          float(tm.get("precision_at_10_ci_hi", np.nan)),
            "test_p_at_50_bs_mean":           float(tm.get("precision_at_50_mean", np.nan)),
            "test_p_at_50_bs_ci_lo":          float(tm.get("precision_at_50_ci_lo", np.nan)),
            "test_p_at_50_bs_ci_hi":          float(tm.get("precision_at_50_ci_hi", np.nan)),
            "test_p_at_100_bs_mean":          float(tm.get("precision_at_100_mean", np.nan)),
            "test_p_at_100_bs_ci_lo":         float(tm.get("precision_at_100_ci_lo", np.nan)),
            "test_p_at_100_bs_ci_hi":         float(tm.get("precision_at_100_ci_hi", np.nan)),
            "bootstrap_n":                    int(tm.get("bootstrap_n", 0)),
            "bootstrap_balanced":             bool(tm.get("bootstrap_balanced", False)),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("cv_mean_auc", ascending=False)
    summary_path = comp_dir / "model_comparison_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\n  Saved model comparison summary: {summary_path}")
    print(f"  Saved comparison plots in:      {comp_dir}/")
    print("\n  Model ranking by CV ROC-AUC:")
    for _, row in summary_df.iterrows():
        test_auc = row.get("test_auc", float("nan"))
        test_str = f"  Test AUC={test_auc:.4f}" if not np.isnan(test_auc) else ""
        print(f"    {row['display_name']:25s}  "
              f"CV AUC={row['cv_mean_auc']:.4f} +/- {row['cv_std_auc']:.4f}  "
              f"AP={row['cv_average_precision']:.4f}  "
              f"P@100={row['precision_at_100']:.3f}"
              f"{test_str}")


# ============================================================================
# Training & Evaluation (single dataset)
# ============================================================================




# ============================================================================
# Overfitting Diagnostics
# ============================================================================
#
# CV AUC is well known to be optimistic: every fold is drawn from the same
# experiment, so models that exploit dataset-specific quirks (batch effects,
# library-prep biases, sequencing depth) score well even when they wouldn't
# generalise to a *different* experiment. The functions below expose the most
# informative within-dataset signals of overfitting:
#
#   1. Train vs CV AUC gap — a model with high train AUC and much lower CV AUC
#      is memorising. Run by default; cheap.
#   2. CV fold variance — high std across folds means the model is sensitive
#      to small data perturbations. Run by default.
#   3. Permutation test — refit on shuffled labels. The AUC under shuffled
#      labels is the model's ability to memorise noise. Optional, expensive.
#   4. Learning curves — train vs validation AUC at increasing training-set
#      sizes. Validation that plateaus far below train = overfitting. Optional.
#
# For the *real* generalisation question (will model X picked on dataset A
# work on dataset B?) use cross_dataset_heatmap.py with each saved model
# pickle (rf_model.pkl, xgb_model.pkl, …).
# ============================================================================

def generate_overfitting_diagnostics(results: list[dict], outdir: Path):
    """Plot train-vs-CV AUC gap and produce overfitting summary table.

    Reads existing model results — no extra training. Produces two artefacts
    in <outdir>/model_comparison/:
      * train_vs_cv_auc.{png,pdf} — bar chart with paired bars per model
      * overfitting_summary.tsv — flat table with all diagnostic metrics
    """
    if not results:
        return

    comp_dir = outdir / "model_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(results)
    palette  = plt.cm.tab10(np.linspace(0, 0.9, max(n_models, 3)))
    names    = [r["display_name"] for r in results]

    # Compute train AUC for each model from saved final-model probabilities
    rows = []
    for r in results:
        cv = r["cv_info"]
        y       = cv["cv_oof_labels"]
        train_p = cv["final_probs"]
        train_auc = (roc_auc_score(y, train_p)
                     if len(np.unique(y)) == 2 else float("nan"))
        gap = train_auc - cv["cv_mean_auc"]
        rows.append({
            "model":          r["model_name"],
            "display_name":   r["display_name"],
            "train_auc":      train_auc,
            "cv_mean_auc":    cv["cv_mean_auc"],
            "cv_std_auc":     cv["cv_std_auc"],
            "gap_train_cv":   gap,
            "fold_aucs":      list(cv["cv_scores"]),
        })

    summary_df = pd.DataFrame(rows).sort_values("cv_mean_auc", ascending=False)
    flat_df = summary_df.drop(columns=["fold_aucs"]).copy()

    # Add interpretation flag
    def _interpret(gap, std):
        if gap > 0.15:    return "OVERFIT"     # large memorisation
        if gap > 0.08:    return "moderate"
        if std > 0.10:    return "unstable"    # gap small but folds vary
        return "ok"
    flat_df["overfitting_flag"] = [
        _interpret(g, s) for g, s in zip(flat_df["gap_train_cv"], flat_df["cv_std_auc"])
    ]
    flat_df.to_csv(comp_dir / "overfitting_summary.tsv", sep="\t", index=False)

    # ── Plot 1: train AUC vs CV AUC paired bars ──────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * n_models + 2), 6))
    x = np.arange(n_models)
    w = 0.38
    train_aucs = [r["train_auc"]    for r in rows]
    cv_aucs    = [r["cv_mean_auc"]  for r in rows]
    cv_stds    = [r["cv_std_auc"]   for r in rows]
    ax.bar(x - w/2, train_aucs, w, label="Train AUC (in-sample)",
           color="lightcoral", edgecolor="black", linewidth=0.8)
    ax.bar(x + w/2, cv_aucs,    w, yerr=cv_stds, capsize=5,
           label="CV AUC (out-of-fold)",
           color="steelblue", edgecolor="black", linewidth=0.8)

    # Annotate the gap
    for i, r in enumerate(rows):
        gap = r["train_auc"] - r["cv_mean_auc"]
        flag = "OVERFIT" if gap > 0.15 else ("moderate" if gap > 0.08 else "ok")
        col = {"OVERFIT": "red", "moderate": "darkorange", "ok": "darkgreen"}[flag]
        y_pos = max(r["train_auc"], r["cv_mean_auc"]) + 0.03
        ax.text(i, y_pos, f"Δ={gap:.3f}", ha="center", va="bottom",
                fontsize=9, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([r["display_name"] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Train AUC vs CV AUC — Overfitting Gap\n"
                 "Large Δ (red) = model memorising training data",
                 fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="grey", lw=0.5, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, comp_dir / "train_vs_cv_auc")

    # ── Console digest ───────────────────────────────────────────────────────
    print(f"\n  Overfitting diagnostics → {comp_dir}/overfitting_summary.tsv")
    print(f"\n  {'Model':<22} {'Train AUC':>10} {'CV AUC':>10} {'Gap':>8} {'CV std':>8}  Flag")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8} {'-'*8}  {'-'*10}")
    for _, r in flat_df.iterrows():
        flag = r["overfitting_flag"]
        print(f"  {r['display_name']:<22} {r['train_auc']:>10.4f} "
              f"{r['cv_mean_auc']:>10.4f} {r['gap_train_cv']:>+8.3f} "
              f"{r['cv_std_auc']:>8.3f}  {flag}")


# ============================================================================
# Combined Publication Figure
# ============================================================================
#
# Single multi-panel figure that complements the individual model_comparison/*
# files in one publication-quality SVG (also written as PDF and PNG@300dpi).
#
# Layout (8 panels):
#   Row 1: ROC | PR | Top-K precision
#   Row 2: Calibration | CV-vs-Test gap | Test metrics heatmap
#   Row 3: Train/CV/Test AUC comparison (full width)
#   Row 4: Score distributions per model (full width — small multiples)
#
# Rationale: panels are non-redundant. Generic CV AUC bars are folded into
# row 3 (which shows Train/CV/Test side by side); fold stability, train-vs-CV
# overfitting, and feature importance were dropped in favour of more useful
# diagnostics. Test metrics are shown as a heatmap in row 2, with the same
# numbers also written to model_comparison_summary.tsv as a separate file.
# ============================================================================


def _add_panel_label(ax, letter: str, x: float = -0.12, y: float = 1.05):
    """Bold panel label (a, b, c, ...) in the upper-left of an axes."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="right")


def generate_combined_publication_figure(results: list[dict], outdir: Path):
    """Produce a single multi-panel publication-ready figure (SVG + PDF + PNG).

    Layout (8 panels, 4 rows):
      Row 1: ROC | PR | Top-K precision
      Row 2: Calibration | CV-vs-Test gap | Test metrics heatmap
      Row 3: Train/CV/Test AUC comparison (full width)
      Row 4: Score distributions per model on held-out test (full width, small multiples)
    """
    if not results:
        return

    comp_dir = outdir / "model_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(results)
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(n_models, 3)))
    names = [r["display_name"] for r in results]

    # Detect whether any model has test metrics (consistent across all models
    # since they share the same train/test split)
    has_test = any(
        not np.isnan(r["cv_info"].get("test_metrics", {}).get("auc", float("nan")))
        for r in results
    )

    # ── Figure layout ────────────────────────────────────────────────────────
    # Total height tuned for 10 models; row 4 is taller because it holds
    # n_models small multiples laid out as a single horizontal strip.
    fig_h = 5 + 5 + 5.5 + 5  # rows 1-4
    fig = plt.figure(figsize=(18, fig_h))
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[1, 1, 1.2, 1.15],
        hspace=0.65, wspace=0.32,
    )

    rc_saved = {
        "font.size":       11,
        "axes.titlesize":  12,
        "axes.labelsize":  11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
    with plt.rc_context(rc_saved):
        panel = iter("abcdefghij")

        # ── Panel a: ROC curves (CV out-of-fold) ─────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        for i, r in enumerate(results):
            y_true = r["cv_info"]["cv_oof_labels"]
            y_score = r["cv_info"]["cv_oof_probs"]
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = roc_auc_score(y_true, y_score)
            ax.plot(fpr, tpr, color=palette[i], lw=2,
                    label=f"{r['display_name']} ({auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC (out-of-fold CV)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        _add_panel_label(ax, next(panel))

        # ── Panel b: PR curves (CV out-of-fold) ──────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        pos_rate = float(np.mean(results[0]["cv_info"]["cv_oof_labels"]))
        for i, r in enumerate(results):
            y_true = r["cv_info"]["cv_oof_labels"]
            y_score = r["cv_info"]["cv_oof_probs"]
            if len(np.unique(y_true)) < 2:
                continue
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            ax.plot(recall, precision, color=palette[i], lw=2,
                    label=f"{r['display_name']} ({ap:.3f})")
        ax.axhline(pos_rate, color="k", ls="--", lw=0.8, alpha=0.4,
                   label=f"Baseline ({pos_rate:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall (out-of-fold CV)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        _add_panel_label(ax, next(panel))

        # ── Panel c: Top-K precision lines ───────────────────────────────────
        ax = fig.add_subplot(gs[0, 2])
        for i, r in enumerate(results):
            m = r["metrics"]
            if m.empty:
                continue
            ax.plot(m["k"], m["precision"], marker="o", color=palette[i], lw=2,
                    label=r["display_name"])
        ax.set_xlabel("Top-K (genes)")
        ax.set_ylabel("Precision")
        ax.set_title("Top-K Precision")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        _add_panel_label(ax, next(panel))

        # ── Panel d: Calibration ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        n_bins = 10
        for i, r in enumerate(results):
            y_true = r["cv_info"]["cv_oof_labels"]
            y_score = r["cv_info"]["cv_oof_probs"]
            s_min, s_max = float(np.min(y_score)), float(np.max(y_score))
            y_norm = ((y_score - s_min) / (s_max - s_min)
                      if s_max > s_min
                      else np.full_like(y_score, 0.5, dtype=float))
            bins = np.linspace(0, 1, n_bins + 1)
            bin_idx = np.clip(np.digitize(y_norm, bins) - 1, 0, n_bins - 1)
            mp, mt = [], []
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.sum() == 0:
                    continue
                mp.append(float(np.mean(y_norm[mask])))
                mt.append(float(np.mean(y_true[mask])))
            ax.plot(mp, mt, marker="o", color=palette[i], lw=2,
                    label=r["display_name"])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Perfect")
        ax.set_xlabel("Mean predicted score (rescaled)")
        ax.set_ylabel("Observed positive fraction")
        ax.set_title("Calibration (out-of-fold)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        _add_panel_label(ax, next(panel))

        # ── Panel e: CV-vs-Test gap ──────────────────────────────────────────
        # Positive gap = CV AUC overestimates real-world performance (overfit
        # in the CV-folds-being-correlated sense). Negative gap = honest.
        ax = fig.add_subplot(gs[1, 1])
        if has_test:
            cv_aucs = [r["cv_info"]["cv_mean_auc"] for r in results]
            # Use bootstrap mean if available, else point estimate
            test_aucs = []
            any_bs = False
            for r in results:
                tm = r["cv_info"].get("test_metrics", {}) or {}
                if not np.isnan(tm.get("auc_mean", np.nan)):
                    test_aucs.append(float(tm["auc_mean"]))
                    any_bs = True
                else:
                    test_aucs.append(float(tm.get("auc", np.nan)))
            gaps = [cv - t for cv, t in zip(cv_aucs, test_aucs)]
            order = np.argsort(gaps)[::-1]  # most-overfit on the left
            sorted_names = [names[i] for i in order]
            sorted_gaps = [gaps[i] for i in order]
            colors = []
            for g in sorted_gaps:
                if g > 0.10:
                    colors.append("#d62728")     # red — substantial overfit
                elif g > 0.05:
                    colors.append("#ff7f0e")     # orange — moderate
                elif g > -0.05:
                    colors.append("#2ca02c")     # green — honest
                else:
                    colors.append("#1f77b4")     # blue — test better than CV
            xpos = np.arange(n_models)
            ax.bar(xpos, sorted_gaps, color=colors,
                   edgecolor="black", linewidth=0.7)
            ax.axhline(0, color="black", lw=0.8)
            for i, g in enumerate(sorted_gaps):
                offset = 0.012 if g >= 0 else -0.012
                va = "bottom" if g >= 0 else "top"
                ax.text(i, g + offset, f"{g:+.3f}",
                        ha="center", va=va, fontsize=8, fontweight="bold")
            ax.set_xticks(xpos)
            ax.set_xticklabels(sorted_names, rotation=22, ha="right")
            ax.set_ylabel("CV AUC − Test AUC")
            title = "CV-vs-Test gap (generalization)\n"
            if any_bs:
                title += "Test = bootstrap mean; +red = overfit, −blue = honest"
            else:
                title += "Positive (red) = CV overestimates; Negative (blue) = honest"
            ax.set_title(title, fontsize=10)
            max_abs = max(0.15, max(abs(g) for g in sorted_gaps) * 1.3)
            ax.set_ylim(-max_abs, max_abs)
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5,
                    "No test set\n(--test-fraction 0)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="grey", style="italic", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        _add_panel_label(ax, next(panel))

        # ── Panel f: Test metrics heatmap ────────────────────────────────────
        # Replaces the embedded table. Same numbers, more readable, color-coded.
        # CSV is in model_comparison_summary.tsv.
        ax = fig.add_subplot(gs[1, 2])
        if has_test:
            metric_keys = [
                ("auc",                  "AUC"),
                ("average_precision",    "AP"),
                ("precision_at_10",      "P@10"),
                ("precision_at_50",      "P@50"),
                ("precision_at_100",     "P@100"),
            ]

            # Detect whether bootstrap was run — switch to bootstrap means if so
            any_bs = any(
                not np.isnan(r["cv_info"].get("test_metrics", {})
                             .get("auc_mean", np.nan))
                for r in results
            )

            def _val(tm, key):
                """Prefer bootstrap mean, fall back to point estimate."""
                if any_bs and not np.isnan(tm.get(f"{key}_mean", np.nan)):
                    return float(tm[f"{key}_mean"])
                return float(tm.get(key, np.nan))

            # Sort models by AUC (use bootstrap mean if available)
            test_aucs_for_sort = [
                _val(r["cv_info"].get("test_metrics", {}) or {}, "auc")
                for r in results
            ]
            test_aucs_for_sort = [
                v if not np.isnan(v) else -np.inf for v in test_aucs_for_sort
            ]
            order = np.argsort(test_aucs_for_sort)[::-1]
            sorted_names = [names[i] for i in order]
            heatmap_rows = []
            for i in order:
                tm = results[i]["cv_info"].get("test_metrics", {}) or {}
                heatmap_rows.append([_val(tm, k) for k, _ in metric_keys])
            heatmap_df = pd.DataFrame(
                heatmap_rows, index=sorted_names,
                columns=[lbl for _, lbl in metric_keys],
            )
            sns.heatmap(heatmap_df, annot=True, fmt=".3f",
                        cmap="YlGn", linewidths=0.4, ax=ax,
                        cbar=False, vmin=0, vmax=1)
            title = ("Held-out test metrics (bootstrap mean)\n(rows sorted by AUC)"
                     if any_bs
                     else "Held-out test metrics\n(rows sorted by AUC)")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel("")
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=0, fontsize=10)
        else:
            ax.text(0.5, 0.5,
                    "No test set\n(--test-fraction 0)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="grey", style="italic", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        _add_panel_label(ax, next(panel))

        # ── Panel g: Train / CV / Test AUC comparison (full width, row 3) ────
        ax = fig.add_subplot(gs[2, :])
        x = np.arange(n_models)
        cv_means = [r["cv_info"]["cv_mean_auc"] for r in results]
        cv_stds = [r["cv_info"]["cv_std_auc"] for r in results]
        train_aucs = []
        for r in results:
            y = r["cv_info"]["cv_oof_labels"]
            train_p = r["cv_info"]["final_probs"]
            train_aucs.append(roc_auc_score(y, train_p)
                              if len(np.unique(y)) == 2 else float("nan"))

        # Use bootstrap mean if available, else point estimate
        any_bs = any(
            not np.isnan(r["cv_info"].get("test_metrics", {}).get("auc_mean", np.nan))
            for r in results
        )
        test_central = []
        test_err_lo = []
        test_err_hi = []
        for r in results:
            tm = r["cv_info"].get("test_metrics", {}) or {}
            if any_bs and not np.isnan(tm.get("auc_mean", np.nan)):
                m = float(tm["auc_mean"])
                test_central.append(m)
                test_err_lo.append(m - float(tm["auc_ci_lo"]))
                test_err_hi.append(float(tm["auc_ci_hi"]) - m)
            else:
                test_central.append(float(tm.get("auc", np.nan)))
                test_err_lo.append(0.0)
                test_err_hi.append(0.0)
        test_yerr = np.array([test_err_lo, test_err_hi])

        ww = 0.27
        ax.bar(x - ww, train_aucs, ww, label="Train (in-sample)",
               color="lightcoral", edgecolor="black", linewidth=0.7)
        ax.bar(x, cv_means, ww, yerr=cv_stds, capsize=4,
               label="CV (out-of-fold)",
               color="steelblue", edgecolor="black", linewidth=0.7)
        test_label = ("Test (95% CI)" if any_bs
                      else ("Test (held-out)" if has_test else "Test (n/a)"))
        ax.bar(x + ww, test_central, ww,
               yerr=test_yerr if any_bs else None,
               capsize=4 if any_bs else 0,
               label=test_label,
               color="seagreen", edgecolor="black", linewidth=0.7)
        # Annotate test AUC on top of the test bar (use upper CI if bootstrap)
        for i, t in enumerate(test_central):
            if not np.isnan(t):
                top = t + (test_err_hi[i] if any_bs else 0)
                ax.text(i + ww, top + 0.020, f"{t:.3f}",
                        ha="center", va="bottom", fontsize=8,
                        color="darkgreen", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=22, ha="right")
        ax.set_ylabel("ROC-AUC")
        title_suffix = ("Test AUC is the most honest single number — "
                        "models scored on rows the model never saw"
                        if has_test
                        else "(Test bar absent: --test-fraction 0)")
        if any_bs:
            balanced_str = ""
            tm0 = results[0]["cv_info"].get("test_metrics", {})
            if tm0.get("bootstrap_balanced"):
                balanced_str = ", balanced sampling"
            title_suffix += f"\n(Test bars: bootstrap mean ± 95% CI, n={tm0.get('bootstrap_n', '?')}{balanced_str})"
        ax.set_title(f"Train vs CV vs Held-Out Test\n{title_suffix}",
                     fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.legend(loc="lower center", fontsize=9, ncol=3)
        ax.grid(axis="y", alpha=0.3)
        _add_panel_label(ax, next(panel))

        # ── Panel h: Score distributions on held-out test (full-width row 4) ─
        # Small multiples: one mini-axes per model. Two overlapping histograms
        # (positive vs negative class) of the model's scores on the held-out
        # set. A cleanly bimodal distribution = real signal; heavy overlap =
        # AUC may be inflated by a few outliers.
        if has_test:
            # Two-row sub-gridspec: a thin row for the section title, then
            # the strip of small multiples below it. This avoids hand-tuning
            # absolute fig coordinates for the title placement.
            sub_gs_outer = gs[3, :].subgridspec(2, 1, height_ratios=[0.18, 1.0],
                                                hspace=0.30)
            ax_title = fig.add_subplot(sub_gs_outer[0, 0])
            ax_title.text(0.5, 0.5,
                          "Held-out test score distributions — "
                          "separation between positives (color) and negatives (grey)",
                          ha="center", va="center", transform=ax_title.transAxes,
                          fontsize=11, fontweight="bold")
            ax_title.set_xticks([])
            ax_title.set_yticks([])
            for spine in ax_title.spines.values():
                spine.set_visible(False)

            sub_gs = sub_gs_outer[1, 0].subgridspec(1, n_models, wspace=0.32)
            for i, r in enumerate(results):
                cv_info = r["cv_info"]
                tm = cv_info.get("test_metrics", {})
                if not tm or np.isnan(tm.get("auc", np.nan)):
                    continue
                test_scores = cv_info.get("test_scores")
                test_labels = cv_info.get("test_labels")
                if test_scores is None or test_labels is None:
                    continue
                ax_sub = fig.add_subplot(sub_gs[0, i])
                pos_scores = test_scores[test_labels == 1]
                neg_scores = test_scores[test_labels == 0]
                # Min-max rescale per-model so all panels share an x-axis [0,1]
                lo, hi = float(np.min(test_scores)), float(np.max(test_scores))
                rng = hi - lo if hi > lo else 1.0
                pos_n = (pos_scores - lo) / rng
                neg_n = (neg_scores - lo) / rng
                bins = np.linspace(0, 1, 16)
                ax_sub.hist(neg_n, bins=bins, color="lightgrey",
                            edgecolor="grey", linewidth=0.5,
                            label="Neg", alpha=0.85)
                ax_sub.hist(pos_n, bins=bins, color=palette[i],
                            edgecolor="black", linewidth=0.5,
                            label="Pos", alpha=0.7)
                ax_sub.set_title(r["display_name"], fontsize=8.5, pad=2)
                ax_sub.tick_params(axis="both", labelsize=7)
                ax_sub.set_xlim(0, 1)
                ax_sub.set_xlabel("Score", fontsize=7.5)
                if i == 0:
                    ax_sub.set_ylabel("Count", fontsize=7.5)
                    ax_sub.legend(fontsize=6, loc="upper center", framealpha=0.7)
                if i == 0:
                    _add_panel_label(ax_sub, next(panel), x=-0.30, y=1.18)
        else:
            ax = fig.add_subplot(gs[3, :])
            ax.text(0.5, 0.5,
                    "No test set — score distributions unavailable\n"
                    "(re-run with --test-fraction > 0)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="grey", style="italic", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            _add_panel_label(ax, next(panel))

    # Save in three formats. SVG is the publication-quality vector format.
    for ext in ("svg", "pdf", "png"):
        path = comp_dir / f"figure_publication.{ext}"
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(path, format=ext, bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"\n  Saved publication figure: "
          f"{comp_dir}/figure_publication.{{svg,pdf,png}}")



# ============================================================================
# Batch LOO Training
# ============================================================================

def run_batch_loo_training(
    meta_df: pd.DataFrame,
    outdir: Path,
    normalization: str,
    editing_type: str,
    min_edit_pct: float,
    min_reads: int,
    feature_weights: dict | None = None,
    bg_filter_model: object = None,
    feature_set: str = "full22",
    min_total_reads: int = 0,
    min_fold_change: float = 0.0,
):
    """For each dataset D, train on D alone; save to outdir/Trained_<label>/."""
    active_cols = get_feature_columns(feature_set)
    print(f"\n{'='*70}")
    print("BATCH TRAINING")
    print(f"{'='*70}")
    print(f"  Datasets     : {len(meta_df)}")
    print(f"  Feature set  : {feature_set} ({len(active_cols)} features)")
    print(f"  Strategy     : Train one model per dataset; test on all others via cross_dataset_heatmap.py")
    has_per_dataset_bg = "background_files" in meta_df.columns

    # Pre-load all datasets
    print("\nPre-loading all datasets...")
    all_features: dict[str, pd.DataFrame] = {}
    all_targets:  dict[str, set]          = {}
    for _, row in meta_df.iterrows():
        lbl = row["label"]
        print(f"  Loading: {lbl}")
        try:
            # Per-dataset background filter takes priority over CLI-level one
            row_bg_filter = bg_filter_model
            if has_per_dataset_bg and pd.notna(row.get("background_files", None)):
                bg_paths = parse_files(str(row["background_files"]))
                if bg_paths:
                    print(f"    Learning per-dataset background filter ({len(bg_paths)} file(s))...")
                    bg_dfs = [load_editing_sites(Path(p), editing_type=editing_type,
                                                 min_edit_pct=0.0, min_reads=0)
                              for p in bg_paths]
                    row_bg_filter = learn_background_filter(bg_dfs)

            feat_df, targets_set = load_dataset_features(
                row, normalization, editing_type, min_edit_pct, min_reads,
                bg_filter_model=row_bg_filter,
                feature_set=feature_set,
                min_total_reads=min_total_reads,
                min_fold_change=min_fold_change,
            )
            all_features[lbl] = feat_df
            all_targets[lbl]  = targets_set
            print(f"    {len(feat_df):,} genes,  {len(targets_set)} targets")
        except Exception as e:
            print(f"    ERROR: {e}")

    loaded_labels = [lbl for lbl in meta_df["label"].tolist() if lbl in all_features]
    model_results = []

    for train_label in loaded_labels:
        print(f"\n{'─'*60}")
        print(f"  Training on : '{train_label}'")
        test_labels = [lbl for lbl in loaded_labels if lbl != train_label]
        print(f"  Will test on: {test_labels}")

        # Train on this single dataset only
        train_df = all_features[train_label].copy()
        train_df["is_target"] = train_df["Name"].isin(all_targets[train_label]).astype(int)
        train_df["_dataset"]  = train_label
        combined_df = train_df

        n_genes   = len(combined_df)
        n_targets = int(combined_df["is_target"].sum())
        print(f"  Training data: {n_genes:,} genes, {n_targets} targets")

        # Rename holdout_label -> train_label throughout the rest of the block
        holdout_label = train_label

        loo_dir = outdir / f"Trained_{train_label}"
        loo_dir.mkdir(parents=True, exist_ok=True)

        X_orig = combined_df[active_cols].fillna(0).values
        y      = combined_df["is_target"].values
        X, expanded_cols, expansion_mapping = apply_feature_weights(
            X_orig, active_cols, feature_weights
        )

        # 5-fold CV
        print("\n  5-fold cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores    = []
        cv_oof_probs = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1,
            )
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[val_idx])[:, 1]
            cv_oof_probs[val_idx] = probs
            auc = (roc_auc_score(y[val_idx], probs)
                   if len(np.unique(y[val_idx])) == 2 else float("nan"))
            cv_scores.append(auc)
            print(f"    Fold {fold}: AUC = {auc:.4f}")

        mean_auc = float(np.nanmean(cv_scores))
        std_auc  = float(np.nanstd(cv_scores))
        print(f"    Mean CV ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

        # Final model on full training set
        print("  Training final model...")
        final_clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1,
        )
        final_clf.fit(X, y)
        final_probs = final_clf.predict_proba(X)[:, 1]

        with open(loo_dir / "rf_model.pkl", "wb") as f:
            pickle.dump(final_clf, f)

        # Feature importance
        importance_df = aggregate_importance(
            final_clf.feature_importances_, expanded_cols, expansion_mapping,
            all_original_cols=active_cols,
        )
        importance_df.to_csv(loo_dir / "feature_importance.tsv", sep="\t", index=False)

        # Save combined training features
        combined_df.to_csv(loo_dir / "gene_features_combined.tsv", sep="\t", index=False)

        # Predictions on training data (for top-K)
        preds_df = pd.DataFrame({
            "Name":              combined_df["Name"].values,
            "is_target":         y,
            "cv_probability":    cv_oof_probs,
            "final_probability": final_probs,
        }).sort_values("final_probability", ascending=False)
        preds_df["rank"] = range(1, len(preds_df) + 1)
        preds_df.to_csv(loo_dir / "predictions_train.tsv", sep="\t", index=False)

        topk_rows = []
        for k in [10, 20, 50, 100, 150, 200, 250, 500, 1000]:
            if k > len(preds_df):
                continue
            n_found   = int(preds_df.head(k)["is_target"].sum())
            topk_rows.append({"k": k, "n_targets": n_found, "precision": n_found / k})
        metrics_df = pd.DataFrame(topk_rows)
        metrics_df.to_csv(loo_dir / "topk_metrics.tsv", sep="\t", index=False)

        # Save config
        config = {
            "training_dataset":      train_label,
            "test_datasets":         test_labels,
            "normalization":         normalization,
            "editing_type":          editing_type,
            "min_edit_pct":          min_edit_pct,
            "min_reads":             min_reads,
            "min_total_reads":       min_total_reads,
            "min_fold_change":       min_fold_change,
            "n_genes":               n_genes,
            "n_targets":             n_targets,
            "feature_columns":       active_cols,
            "feature_set":           feature_set,
            "feature_weights":       feature_weights,
            "cv_mean_auc":           mean_auc,
            "cv_std_auc":            std_auc,
            "has_background_filter": bg_filter_model is not None,
        }
        with open(loo_dir / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save background filter alongside this model if one was used
        if bg_filter_model is not None:
            save_background_filter(bg_filter_model, loo_dir / "background_filter.pkl")

        # Per-model plots
        generate_per_model_plots(
            predictions_df=preds_df,
            importance_df=importance_df,
            metrics_df=metrics_df,
            cv_scores=cv_scores,
            cv_oof_probs=cv_oof_probs,
            cv_oof_labels=y,
            final_probs=final_probs,
            outdir=loo_dir,
            label=f"Trained_{train_label}",
        )

        model_results.append({
            "label":            f"Trained_{train_label}",
            "training_dataset": train_label,
            "n_genes":          n_genes,
            "n_targets":        n_targets,
            "cv_scores":        cv_scores,
            "cv_mean_auc":      mean_auc,
            "cv_std_auc":       std_auc,
            "importance_df":    importance_df,
            "metrics_df":       metrics_df,
        })
        print(f"  Saved LOO model: {loo_dir}")

    # Cross-model comparison plots
    print(f"\n{'─'*60}")
    print("  Generating cross-model comparison plots...")
    generate_model_comparison_plots(model_results, outdir)

    print(f"\n{'='*70}")
    print("BATCH TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  {len(model_results)} models saved in: {outdir}")
    print(f"  Comparison plots:  {outdir / 'comparison'}")
    print(f"\nNext step — cross-dataset testing:")
    print(f"  python cross_dataset_heatmap.py \\")
    print(f"      --metadata <your_metadata.tsv> \\")
    print(f"      --loo-dir {outdir} \\")
    print(f"      --outdir <cross_test_outdir>")
    print(f"  Each LOO model will be tested on all datasets EXCEPT the one it was trained on.")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train RF model for RBP target prediction (18-feature schema)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--expt", nargs="+",
        help="Single mode: experiment replicate file(s)",
    )
    mode.add_argument(
        "--metadata", type=str,
        help="Batch mode: metadata TSV with label, expt_files, ctrl_files, targets_file "
             "(bg_files accepted as alias for ctrl_files for backward compatibility)",
    )

    parser.add_argument("--ctrl", nargs="+",
                        help="Single mode: control replicate file(s) for fold-change "
                             "(was --bg in previous versions)")
    parser.add_argument("--targets", type=Path,
                        help="Single mode: known target gene names file")
    parser.add_argument(
        "--background", nargs="+", default=None,
        help="Optional: background control file(s) (e.g. gDNA, no-enzyme control). "
             "Sites matching the background noise pattern will be filtered before "
             "feature extraction. The learned filter is saved as background_filter.pkl "
             "and auto-applied during testing. Can also be specified per-dataset in "
             "the metadata TSV as a 'background_files' column.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--normalization",
                        choices=["simple", "median_of_ratios"], default="simple")
    parser.add_argument("--min-edit-pct", type=float, default=0.0)
    parser.add_argument("--min-reads", type=int, default=20)
    parser.add_argument(
        "--min-total-reads", type=int, default=0,
        help="Gene-level filter: minimum average total reads across experiment "
             "replicates after aggregation. Genes below this threshold are "
             "excluded from features/training/prediction. Default 0 = off. "
             "Recommended: 20.",
    )
    parser.add_argument(
        "--min-fold-change", type=float, default=0.0,
        help="Gene-level filter: minimum fold change (avg cumulative expt edit%% / "
             "avg cumulative ctrl edit%%) after aggregation. Genes below this "
             "threshold are excluded from features/training/prediction. "
             "Default 0.0 = off. Use 1.0 to keep only genes enriched over control.",
    )
    parser.add_argument("--editing-type", choices=["AtoG", "CtoT"], default="AtoG")
    parser.add_argument(
        "--feature-set",
        choices=VALID_FEATURE_SETS,
        default="full18",
        help=(
            "Feature set to use for training (default: full18 — all 18 features). "
            "Legacy names full12 / full24 / full22 / reduced17 / core8 are accepted "
            "and map to the same 18-feature schema for backward compatibility."
        ),
    )
    parser.add_argument(
        "--feature-weights", nargs="*", default=None,
        help="Per-feature weight overrides as name:weight pairs. "
             "Example: --feature-weights fold_change:3 site_enrichment:0.5",
    )
    parser.add_argument("--feature-weights-file", type=str, default=None)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=(
            "Which models to train (default: all). Pass 'all' or any subset of: "
            + ", ".join(MODEL_REGISTRY.keys()) + ". "
            "XGBoost ('xgb') and LightGBM ('lgbm') are skipped silently if their "
            "libraries are not installed."
        ),
    )
    parser.add_argument(
        "--drop-models", nargs="+", default=None,
        help="Models to remove from the selection (applied after --models). "
             "Example: --drop-models linreg svm",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.0,
        help="Fraction of rows to hold out as a test set for the per-model "
             "test panel of the publication figure. Default: 0.20 (stratified "
             "by label). Pass 0 to skip the split entirely (test panel will "
             "show NaN values).",
    )
    parser.add_argument(
        "--test-bootstrap", type=int, default=0, metavar="N",
        help="If > 0, bootstrap-resample the test set N times and report "
             "mean + 95%% CI for every test metric. Default: 0 (off). "
             "Recommended N: 100-1000.",
    )
    parser.add_argument(
        "--test-bootstrap-balance", action="store_true",
        help="When bootstrapping, draw balanced pos/neg samples on each "
             "iteration. Use this when the test set is heavily imbalanced "
             "(e.g. positive rate > 30%%) and metrics like P@K are saturating "
             "at 1.0. Has no effect unless --test-bootstrap is also set.",
    )
    parser.add_argument("--log", type=str, nargs="?", const="Logs", default=None)

    args = parser.parse_args()

    if args.expt:
        if not args.ctrl:
            parser.error("--ctrl is required with --expt (control replicates for fold-change)")
        if not args.targets:
            parser.error("--targets is required with --expt")

    log_path = setup_logging(args.log, prefix="train")

    # Resolve which models to train
    try:
        selected_models = resolve_model_selection(args.models, args.drop_models)
    except ValueError as e:
        parser.error(str(e))
    print(f"\n  Models to train   : {', '.join(MODEL_REGISTRY[n][0] for n in selected_models)}")

    # Feature weights
    feature_weights = None
    if args.feature_weights_file:
        feature_weights = load_feature_weights_file(args.feature_weights_file)
        print(f"  Loaded feature weights from: {args.feature_weights_file}")
    elif args.feature_weights:
        feature_weights = {}
        for item in args.feature_weights:
            if ":" not in item:
                parser.error(f"Invalid --feature-weights format: '{item}'")
            name, w = item.rsplit(":", 1)
            try:
                feature_weights[name.strip()] = float(w.strip())
            except ValueError:
                parser.error(f"Invalid weight in '{item}'.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RBP TARGET PREDICTION - TRAINING")
    print("=" * 70)
    print(f"  Mode              : {'Batch' if args.metadata else 'Single dataset'}")
    print(f"  Feature set       : {args.feature_set} ({len(get_feature_columns(args.feature_set))} features)")
    print(f"  Normalization     : {args.normalization}")
    print(f"  Editing type      : {args.editing_type}")
    print(f"  Min edit %%        : {args.min_edit_pct}")
    print(f"  Min reads         : {args.min_reads}")
    print(f"  Min total reads   : {args.min_total_reads}  (gene-level, post-aggregation)")
    print(f"  Min fold change   : {args.min_fold_change}  (gene-level, post-aggregation)")
    if feature_weights:
        print(f"  Feature weights   : {feature_weights}")

    # ── Batch mode ────────────────────────────────────────────────────────
    if args.metadata:
        meta_df = load_metadata(Path(args.metadata))

        # Learn background filter from CLI --background files if provided
        # (applies same filter to all datasets in batch)
        bg_filter_model = None
        if args.background:
            print("\nLearning background filter from CLI --background files...")
            bg_dfs = []
            for p in args.background:
                print(f"  {p}")
                df = load_editing_sites(Path(p), editing_type=args.editing_type,
                                        min_edit_pct=0.0, min_reads=0)
                bg_dfs.append(df)
            bg_filter_model = learn_background_filter(bg_dfs)
            save_background_filter(bg_filter_model, args.outdir / "background_filter.pkl")

        run_batch_loo_training(
            meta_df=meta_df,
            outdir=args.outdir,
            normalization=args.normalization,
            editing_type=args.editing_type,
            min_edit_pct=args.min_edit_pct,
            min_reads=args.min_reads,
            feature_weights=feature_weights,
            bg_filter_model=bg_filter_model,
            feature_set=args.feature_set,
            min_total_reads=args.min_total_reads,
            min_fold_change=args.min_fold_change,
        )

    # ── Single dataset mode ───────────────────────────────────────────────
    else:
        print(f"\n  Experiment replicates : {len(args.expt)}")
        print(f"  Control replicates    : {len(args.ctrl)}")
        if args.background:
            print(f"  Background files      : {len(args.background)}")

        print("\nLoading data...")
        expt_dfs = []
        for p in args.expt:
            print(f"  [expt] {p}")
            df = load_editing_sites(Path(p), editing_type=args.editing_type,
                                    min_edit_pct=args.min_edit_pct,
                                    min_reads=args.min_reads)
            print(f"    {len(df):,} sites")
            expt_dfs.append(df)

        ctrl_dfs = []
        for p in args.ctrl:
            print(f"  [ctrl] {p}")
            df = load_editing_sites(Path(p), editing_type=args.editing_type,
                                    min_edit_pct=args.min_edit_pct,
                                    min_reads=args.min_reads)
            print(f"    {len(df):,} sites")
            ctrl_dfs.append(df)

        # Optional background filter
        bg_filter_model = None
        if args.background:
            print("\nLearning background filter...")
            bg_raw_dfs = []
            for p in args.background:
                print(f"  [bg] {p}")
                df = load_editing_sites(Path(p), editing_type=args.editing_type,
                                        min_edit_pct=0.0, min_reads=0)
                print(f"    {len(df):,} sites")
                bg_raw_dfs.append(df)
            bg_filter_model = learn_background_filter(bg_raw_dfs)
            save_background_filter(bg_filter_model, args.outdir / "background_filter.pkl")

        targets_raw = args.targets.read_text().strip().splitlines()
        targets_set = {t.strip().lower() for t in targets_raw if t.strip()}
        print(f"\n  Known targets: {len(targets_set)}")

        gene_names = sorted(
            set().union(*(df["Name"].unique() for df in expt_dfs + ctrl_dfs))
        )
        print(f"  Unique genes:  {len(gene_names):,}")

        print("\nBuilding features...")
        features_df = build_features(expt_dfs, ctrl_dfs, gene_names,
                                     normalization=args.normalization,
                                     bg_filter_model=bg_filter_model,
                                     feature_set=args.feature_set,
                                     min_total_reads=args.min_total_reads,
                                     min_fold_change=args.min_fold_change)
        features_df.to_csv(args.outdir / "gene_features.tsv", sep="\t", index=False)
        print(f"  Saved features: {args.outdir / 'gene_features.tsv'}")

        active_cols = get_feature_columns(args.feature_set)
        config = {
            "normalization":         args.normalization,
            "editing_type":          args.editing_type,
            "min_edit_pct":          args.min_edit_pct,
            "min_reads":             args.min_reads,
            "min_total_reads":       args.min_total_reads,
            "min_fold_change":       args.min_fold_change,
            "n_expt":                len(args.expt),
            "n_ctrl":                len(args.ctrl),
            "n_genes":               len(gene_names),
            "n_targets":             len(targets_set),
            "feature_columns":       active_cols,
            "feature_set":           args.feature_set,
            "feature_weights":       feature_weights,
            "has_background_filter": bg_filter_model is not None,
        }
        with open(args.outdir / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved config: {args.outdir / 'model_config.json'}")

        print("\nTraining models...")
        train_all_models(
            features_df, targets_set, args.outdir,
            model_names=selected_models,
            feature_weights=feature_weights,
            label=args.outdir.name,
            make_plots=True,
            test_fraction=args.test_fraction,
            bootstrap_n=args.test_bootstrap,
            bootstrap_balanced=args.test_bootstrap_balance,
        )

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  All outputs in: {args.outdir}")
    if log_path:
        print(f"  Log file:       {log_path}")
    print(f"  Finished:       {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
