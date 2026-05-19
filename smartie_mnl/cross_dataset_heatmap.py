"""
====================================================================
Cross-Dataset Testing — One or Many Models vs All Datasets
====================================================================

THREE operating modes:

  1. Single model  (--model path/rf_model.pkl)
     Test ONE model on every dataset in the metadata file.
     Generates precision heatmaps, line plots, ROC/PR curves,
     score distributions, feature-value heatmap, and gene-rank
     overlap heatmap.

  2. Batch auto-discovery  (--loo-dir path/)
     Automatically finds all Trained_*/rf_model.pkl models saved by
     train_model.py --metadata.  For each model trained on dataset D,
     tests it on ALL other datasets (skips D itself).
     Generates per-model plots AND multi-model comparison plots.

  3. Explicit multi-model  (--models path1 path2 ...)
     Same as LOO mode but you supply model paths manually.
     Optional --model-labels overrides inferred names.

Metadata file format (TSV, tab-separated, header required):
    label       expt_files                      ctrl_files                      targets_file        background_files
    Ataxin-2    data/Atx_1.txt;data/Atx_2.txt   data/Adar_1.txt;data/Adar_2.txt targets/atx.txt     gDNA.txt
    Thor        data/Thor_1.txt;Thor_2.txt       data/BG_1.txt;BG_2.txt          targets/thor.txt

  - ctrl_files: control replicates for fold-change (bg_files also accepted)
  - targets_file: CSV or text file of known target genes (for precision eval)
  - background_files: optional — per-dataset background for site-level noise filtering

Usage examples:
    # Single model
    python cross_dataset_heatmap.py \\
        --metadata datasets.tsv \\
        --model outputs/training/rf_model.pkl \\
        --outdir outputs/cross_test

    # Batch auto-discovery (typical after train_model.py --metadata)
    python cross_dataset_heatmap.py \\
        --metadata datasets.tsv \\
        --loo-dir outputs/batch_training \\
        --outdir outputs/cross_test

    # Explicit multi-model
    python cross_dataset_heatmap.py \\
        --metadata datasets.tsv \\
        --models outputs/batch_training/Trained_Ataxin/rf_model.pkl \\
                 outputs/batch_training/Trained_Adar/rf_model.pkl \\
        --outdir outputs/cross_test

Outputs (single model):
    <outdir>/<dataset>/predictions.tsv
    <outdir>/<dataset>/gene_features.tsv
    <outdir>/comparison/precision_heatmap.pdf/png
    <outdir>/comparison/precision_lineplot.pdf/png
    <outdir>/comparison/roc_curves.pdf/png
    <outdir>/comparison/roc_auc_barchart.pdf/png
    <outdir>/comparison/pr_curves.pdf/png
    <outdir>/comparison/score_distributions.pdf/png
    <outdir>/comparison/feature_value_heatmap.pdf/png
    <outdir>/comparison/gene_rank_overlap.pdf/png
    <outdir>/comparison/summary.tsv

Outputs (multi-model / LOO, additionally):
    <outdir>/comparison/multi_model_precision_heatmap.pdf/png
    <outdir>/comparison/multi_model_roc_heatmap.pdf/png
    <outdir>/comparison/roc_pr_<dataset>.pdf/png   (per dataset)
    <outdir>/comparison/multi_model_auc_comparison.pdf/png
    <outdir>/comparison/multi_model_feature_importance_heatmap.pdf/png
    <outdir>/comparison/multi_model_summary.tsv
"""

import argparse
import json
import math
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
_CIVIDIS_LIGHT = LinearSegmentedColormap.from_list("cividis_light", __import__("matplotlib.pyplot", fromlist=["cm"]).cm.cividis_r(__import__("numpy").linspace(0.0, 0.85, 256)))
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

warnings.filterwarnings("ignore")

from feature_extraction import load_editing_sites, build_features, FEATURE_COLUMNS, \
    VALID_FEATURE_SETS, get_feature_columns, \
    learn_background_filter, load_background_filter
from train_model import apply_feature_weights, setup_logging


# ============================================================================
# METADATA PARSING
# ============================================================================

def load_metadata(path: Path) -> pd.DataFrame:
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                first = line
                break
        else:
            raise ValueError("Metadata file has no non-comment lines")
    sep = "\t" if "\t" in first else ","
    df = pd.read_csv(path, sep=sep, comment="#", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {}
    if "dataset_name" in df.columns and "label" not in df.columns:
        rename_map["dataset_name"] = "label"
    # Accept bg_files as alias for ctrl_files (backward compat)
    if "bg_files" in df.columns and "ctrl_files" not in df.columns:
        rename_map["bg_files"] = "ctrl_files"
    if "validation_targets" in df.columns and "targets_file" not in df.columns:
        rename_map["validation_targets"] = "targets_file"
    if rename_map:
        df = df.rename(columns=rename_map)
    required = {"label", "expt_files", "ctrl_files", "targets_file"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df.dropna(subset=["label", "expt_files", "ctrl_files", "targets_file"]).reset_index(drop=True)


def parse_files(s: str) -> list[str]:
    return [f.strip() for f in s.split(";") if f.strip()]


def load_targets_set(targets_file: str) -> set[str]:
    p = Path(targets_file)
    try:
        tdf = pd.read_csv(p)
        name_col = "Name" if "Name" in tdf.columns else tdf.columns[0]
        return {
            g.strip().lower() for g in tdf[name_col].dropna().astype(str)
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


# ============================================================================
# SAVE FIGURE HELPER
# ============================================================================

def _savefig(fig, path_stem: Path):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# PREDICTION FOR ONE DATASET
# ============================================================================

def predict_on_dataset(
    rf_model,
    dataset_label:       str,
    ds_row:              pd.Series,
    normalization:       str,
    editing_type:        str,
    min_edit_pct:        float,
    min_reads:           int,
    outdir:              Path,
    feature_weights:     dict | None = None,
    bg_filter_model:     object = None,
    config_feature_set:  str = "full22",
) -> Path | None:
    ds_dir     = outdir / dataset_label
    expt_files = parse_files(ds_row["expt_files"])
    ctrl_files = parse_files(ds_row["ctrl_files"])

    try:
        ds_dir.mkdir(parents=True, exist_ok=True)

        expt_dfs = [
            load_editing_sites(Path(p), editing_type=editing_type,
                               min_edit_pct=min_edit_pct, min_reads=min_reads)
            for p in expt_files
        ]
        ctrl_dfs = [
            load_editing_sites(Path(p), editing_type=editing_type,
                               min_edit_pct=min_edit_pct, min_reads=min_reads)
            for p in ctrl_files
        ]
        gene_names = sorted(
            set().union(*(df["Name"].unique() for df in expt_dfs + ctrl_dfs))
        )
        features_df = build_features(expt_dfs, ctrl_dfs, gene_names,
                                     normalization=normalization,
                                     bg_filter_model=bg_filter_model,
                                     feature_set=config_feature_set)
        features_df.to_csv(ds_dir / "gene_features.tsv", sep="\t", index=False)

        active_cols = get_feature_columns(config_feature_set)
        X = features_df[active_cols].fillna(0).values
        X, _, _ = apply_feature_weights(X, active_cols, feature_weights)
        probs = rf_model.predict_proba(X)[:, 1]

        results_df = pd.DataFrame({
            "gene":           features_df["Name"].values,
            "rf_probability": probs,
        }).sort_values("rf_probability", ascending=False)
        results_df["rank"] = range(1, len(results_df) + 1)
        results_df.to_csv(ds_dir / "predictions.tsv", sep="\t", index=False)

        print(f"  [{dataset_label}]  {len(results_df)} genes  "
              f"top1={results_df.iloc[0]['gene']} "
              f"(p={results_df.iloc[0]['rf_probability']:.4f})")
        return ds_dir

    except Exception as e:
        print(f"  [{dataset_label}]  ERROR: {e}")
        return None


# ============================================================================
# PRECISION AT TOP-N% OF PUBLISHED TARGETS
# ============================================================================

def compute_pct_recovery(pred_path: Path, targets_file: str) -> pd.DataFrame | None:
    if not pred_path.exists():
        return None
    pred_df = pd.read_csv(pred_path, sep="\t")
    if "rf_probability" not in pred_df.columns:
        return None
    pub_set = load_targets_set(targets_file)
    N = len(pub_set)
    if N == 0:
        return None

    df = pred_df[pred_df["rf_probability"] >= 0.01].copy()
    gene_col = "gene" if "gene" in df.columns else "Name"
    df = (
        df.sort_values("rank", ascending=True)
        .drop_duplicates(subset=[gene_col], keep="first")
        .reset_index(drop=True)
    )
    df["is_target"] = df[gene_col].str.lower().isin(pub_set).astype(int)
    y = df["is_target"].values

    rows = []
    for p in range(10, 101, 10):
        cutoff = max(1, math.ceil(N * p / 100))
        actual = min(cutoff, len(y))
        found  = int(y[:actual].sum())
        rows.append({
            "top_percent":   p,
            "top_n":         actual,
            "total_targets": N,
            "targets_found": found,
            "precision":     found / actual if actual > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def compute_roc_pr(pred_path: Path, targets_file: str) -> dict | None:
    """Compute ROC-AUC and PR-AUC for a dataset (requires targets)."""
    if not pred_path.exists():
        return None
    pred_df = pd.read_csv(pred_path, sep="\t")
    if "rf_probability" not in pred_df.columns:
        return None
    pub_set = load_targets_set(targets_file)
    if not pub_set:
        return None

    gene_col = "gene" if "gene" in pred_df.columns else "Name"
    pred_df["is_target"] = pred_df[gene_col].str.lower().isin(pub_set).astype(int)
    y = pred_df["is_target"].values
    scores = pred_df["rf_probability"].values

    if len(np.unique(y)) < 2:
        return None

    fpr, tpr, _   = roc_curve(y, scores)
    prec, rec, _  = precision_recall_curve(y, scores)
    roc_auc = roc_auc_score(y, scores)
    pr_auc  = average_precision_score(y, scores)
    baseline = y.mean()

    return {
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "prec": prec, "rec": rec, "pr_auc": pr_auc,
        "baseline": baseline,
        "scores": scores, "labels": y,
    }


def collect_all_results(outdir, dataset_labels, meta_df):
    rows = []
    for dl in dataset_labels:
        pred_path = outdir / dl / "predictions.tsv"
        ds_row    = meta_df[meta_df["label"] == dl].iloc[0]
        targets   = ds_row["targets_file"].strip()
        rec_df    = compute_pct_recovery(pred_path, targets)
        if rec_df is None:
            continue
        for _, row in rec_df.iterrows():
            rows.append({
                "dataset":       dl,
                "top_percent":   int(row["top_percent"]),
                "precision":     float(row["precision"]),
                "targets_found": int(row["targets_found"]),
                "total_targets": int(row["total_targets"]),
            })
    return pd.DataFrame(rows)


# ============================================================================
# SINGLE-MODEL PLOTS
# ============================================================================

def generate_single_model_plots(
    results_df: pd.DataFrame,
    outdir: Path,
    dataset_labels: list[str],
    meta_df: pd.DataFrame,
    train_label: str,
    model_outdir: Path | None = None,
):
    """Generate all comparison plots for one model tested on multiple datasets."""
    comp_dir = (model_outdir or outdir) / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    pcts   = sorted(results_df["top_percent"].unique())
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(dataset_labels), 1)))

    # Collect ROC/PR data for datasets that have targets
    roc_pr_data = {}
    for dl in dataset_labels:
        pred_path  = outdir / dl / "predictions.tsv"
        ds_row     = meta_df[meta_df["label"] == dl].iloc[0]
        targets    = ds_row["targets_file"].strip()
        rp = compute_roc_pr(pred_path, targets)
        if rp:
            roc_pr_data[dl] = rp

    # ── 1. Precision heatmap ───────────────────────────────────────────────
    hmap_rows, hmap_idx = [], []
    for dl in dataset_labels:
        sub = results_df[results_df["dataset"] == dl].set_index("top_percent")
        if sub.empty:
            continue
        n_pub = int(sub["total_targets"].iloc[0])
        hmap_rows.append([float(sub["precision"].get(p, 0)) for p in pcts])
        hmap_idx.append(f"{dl}  (N={n_pub})")

    if hmap_rows:
        hmap_df = pd.DataFrame(hmap_rows, index=hmap_idx,
                               columns=[f"{p}%" for p in pcts])
        hmap_df.to_csv(comp_dir / "precision_heatmap_data.tsv", sep="\t")
        # Transpose: prediction depth on Y-axis (rows), datasets on X-axis (columns)
        hmap_df_T = hmap_df.T
        fig, ax = plt.subplots(figsize=(max(9, len(pcts) * 0.95 + 3),
                                        max(3, len(hmap_rows) * 0.75 + 1.5)))
        sns.heatmap(hmap_df * 100, annot=True, fmt=".0f", cmap=_CIVIDIS_LIGHT,
                    vmin=0, vmax=100, linewidths=0.8, linecolor="black", ax=ax,
                    cbar_kws={"label": "Precision %"})
        ax.set_title(f"Prediction Precision Across Datasets\n(model: {train_label})",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Prediction depth (% of published targets)")
        ax.set_ylabel("Test dataset  (N = published targets)")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        _savefig(fig, comp_dir / "precision_heatmap")

    # ── 2. Precision line plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, dl in enumerate(dataset_labels):
        sub = results_df[results_df["dataset"] == dl]
        if sub.empty:
            continue
        avg = sub.set_index("top_percent")["precision"]
        ax.plot([f"{p}%" for p in avg.index], avg.values * 100,
                marker="o", lw=2, color=colors[i % len(colors)], label=dl)
    ax.set_xlabel("Prediction depth (% of published targets)", fontsize=12)
    ax.set_ylabel("Precision %", fontsize=12)
    ax.set_title(f"Precision by Prediction Depth\n(model: {train_label})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, comp_dir / "precision_lineplot")

    # ── 3. Per-dataset precision bar chart at 10% ──────────────────────────
    summary_rows = []
    for dl in dataset_labels:
        sub = results_df[results_df["dataset"] == dl]
        if sub.empty:
            continue
        row = {"dataset": dl, "total_targets": int(sub["total_targets"].iloc[0])}
        for p in [10, 20, 50, 100]:
            s2 = sub[sub["top_percent"] == p]
            row[f"prec_{p}pct"]  = float(s2["precision"].iloc[0]) if len(s2) else float("nan")
            row[f"found_{p}pct"] = int(s2["targets_found"].iloc[0]) if len(s2) else 0
        # Add ROC/PR AUC if available
        if dl in roc_pr_data:
            row["roc_auc"] = roc_pr_data[dl]["roc_auc"]
            row["pr_auc"]  = roc_pr_data[dl]["pr_auc"]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("prec_10pct", ascending=False).reset_index(drop=True)
    summary_df.to_csv(comp_dir / "summary.tsv", sep="\t", index=False)

    if len(summary_df) > 0:
        bar_colors = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(summary_df)))
        fig, ax = plt.subplots(figsize=(10, max(3, len(summary_df) * 0.7 + 1.5)))
        vals = summary_df["prec_10pct"].values * 100
        bars = ax.barh(range(len(summary_df)), vals, color=bar_colors, alpha=0.9)
        ax.set_yticks(range(len(summary_df)))
        ax.set_yticklabels(
            [f"{dl}  (N={n})" for dl, n in
             zip(summary_df["dataset"], summary_df["total_targets"])],
            fontsize=9,
        )
        ax.invert_yaxis()
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if not np.isnan(val):
                ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("Precision @ 10% of published targets", fontsize=12)
        ax.set_title(f"Cross-Dataset Performance\n(model: {train_label})",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(0, 115)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        _savefig(fig, comp_dir / "dataset_precision_barchart")

    # ── 4. ROC curves (all datasets on same axes) ──────────────────────────
    if roc_pr_data:
        fig, ax = plt.subplots(figsize=(8, 7))
        for i, (dl, rp) in enumerate(roc_pr_data.items()):
            ax.plot(rp["fpr"], rp["tpr"], lw=2, color=colors[i % len(colors)],
                    label=f"{dl}  (AUC={rp['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curves — {train_label}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _savefig(fig, comp_dir / "roc_curves")

        # ── 5. ROC-AUC bar chart ───────────────────────────────────────────
        auc_labels = list(roc_pr_data.keys())
        auc_vals   = [roc_pr_data[dl]["roc_auc"] for dl in auc_labels]
        sorted_idx = np.argsort(auc_vals)[::-1]
        fig, ax = plt.subplots(figsize=(10, max(4, len(auc_labels) * 0.65 + 2)))
        bar_c = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(auc_labels)))
        bars = ax.barh(range(len(auc_labels)),
                       [auc_vals[i] for i in sorted_idx], color=bar_c, alpha=0.9)
        ax.set_yticks(range(len(auc_labels)))
        ax.set_yticklabels([auc_labels[i] for i in sorted_idx], fontsize=9)
        ax.invert_yaxis()
        for i, v in enumerate([auc_vals[j] for j in sorted_idx]):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlabel("ROC-AUC", fontsize=12)
        ax.set_title(f"ROC-AUC by Dataset — {train_label}",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.axvline(0.5, color="grey", ls="--", lw=1, alpha=0.6)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        _savefig(fig, comp_dir / "roc_auc_barchart")

        # ── 6. PR curves ───────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 7))
        for i, (dl, rp) in enumerate(roc_pr_data.items()):
            ax.plot(rp["rec"], rp["prec"], lw=2, color=colors[i % len(colors)],
                    label=f"{dl}  (AP={rp['pr_auc']:.3f})")
            ax.axhline(rp["baseline"], color=colors[i % len(colors)],
                       ls=":", lw=1, alpha=0.4)
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision–Recall Curves — {train_label}",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _savefig(fig, comp_dir / "pr_curves")

        # ── 7. Score distributions (violin per dataset) ────────────────────
        fig, ax = plt.subplots(figsize=(max(8, len(roc_pr_data) * 1.3 + 2), 5))
        vp_data   = []
        vp_labels = []
        for dl, rp in roc_pr_data.items():
            vp_data.append(rp["scores"])
            vp_labels.append(dl)
        vp = ax.violinplot(vp_data, positions=range(1, len(vp_data) + 1),
                           showmedians=True, showextrema=True)
        for i, (body, color) in enumerate(zip(vp["bodies"], colors)):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        ax.set_xticks(range(1, len(vp_labels) + 1))
        ax.set_xticklabels(vp_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("RF Prediction Score", fontsize=11)
        ax.set_title(f"Score Distributions Across Datasets — {train_label}",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        _savefig(fig, comp_dir / "score_distributions")

    # ── 8. Feature value heatmap (mean Z-score per dataset) ───────────────
    feat_means = {}
    for dl in dataset_labels:
        feat_path = outdir / dl / "gene_features.tsv"
        if not feat_path.exists():
            continue
        fdf = pd.read_csv(feat_path, sep="\t")
        feat_cols = [c for c in FEATURE_COLUMNS if c in fdf.columns]
        if feat_cols:
            feat_means[dl] = fdf[feat_cols].mean()

    if len(feat_means) >= 2:
        feat_df = pd.DataFrame(feat_means).T
        # Z-score normalise columns
        feat_z = (feat_df - feat_df.mean()) / (feat_df.std() + 1e-12)
        fig, ax = plt.subplots(
            figsize=(max(12, len(feat_z.columns) * 0.85 + 3),
                     max(4, len(feat_z) * 0.75 + 2))
        )
        sns.heatmap(feat_z, cmap="PuOr", center=0,
                    annot=True, fmt=".2f", linewidths=0.8, linecolor="black", ax=ax,
                    cbar_kws={"label": "Z-score of mean feature value"})
        ax.set_title(f"Mean Feature Values Across Datasets (Z-scored)\n(model: {train_label})",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Feature", fontsize=11)
        ax.set_ylabel("Dataset", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        _savefig(fig, comp_dir / "feature_value_heatmap")

    # ── 9. Gene rank overlap (Spearman r of gene scores between datasets) ──
    score_vecs = {}
    for dl in dataset_labels:
        pred_path = outdir / dl / "predictions.tsv"
        if pred_path.exists():
            pdf = pd.read_csv(pred_path, sep="\t")
            gene_col = "gene" if "gene" in pdf.columns else "Name"
            score_vecs[dl] = pdf.set_index(gene_col)["rf_probability"]

    if len(score_vecs) >= 2:
        keys = list(score_vecs.keys())
        n = len(keys)
        spearman_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                common = score_vecs[keys[i]].index.intersection(score_vecs[keys[j]].index)
                if len(common) >= 5:
                    r, _ = spearmanr(score_vecs[keys[i]][common],
                                     score_vecs[keys[j]][common])
                else:
                    r = float("nan")
                spearman_matrix[i, j] = r
                spearman_matrix[j, i] = r

        mask = np.zeros_like(spearman_matrix, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        fig, ax = plt.subplots(figsize=(max(5, n * 0.9 + 2), max(5, n * 0.9 + 2)))
        sns.heatmap(spearman_matrix, annot=True, fmt=".2f", cmap="PuOr",
                    vmin=-1, vmax=1, mask=mask,
                    xticklabels=keys, yticklabels=keys,
                    linewidths=0.8, linecolor="black", ax=ax,
                    cbar_kws={"label": "Spearman r"})
        ax.set_title(f"Gene Score Correlation Across Datasets (Spearman r)\n(model: {train_label})",
                     fontsize=13, fontweight="bold")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        _savefig(fig, comp_dir / "gene_rank_overlap")

    print(f"  Single-model plots saved: {comp_dir}")
    return summary_df if summary_rows else pd.DataFrame()


# ============================================================================
# MULTI-MODEL PLOTS
# ============================================================================

def generate_multi_model_plots(
    model_summaries: list[dict],
    outdir: Path,
    meta_df: pd.DataFrame,
):
    """
    Generate cross-model comparison plots.

    Each entry in model_summaries:
        model_label, model_path, holdout_dataset (or None),
        dataset_results (dict: dataset_label -> {roc_auc, pr_auc, prec_10, prec_20, prec_50}),
        importance_df (optional),
    """
    comp_dir = outdir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    model_labels  = [m["model_label"] for m in model_summaries]
    all_ds_labels = list(meta_df["label"])
    n_models      = len(model_labels)
    n_datasets    = len(all_ds_labels)

    palette_models   = plt.cm.tab10(np.linspace(0, 0.9, max(n_models, 1)))
    palette_datasets = plt.cm.Set2(np.linspace(0, 0.9,  max(n_datasets, 1)))

    # ── 1. Multi-model precision heatmap — full 10%–100% (models × datasets) ─
    # Collect precision at every 10% step for each model × dataset combination
    all_pcts = list(range(10, 101, 10))

    # Build a dict: model_label -> {dataset -> {pct -> precision}}
    model_pct_data = {}
    for m in model_summaries:
        ml = m["model_label"]
        model_pct_data[ml] = {}
        for dl in all_ds_labels:
            dr = m["dataset_results"].get(dl, {})
            pct_dict = dr.get("pct_precisions", {})   # populated below if present
            model_pct_data[ml][dl] = pct_dict

    # For each depth level produce a heatmap (models × datasets)
    for pct in all_pcts:
        mat = []
        for m in model_summaries:
            row = []
            for dl in all_ds_labels:
                dr = m["dataset_results"].get(dl, {})
                # Try the full pct_precisions dict first, fall back to coarse keys
                pp = dr.get("pct_precisions", {})
                if pct in pp:
                    v = pp[pct] * 100
                else:
                    # Map 10/20/50 to the stored coarse keys
                    coarse = {10: "prec_10", 20: "prec_20", 50: "prec_50"}
                    ck = coarse.get(pct)
                    raw = dr.get(ck, float("nan")) if ck else float("nan")
                    v = raw * 100 if not np.isnan(raw) else float("nan")
                row.append(v)
            mat.append(row)

        mat_df = pd.DataFrame(mat, index=model_labels, columns=all_ds_labels)

        # Only save one combined heatmap (all pcts as columns, models as rows, one file per dataset pair)
        # — individual-pct heatmaps would be too many files; instead build a combined heatmap below
        pass  # data collected above; full heatmap plotted next

    # ── Combined full-range precision heatmap: average precision across datasets per model ──
    # For each model, collect avg precision at each % step across all test datasets
    avg_prec_rows = []
    for m in model_summaries:
        row_vals = []
        for pct in all_pcts:
            vals_at_pct = []
            for dl in all_ds_labels:
                dr = m["dataset_results"].get(dl, {})
                pp = dr.get("pct_precisions", {})
                if pct in pp:
                    vals_at_pct.append(pp[pct] * 100)
                else:
                    coarse = {10: "prec_10", 20: "prec_20", 50: "prec_50"}
                    ck = coarse.get(pct)
                    raw = dr.get(ck, float("nan")) if ck else float("nan")
                    if not np.isnan(raw):
                        vals_at_pct.append(raw * 100)
            row_vals.append(np.nanmean(vals_at_pct) if vals_at_pct else float("nan"))
        avg_prec_rows.append(row_vals)

    avg_prec_df = pd.DataFrame(avg_prec_rows, index=model_labels,
                               columns=[f"{p}%" for p in all_pcts])
    avg_prec_df.to_csv(comp_dir / "multi_model_avg_precision_full.tsv", sep="\t")

    # Transpose: prediction depth on Y-axis, models on X-axis
    avg_prec_df_T = avg_prec_df.T
    fig, ax = plt.subplots(
        figsize=(max(3, n_models * 0.75 + 1.5), max(9, len(all_pcts) * 0.95 + 3))
    )
    sns.heatmap(avg_prec_df_T, annot=True, fmt=".1f", cmap=_CIVIDIS_LIGHT,
                vmin=0, vmax=100, linewidths=0.8, linecolor="black", ax=ax,
                cbar_kws={"label": "Mean Precision % (avg across test datasets)"})
    ax.set_title("Average Precision @ 10–100% of Targets — All Models\n(averaged across all test datasets)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Model (trained on)", fontsize=11)
    ax.set_ylabel("Prediction depth (% of published targets)", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    _savefig(fig, comp_dir / "multi_model_avg_precision_heatmap")

    # ── Per-dataset full-range precision heatmap (models × 10-100%) ───────
    for dl in all_ds_labels:
        pct_mat = []
        valid_models = []
        for m in model_summaries:
            dr = m["dataset_results"].get(dl, {})
            row_vals = []
            has_any = False
            for pct in all_pcts:
                pp = dr.get("pct_precisions", {})
                if pct in pp:
                    row_vals.append(pp[pct] * 100)
                    has_any = True
                else:
                    coarse = {10: "prec_10", 20: "prec_20", 50: "prec_50"}
                    ck = coarse.get(pct)
                    raw = dr.get(ck, float("nan")) if ck else float("nan")
                    row_vals.append(raw * 100 if not np.isnan(raw) else float("nan"))
                    if not np.isnan(raw):
                        has_any = True
            if has_any:
                pct_mat.append(row_vals)
                valid_models.append(m["model_label"])
        if not pct_mat:
            continue
        pct_df = pd.DataFrame(pct_mat, index=valid_models,
                              columns=[f"{p}%" for p in all_pcts])
        # Transpose: prediction depth on Y-axis, models on X-axis
        pct_df_T = pct_df.T
        fig, ax = plt.subplots(
            figsize=(max(3, len(valid_models) * 0.75 + 1.5),
                     max(9, len(all_pcts) * 0.95 + 3))
        )
        sns.heatmap(pct_df_T, annot=True, fmt=".1f", cmap=_CIVIDIS_LIGHT,
                    vmin=0, vmax=100, linewidths=0.8, linecolor="black", ax=ax,
                    cbar_kws={"label": "Precision %"})
        ax.set_title(f"Precision @ 10–100% of Targets — Test dataset: {dl}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Model (trained on)", fontsize=11)
        ax.set_ylabel("Prediction depth (% of published targets)", fontsize=11)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        safe_dl = dl.replace("/", "_").replace(" ", "_")
        _savefig(fig, comp_dir / f"precision_full_{safe_dl}")

    # ── Model ranking by mean precision across test datasets ───────────────
    model_mean_prec = {}
    model_std_prec  = {}
    for m in model_summaries:
        vals = []
        for dl in all_ds_labels:
            dr = m["dataset_results"].get(dl, {})
            # Use precision at 10% as the primary ranking metric
            pp = dr.get("pct_precisions", {})
            v = pp.get(10, dr.get("prec_10", float("nan")))
            if not np.isnan(v):
                vals.append(v * 100)
        model_mean_prec[m["model_label"]] = np.nanmean(vals) if vals else float("nan")
        model_std_prec[m["model_label"]]  = np.nanstd(vals)  if vals else float("nan")

    # Rank models from best to worst
    ranked = sorted(model_mean_prec.items(), key=lambda x: x[1], reverse=True)
    rank_labels = [r[0] for r in ranked]
    rank_means  = [r[1] for r in ranked]
    rank_stds   = [model_std_prec[r[0]] for r in ranked]
    rank_nums   = list(range(1, len(ranked) + 1))

    rank_df_out = pd.DataFrame({
        "rank": rank_nums,
        "model": rank_labels,
        "mean_prec_10pct": rank_means,
        "std_prec_10pct":  rank_stds,
    })
    rank_df_out.to_csv(comp_dir / "model_ranking_by_precision.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(ranked) * 0.65 + 2)))
    bar_colors_rank = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(ranked)))
    bars = ax.barh(range(len(ranked)), rank_means,
                   xerr=rank_stds, color=bar_colors_rank, alpha=0.9,
                   capsize=4, error_kw={"lw": 1.5, "ecolor": "dimgrey"})
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(
        [f"#{i+1}  {lbl}" for i, lbl in enumerate(rank_labels)], fontsize=9
    )
    ax.invert_yaxis()
    for i, (m, s) in enumerate(zip(rank_means, rank_stds)):
        if not np.isnan(m):
            ax.text(m + s + 0.3, i, f"{m:.1f}% ± {s:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Mean Precision @ top-10% of targets (across test datasets)", fontsize=11)
    ax.set_title("Model Ranking by Cross-Dataset Precision\n(mean ± SD across all test datasets)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(v for v in rank_means if not np.isnan(v)) * 1.35 + 5)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, comp_dir / "model_ranking_by_precision")

    # ── 2. Multi-model ROC-AUC heatmap ────────────────────────────────────
    roc_mat = []
    for m in model_summaries:
        row = [m["dataset_results"].get(dl, {}).get("roc_auc", float("nan"))
               for dl in all_ds_labels]
        roc_mat.append(row)
    roc_df = pd.DataFrame(roc_mat, index=model_labels, columns=all_ds_labels)
    roc_df.to_csv(comp_dir / "multi_model_roc_auc_matrix.tsv", sep="\t")

    if not roc_df.isna().all().all():
        fig, ax = plt.subplots(
            figsize=(max(8, n_datasets * 1.1 + 3), max(4, n_models * 0.8 + 2))
        )
        sns.heatmap(roc_df, annot=True, fmt=".3f", cmap="viridis",
                    vmin=0.4, vmax=1.0, linewidths=0.8, linecolor="black", ax=ax,
                    cbar_kws={"label": "ROC-AUC"})
        ax.set_title("ROC-AUC — All Models × All Datasets",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Test dataset", fontsize=11)
        ax.set_ylabel("Model (LOO held-out)", fontsize=11)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        _savefig(fig, comp_dir / "multi_model_roc_heatmap")

    # ── 3. Per-dataset ROC + PR side by side (all models on each dataset) ──
    for dl in all_ds_labels:
        has_data = any(
            "fpr" in m["dataset_results"].get(dl, {})
            for m in model_summaries
        )
        if not has_data:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for i, (m, color) in enumerate(zip(model_summaries, palette_models)):
            dr = m["dataset_results"].get(dl, {})
            lbl = m["model_label"]
            if "fpr" in dr:
                axes[0].plot(dr["fpr"], dr["tpr"], lw=2, color=color,
                             label=f"{lbl}  (AUC={dr['roc_auc']:.3f})")
            if "rec" in dr:
                axes[1].plot(dr["rec"], dr["prec"], lw=2, color=color,
                             label=f"{lbl}  (AP={dr['pr_auc']:.3f})")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        axes[0].set_xlabel("False Positive Rate", fontsize=11)
        axes[0].set_ylabel("True Positive Rate", fontsize=11)
        axes[0].set_title(f"ROC — {dl}", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("Recall", fontsize=11)
        axes[1].set_ylabel("Precision", fontsize=11)
        axes[1].set_title(f"Precision–Recall — {dl}", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        plt.suptitle(f"ROC & PR Curves: {dl} (all models)",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        safe_dl = dl.replace("/", "_").replace(" ", "_")
        _savefig(fig, comp_dir / f"roc_pr_{safe_dl}")

    # ── 4. Grouped AUC bar chart (datasets × models) ──────────────────────
    auc_data = {}
    for m in model_summaries:
        for dl in all_ds_labels:
            auc_val = m["dataset_results"].get(dl, {}).get("roc_auc", float("nan"))
            auc_data.setdefault(dl, {})[m["model_label"]] = auc_val

    valid_ds = [dl for dl in all_ds_labels
                if not all(np.isnan(v) for v in auc_data.get(dl, {}).values())]

    if valid_ds and n_models > 1:
        fig, ax = plt.subplots(figsize=(max(10, len(valid_ds) * 1.2 + 3), 6))
        x = np.arange(len(valid_ds))
        width = 0.8 / n_models
        for i, (m, color) in enumerate(zip(model_summaries, palette_models)):
            vals = [auc_data.get(dl, {}).get(m["model_label"], float("nan"))
                    for dl in valid_ds]
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9, label=m["model_label"],
                   color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_ds, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("ROC-AUC", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
        ax.set_title("ROC-AUC per Dataset — All LOO Models",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        _savefig(fig, comp_dir / "multi_model_auc_comparison")

    # ── 5. Feature importance heatmap across models ────────────────────────
    imp_rows = {}
    for m in model_summaries:
        if m.get("importance_df") is not None:
            imp = m["importance_df"].set_index("feature")["importance"]
            imp_rows[m["model_label"]] = {f: float(imp.get(f, 0.0))
                                          for f in FEATURE_COLUMNS}

    if len(imp_rows) >= 2:
        feat_imp_df = pd.DataFrame(imp_rows).T
        feat_imp_df.to_csv(comp_dir / "multi_model_feature_importance_matrix.tsv", sep="\t")
        n_models_fi   = len(imp_rows)
        n_features_fi = len(FEATURE_COLUMNS)
        fig, ax = plt.subplots(
            figsize=(max(14, n_models_fi * 0.9 + 4),
                     max(6,  n_features_fi * 0.55 + 2))
        )
        sns.heatmap(feat_imp_df.T, annot=True, fmt=".3f", cmap=_CIVIDIS_LIGHT,
                    linewidths=0.8, linecolor="black", ax=ax,
                    cbar_kws={"label": "MDI Importance"})
        ax.set_title("Feature Importance Across LOO Models",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)
        plt.xticks(rotation=35, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        _savefig(fig, comp_dir / "multi_model_feature_importance_heatmap")

    # ── 6. Multi-model summary table ──────────────────────────────────────
    multi_rows = []
    for m in model_summaries:
        for dl in all_ds_labels:
            dr = m["dataset_results"].get(dl, {})
            multi_rows.append({
                "model":           m["model_label"],
                "trained_on":      m.get("training_dataset", ""),
                "test_dataset":    dl,
                "roc_auc":         dr.get("roc_auc", float("nan")),
                "pr_auc":          dr.get("pr_auc",  float("nan")),
                "prec_10pct":      dr.get("prec_10",  float("nan")),
                "prec_20pct":      dr.get("prec_20",  float("nan")),
                "prec_50pct":      dr.get("prec_50",  float("nan")),
            })
    pd.DataFrame(multi_rows).to_csv(comp_dir / "multi_model_summary.tsv", sep="\t", index=False)
    print(f"\n  Multi-model comparison plots saved: {comp_dir}")


# ============================================================================
# ORCHESTRATION: single model run
# ============================================================================

def run_single_model(
    rf_model,
    meta_df: pd.DataFrame,
    normalization: str,
    editing_type: str,
    min_edit_pct: float,
    min_reads: int,
    outdir: Path,
    train_label: str = "unknown",
    feature_weights: dict | None = None,
    skip_labels: set | None = None,
    bg_filter_model: object = None,
    config_feature_set: str = "full22",
) -> dict:
    """
    Run predictions for all datasets with a single model.
    Returns dict: dataset_label -> {roc_auc, pr_auc, prec_10, prec_20, prec_50, fpr, tpr, ...}
    """
    skip_labels = skip_labels or set()
    dataset_labels = [lbl for lbl in meta_df["label"] if lbl not in skip_labels]
    has_per_dataset_bg = "background_files" in meta_df.columns

    print(f"\n  Predicting ({train_label}), {len(dataset_labels)} datasets...")
    for _, ds_row in meta_df[meta_df["label"].isin(dataset_labels)].iterrows():
        # Per-dataset background filter takes priority over model-level one
        row_bg_filter = bg_filter_model
        if has_per_dataset_bg and pd.notna(ds_row.get("background_files", None)):
            bg_paths = parse_files(str(ds_row["background_files"]))
            if bg_paths:
                print(f"\n  [{ds_row['label']}] Learning per-dataset background filter...")
                bg_dfs = [load_editing_sites(Path(p), editing_type=editing_type,
                                             min_edit_pct=0.0, min_reads=0)
                          for p in bg_paths]
                row_bg_filter = learn_background_filter(bg_dfs)

        predict_on_dataset(
            rf_model=rf_model,
            dataset_label=ds_row["label"],
            ds_row=ds_row,
            normalization=normalization,
            editing_type=editing_type,
            min_edit_pct=min_edit_pct,
            min_reads=min_reads,
            outdir=outdir,
            feature_weights=feature_weights,
            bg_filter_model=row_bg_filter,
            config_feature_set=config_feature_set,
        )

    # Collect precision metrics
    results_df = collect_all_results(outdir, dataset_labels, meta_df)
    if not results_df.empty:
        results_df.to_csv(outdir / "all_results.tsv", sep="\t", index=False)

    # Build dataset_results dict
    dataset_results = {}
    for dl in dataset_labels:
        ds_row    = meta_df[meta_df["label"] == dl].iloc[0]
        pred_path = outdir / dl / "predictions.tsv"
        targets   = ds_row["targets_file"].strip()

        dr = {}
        # Precision at all depths (10%–100%) + coarse convenience keys
        if not results_df.empty:
            sub = results_df[results_df["dataset"] == dl]
            pct_precisions = {}
            for _, srow in sub.iterrows():
                pct_precisions[int(srow["top_percent"])] = float(srow["precision"])
            dr["pct_precisions"] = pct_precisions
            # Convenience keys for summary table
            for pct, key in [(10, "prec_10"), (20, "prec_20"), (50, "prec_50")]:
                if pct in pct_precisions:
                    dr[key] = pct_precisions[pct]
        # ROC/PR
        rp = compute_roc_pr(pred_path, targets)
        if rp:
            dr.update({
                "roc_auc": rp["roc_auc"],
                "pr_auc":  rp["pr_auc"],
                "fpr":     rp["fpr"],
                "tpr":     rp["tpr"],
                "prec":    rp["prec"],
                "rec":     rp["rec"],
            })
        dataset_results[dl] = dr

    # Single-model plots
    if not results_df.empty:
        generate_single_model_plots(
            results_df=results_df,
            outdir=outdir,
            dataset_labels=dataset_labels,
            meta_df=meta_df[meta_df["label"].isin(dataset_labels)],
            train_label=train_label,
            model_outdir=outdir,
        )

    return dataset_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test RF model(s) on all datasets; generate precision heatmaps and more.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metadata", type=str, required=True,
        help="Metadata TSV: label, expt_files, ctrl_files, targets_file "
             "(bg_files accepted as alias for ctrl_files; background_files column optional)",
    )

    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", type=str,
        help="Single model mode: path to rf_model.pkl",
    )
    model_group.add_argument(
        "--loo-dir", type=str,
        help="Batch mode: directory containing Trained_*/rf_model.pkl (output of train_model.py --metadata)",
    )
    model_group.add_argument(
        "--models", nargs="+",
        help="Multi-model mode: explicit list of rf_model.pkl paths",
    )

    parser.add_argument(
        "--model-labels", nargs="+", default=None,
        help="Labels for models given in --models (same order). "
             "Default: infer from parent directory names.",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("outputs/cross_test"),
    )
    parser.add_argument(
        "--train-label", type=str, default=None,
        help="Label for the training dataset (single model mode, for plot titles).",
    )
    parser.add_argument("--normalization", choices=["simple", "median_of_ratios"], default=None)
    parser.add_argument("--min-edit-pct", type=float, default=None)
    parser.add_argument("--min-reads", type=int, default=None)
    parser.add_argument("--editing-type", choices=["AtoG", "CtoT"], default=None)
    parser.add_argument(
        "--background", nargs="+", default=None,
        help="Optional: background control file(s) (e.g. gDNA, no-enzyme control) "
             "to filter low-confidence sites before feature extraction. "
             "If not provided, background_filter.pkl is auto-loaded from each model's "
             "directory if present. Can also be specified per-dataset in the metadata "
             "TSV as a 'background_files' column.",
    )
    parser.add_argument(
        "--no-background-filter", action="store_true", default=False,
        help="Disable background filtering even if background_filter.pkl is present "
             "alongside a model.",
    )
    parser.add_argument("--log", type=str, nargs="?", const="Logs", default=None)

    args = parser.parse_args()
    setup_logging(args.log, prefix="cross_test")

    meta_df = load_metadata(Path(args.metadata))

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("CROSS-DATASET TESTING")
    print(f"{'='*70}")

    # ── Determine which models to run ─────────────────────────────────────
    # Each entry: (model_pkl_path, model_label, holdout_dataset_or_None, config)

    model_specs = []

    if args.model:
        # Single model
        model_path = Path(args.model)
        config = {}
        config_path = model_path.parent / "model_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        label = args.train_label or config.get("train_label", model_path.parent.name)
        model_specs.append((model_path, label, None, config))

    elif args.loo_dir:
        # Auto-discover Trained_*/rf_model.pkl
        loo_dir = Path(args.loo_dir)
        loo_paths = sorted(loo_dir.glob("Trained_*/rf_model.pkl"))
        if not loo_paths:
            print(f"  ERROR: No Trained_*/rf_model.pkl found in {loo_dir}")
            return
        print(f"  Discovered {len(loo_paths)} models in {loo_dir}")
        for mp in loo_paths:
            config = {}
            cp = mp.parent / "model_config.json"
            if cp.exists():
                with open(cp) as f:
                    config = json.load(f)
            lbl         = mp.parent.name  # e.g. "Trained_Ataxin"
            # training_dataset is the one the model was trained on — skip it during testing
            training_ds = config.get("training_dataset", lbl.replace("Trained_", ""))
            model_specs.append((mp, lbl, training_ds, config))

    else:
        # Explicit multi-model
        model_paths = [Path(p) for p in args.models]
        if args.model_labels and len(args.model_labels) != len(model_paths):
            parser.error("--model-labels count must match --models count")
        for i, mp in enumerate(model_paths):
            config = {}
            cp = mp.parent / "model_config.json"
            if cp.exists():
                with open(cp) as f:
                    config = json.load(f)
            lbl        = (args.model_labels[i] if args.model_labels
                          else config.get("train_label", mp.parent.name))
            training_ds = config.get("training_dataset", None)
            model_specs.append((mp, lbl, training_ds, config))

    # ── Run each model ────────────────────────────────────────────────────
    is_multi = len(model_specs) > 1
    model_summaries = []

    for model_path, model_label, training_ds, config in model_specs:
        print(f"\n{'─'*60}")
        print(f"  Model       : {model_label}")
        print(f"  Path        : {model_path}")
        if training_ds:
            print(f"  Trained on  : {training_ds} (will be SKIPPED during testing)")

        with open(model_path, "rb") as f:
            rf_model = pickle.load(f)

        # Resolve parameters
        normalization = args.normalization or config.get("normalization", "simple")
        editing_type  = args.editing_type  or config.get("editing_type",  "AtoG")
        min_edit_pct  = args.min_edit_pct  if args.min_edit_pct is not None else config.get("min_edit_pct", 0.0)
        min_reads     = args.min_reads     if args.min_reads is not None     else config.get("min_reads", 20)

        fw_raw = config.get("feature_weights", None)
        feature_weights = None
        if fw_raw and isinstance(fw_raw, dict):
            feature_weights = {k: float(v) for k, v in fw_raw.items()}

        # Feature set: read from model config (so testing always matches training)
        config_feature_set = config.get("feature_set", "full22")
        if config_feature_set not in VALID_FEATURE_SETS:
            config_feature_set = "full22"
        print(f"  Feature set : {config_feature_set} ({len(get_feature_columns(config_feature_set))} features)")

        # ── Background filter: CLI > model-dir auto-load > none ───────────
        bg_filter_model = None
        if not getattr(args, "no_background_filter", False):
            if getattr(args, "background", None):
                # Learn from CLI files (shared across all models in this run)
                if not hasattr(args, "_bg_filter_cache"):
                    print("\n  Learning background filter from --background files...")
                    bg_dfs = [load_editing_sites(Path(p), editing_type=editing_type,
                                                 min_edit_pct=0.0, min_reads=0)
                              for p in args.background]
                    args._bg_filter_cache = learn_background_filter(bg_dfs)
                bg_filter_model = args._bg_filter_cache
            else:
                # Auto-load from model directory
                filter_path = model_path.parent / "background_filter.pkl"
                bg_filter_model = load_background_filter(filter_path)
                if bg_filter_model is not None:
                    print(f"  Auto-loaded background filter: {filter_path}")

        # Output dir for this model's results
        model_outdir = args.outdir / model_label if is_multi else args.outdir
        model_outdir.mkdir(parents=True, exist_ok=True)

        # Skip the dataset this model was trained on (to avoid train/test overlap)
        skip_labels = {training_ds} if training_ds else set()

        dataset_results = run_single_model(
            rf_model=rf_model,
            meta_df=meta_df,
            normalization=normalization,
            editing_type=editing_type,
            min_edit_pct=min_edit_pct,
            min_reads=min_reads,
            outdir=model_outdir,
            train_label=model_label,
            feature_weights=feature_weights,
            skip_labels=skip_labels,
            bg_filter_model=bg_filter_model,
            config_feature_set=config_feature_set,
        )

        # Load importance if available
        importance_df = None
        imp_path = model_path.parent / "feature_importance.tsv"
        if imp_path.exists():
            importance_df = pd.read_csv(imp_path, sep="\t")

        model_summaries.append({
            "model_label":      model_label,
            "model_path":       str(model_path),
            "training_dataset": training_ds,
            "dataset_results":  dataset_results,
            "importance_df":    importance_df,
        })

    # ── Multi-model comparison plots ──────────────────────────────────────
    if is_multi and len(model_summaries) >= 2:
        print(f"\n{'─'*60}")
        print("  Generating multi-model comparison plots...")
        generate_multi_model_plots(
            model_summaries=model_summaries,
            outdir=args.outdir,
            meta_df=meta_df,
        )

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"  Predictions : {args.outdir}/[model]/[dataset]/predictions.tsv")
    print(f"  Plots       : {args.outdir}/comparison/")


if __name__ == "__main__":
    main()
