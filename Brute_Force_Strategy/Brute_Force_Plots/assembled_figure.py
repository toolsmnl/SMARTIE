"""
assembled_figure.py — Publication-ready assembled figure (Panels A–D)
======================================================================

Layout:
    A (full width) : Heatmap — rows=kmax, cols=precision thresholds,
                     cells=mean precision of top-N combos
    B (left half)  : Violin — precision distribution per kmax,
                     Kruskal-Wallis + Dunn-Bonferroni CLD
    C (right half) : Feature ranking — rank-weighted horizontal bar chart
    D (full width) : Precision trend across thresholds, one line per kmax

Usage:
    python assembled_figure.py \\
        --kmax-dir   Figures/ \\
        --core-features fold_change fisher_odds_ratio binom_neg_log10_p \\
            site_enrichment norm_edits log_true_signal site_fraction \\
            avg_cum_expt_edit_pct site_edit_sd \\
            mean_expt_max_site_rate std_expt_max_site_rate \\
        --top-n      10 \\
        --min-combos 10 \\
        --feat-col   features \\
        --sep        | \\
        --outdir     figures/assembled/ \\
        --dpi        300

Flags:
    --kmax-dir       Root dir containing kmax1/, kmax2/, ... subfolders
    --core-features  Feature names to highlight in Panel C
    --top-n          Top-N combos per kmax (default: 10)
    --min-combos     Skip kmax with fewer combos (default: 10)
    --feat-col       Column with feature list (default: features)
    --sep            Separator between features (default: |)
    --outdir         Output directory (default: same as --kmax-dir)
    --dpi            Output resolution (default: 300)
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
warnings.filterwarnings("ignore")


# ============================================================================
# CONSTANTS
# ============================================================================

THRESHOLDS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
HM_LABELS  = [f"{p}%" for p in THRESHOLDS]
PREC_COLS  = [f"precision_at_{lbl}" for lbl in HM_LABELS]

# Short category tag per feature — shown next to feature name in Panel C
# Based on actual biological/statistical feature categories
# per-replicate aggregates (mean/std/max across reps) → [Per Rep]
FEAT_CATEGORY = {
    # Cumulative editing
    "avg_cum_expt_edit_pct":     "[Cum Edit]",
    # Normalisation
    "norm_edits":                "[Norm]",
    # Signal
    "log_true_signal":           "[Signal]",
    "signal_baseline":           "[Signal]",
    # Site-based
    "site_edit_sd":              "[Edit SD]",
    "site_fraction":             "[Sites]",
    "site_enrichment":           "[Sites]",
    "site_enrichment_edit_rate": "[Sites]",
    # Fold change / enrichment
    "fold_change":               "[FC]",
    # Statistical tests
    "fisher_odds_ratio":         "[Fisher]",
    "binom_neg_log10_p":         "[Binom]",
    # Per-replicate aggregates
    "mean_expt_max_site_rate":   "[Per Rep]",
    "std_expt_max_site_rate":    "[Per Rep]",
    "std_expt_n_sites":          "[Per Rep]",
    "max_expt_n_sites":          "[Per Rep]",
    "max_expt_max_site_rate":    "[Per Rep]",
    "mean_expt_n_sites":         "[Per Rep]",
    "std_expt_sum_edits":        "[Per Rep]",
    "max_expt_sum_total":        "[Per Rep]",
    "max_ctrl_sum_edits":        "[Per Rep]",
    "std_ctrl_n_sites":          "[Per Rep]",
    "mean_ctrl_sum_edits":       "[Per Rep]",
    "std_ctrl_sum_total":        "[Per Rep]",
    "mean_ctrl_n_sites":         "[Per Rep]",
    "mean_ctrl_gene_rate":       "[Per Rep]",
}

_PALETTE = [
    "#264653", "#2A9D8F", "#E9C46A", "#E76F51",
    "#9B5DE5", "#F15BB5", "#00BBF9", "#00F5D4",
    "#2C6E91", "#4CAF82", "#E07B39", "#C94040",
    "#6A4C93", "#1982C4",
]

_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "prec_red",
    ["#FFFFFF", "#FCEBEB", "#F5B3B2", "#E24B4A", "#A32D2D", "#5A1717"],
)


# ============================================================================
# STYLE
# ============================================================================

def _apply_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Georgia", "DejaVu Serif", "Times New Roman"],
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "axes.linewidth":     1.1,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.labelsize":    8.5,
        "ytick.labelsize":    8.5,
        "legend.frameon":     False,
        "legend.fontsize":    8.5,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.08,
    })


# ============================================================================
# DATA LOADING
# ============================================================================

def _discover(kmax_dir: Path, min_combos: int, feat_col: str) -> dict:
    """Return {kmax_int -> DataFrame} for all valid kmax subfolders."""
    pattern = re.compile(r"^kmax(\d+)$", re.IGNORECASE)
    found   = {}

    for subdir in sorted(kmax_dir.iterdir()):
        if not subdir.is_dir():
            continue
        m = pattern.match(subdir.name)
        if not m:
            continue
        k = int(m.group(1))

        csvs = sorted(subdir.glob("kmax*ranked_combos.csv")) or \
               sorted(subdir.glob("*.csv"))
        if not csvs:
            print(f"  [skip] kmax={k}: no CSV")
            continue

        try:
            df = pd.read_csv(csvs[0])
        except Exception as exc:
            print(f"  [skip] kmax={k}: {exc}")
            continue

        if len(df) < min_combos:
            print(f"  [skip] kmax={k}: {len(df)} combos < {min_combos}")
            continue

        found[k] = df
        print(f"  kmax={k:2d}: {len(df):,} combos")

    return dict(sorted(found.items()))


# ============================================================================
# PANEL A — HEATMAP
# ============================================================================

def _draw_heatmap(ax, kmax_data: dict, top_n: int) -> None:
    """
    Rows = kmax values, Cols = precision thresholds.
    Cells = mean precision of top-N combos for that kmax × threshold.
    """
    kmax_list = sorted(kmax_data.keys())
    matrix    = np.full((len(kmax_list), len(THRESHOLDS)), np.nan)

    for i, k in enumerate(kmax_list):
        top_df = kmax_data[k].head(top_n)
        for j, lbl in enumerate(HM_LABELS):
            col = f"precision_at_{lbl}"
            if col in top_df.columns:
                vals = pd.to_numeric(top_df[col], errors="coerce").dropna()
                if len(vals):
                    matrix[i, j] = float(vals.mean())

    im = ax.imshow(matrix, aspect="auto", cmap=_HEATMAP_CMAP,
                   vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(THRESHOLDS)))
    ax.set_xticklabels(HM_LABELS, fontsize=8.5)
    ax.set_yticks(range(len(kmax_list)))
    ax.set_yticklabels([f"kmax={k}" for k in kmax_list], fontsize=8.5)
    ax.set_xlabel("Precision Threshold", labelpad=5)
    ax.set_ylabel("kmax", labelpad=5)
    ax.set_title(
        f"A  —  Mean Test-Only Precision  [top {top_n} combos per kmax]",
        fontweight="bold", pad=8, loc="left",
    )

    # Cell annotations
    for i in range(len(kmax_list)):
        for j in range(len(THRESHOLDS)):
            val = matrix[i, j]
            if not np.isnan(val):
                tc = "#FFFFFF" if val > 0.65 else "#2C2C2C"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=tc)

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("Mean Precision", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)


# ============================================================================
# PANEL B — VIOLIN with raw mean precision labels
# ============================================================================

def _draw_violin(ax, kmax_data: dict, top_n: int) -> None:
    """
    Violin + strip plot of mean precision (avg across thresholds) per kmax.
    Raw mean precision value annotated above each violin.
    """
    kmax_list = sorted(kmax_data.keys())
    pal       = {str(k): _PALETTE[i % len(_PALETTE)]
                 for i, k in enumerate(kmax_list)}

    # Build long-form DataFrame: one row per combo, value = mean across thresholds
    rows = []
    for k in kmax_list:
        top_df = kmax_data[k].head(top_n)
        for _, row in top_df.iterrows():
            vals = [pd.to_numeric(row.get(f"precision_at_{lbl}", np.nan),
                                  errors="coerce")
                    for lbl in HM_LABELS]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                rows.append({"kmax": str(k), "precision": float(np.mean(vals))})

    plot_df = pd.DataFrame(rows)
    order   = [str(k) for k in kmax_list]

    sns.violinplot(data=plot_df, x="kmax", y="precision",
                   palette=pal, order=order, width=0.7,
                   linewidth=0.8, cut=0, ax=ax)
    sns.stripplot(data=plot_df, x="kmax", y="precision",
                  palette=pal, order=order, size=2.5,
                  alpha=0.4, jitter=True, zorder=5, ax=ax)

    # Annotate raw mean precision above each violin
    y_max = plot_df["precision"].max()
    y_off = (plot_df["precision"].max() - plot_df["precision"].min()) * 0.04
    for xi, k in enumerate(kmax_list):
        grp_vals = plot_df[plot_df["kmax"] == str(k)]["precision"]
        if len(grp_vals):
            mean_val = float(grp_vals.mean())
            ax.text(xi, y_max + y_off, f"{mean_val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color="#333333")

    ax.set_title(
        "B  —  Precision Distribution per kmax",
        fontweight="bold", pad=8, loc="left", fontsize=10,
    )
    ax.set_xlabel("kmax", labelpad=5)
    ax.set_ylabel("Mean Precision (test-only)", labelpad=5)
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))


# ============================================================================
# ============================================================================
# PANEL C — FEATURE APPEARANCE COUNT
# ============================================================================

def _compute_appearance_counts(kmax_data: dict, top_n: int,
                                feat_col: str, sep: str) -> pd.DataFrame:
    """Count how many times each feature appears across top-N combos per kmax."""
    counts       = defaultdict(int)
    total_combos = 0
    for kmax, df in kmax_data.items():
        top_df = df.head(top_n)
        total_combos += len(top_df)
        for _, row in top_df.iterrows():
            raw = str(row.get(feat_col, ""))
            for feat in [f.strip() for f in raw.split(sep) if f.strip()]:
                counts[feat] += 1
    rows = [{"feature": f, "appearance_count": counts[f]} for f in sorted(counts)]
    result = (pd.DataFrame(rows)
              .sort_values("appearance_count", ascending=False)
              .reset_index(drop=True))
    result.insert(0, "rank", result.index + 1)
    result["total_combos"] = total_combos
    return result


def _draw_feature_ranking(ax, kmax_data: dict, core_features: set,
                           top_n: int, feat_col: str, sep: str) -> None:
    scores_df    = _compute_appearance_counts(kmax_data, top_n, feat_col, sep)
    total_combos = int(scores_df["total_combos"].iloc[0])
    n_feats      = len(scores_df)

    colours = ["#2C6E91" if f in core_features else "#B0C4D8"
               for f in scores_df["feature"]]
    y_pos  = np.arange(n_feats)
    values = scores_df["appearance_count"].values
    max_v  = float(values.max()) if len(values) else 1.0

    bars = ax.barh(y_pos, values, color=colours,
                   edgecolor="none", height=0.72, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max_v * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(val)}", va="center", ha="left",
                fontsize=6.5, color="#333333")

    # Y-axis labels: feature name + category tag
    y_labels = []
    for feat in scores_df["feature"]:
        tag = FEAT_CATEGORY.get(feat, "")
        y_labels.append(f"{feat}  {tag}" if tag else feat)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"Appearance Count (of {total_combos} combos)", labelpad=5)
    ax.set_xlim(left=20)   # start from 20 for breathing room

    kmax_list  = sorted(kmax_data.keys())
    kmax_range = f"kmax {min(kmax_list)}–{max(kmax_list)}"
    ax.set_title(
        f"C  —  Feature Appearance Count  [{kmax_range}]",
        fontweight="bold", pad=8, loc="left",
    )
    ax.grid(axis="x", ls="--", lw=0.5, alpha=0.4, zorder=1)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    n_core = sum(1 for f in scores_df["feature"] if f in core_features)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="#2C6E91", label=f"Core  (n={n_core})"),
            mpatches.Patch(facecolor="#B0C4D8",
                           label=f"Variable  (n={n_feats-n_core})"),
        ],
        loc="lower right", fontsize=8,
    )


# ============================================================================
# PANEL D — PRECISION TREND
# Best kmax (highest avg precision across all thresholds) = thick solid line
# All others = thin dotted lines
# ============================================================================

def _draw_precision_trend(ax, kmax_data: dict, top_n: int) -> None:
    kmax_list = sorted(kmax_data.keys())
    x_vals    = list(THRESHOLDS)

    # Compute per-kmax mean precision for ranking
    kmax_means = {}
    all_means  = {}
    for kmax in kmax_list:
        top_df = kmax_data[kmax].head(top_n)
        means  = []
        for lbl in HM_LABELS:
            col  = f"precision_at_{lbl}"
            vals = (pd.to_numeric(top_df[col], errors="coerce").dropna()
                    if col in top_df.columns else pd.Series(dtype=float))
            means.append(float(vals.mean()) if len(vals) else np.nan)
        arr = np.array(means, dtype=float)
        kmax_means[kmax] = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else np.nan
        all_means[kmax]  = arr

    # Identify best kmax
    valid_kmax = {k: v for k, v in kmax_means.items() if not np.isnan(v)}
    best_kmax  = max(valid_kmax, key=valid_kmax.get) if valid_kmax else None

    for i, kmax in enumerate(kmax_list):
        colour = _PALETTE[i % len(_PALETTE)]
        means  = all_means[kmax]

        if np.all(np.isnan(means)):
            continue

        is_best = (kmax == best_kmax)

        ax.plot(
            x_vals, means,
            color=colour,
            lw=2.8 if is_best else 0.9,
            ls="-"  if is_best else ":",
            marker="o" if is_best else None,
            ms=5 if is_best else 0,
            label=(f"kmax={kmax} ★ best" if is_best else f"kmax={kmax}"),
            zorder=5 if is_best else 2,
            alpha=1.0 if is_best else 0.65,
        )

    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{x}%" for x in x_vals], fontsize=8.5)
    ax.set_xlabel("Precision Threshold", labelpad=5)
    ax.set_ylabel("Mean Precision (test-only)", labelpad=5)
    best_str = f"  [best: kmax={best_kmax}]" if best_kmax else ""
    ax.set_title(
        f"D  —  Precision Trend Across Thresholds  "
        f"[top {top_n} combos per kmax]{best_str}",
        fontweight="bold", pad=8, loc="left",
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4, zorder=1)
    ax.grid(axis="x", ls=":",  lw=0.4, alpha=0.25, zorder=1)

    ncol = 2 if len(kmax_list) > 7 else 1
    ax.legend(title="kmax", title_fontsize=8, ncol=ncol,
              loc="upper right", fontsize=7.5)


# PANEL D — GROUPED BOXPLOT: precision@10%, 50%, 100% for kmax 8-10
# ============================================================================

def _draw_precision_trend(ax, kmax_data: dict, top_n: int) -> None:
    """
    Grouped boxplot showing precision at 10%, 50%, and 100% thresholds
    for kmax 8, 9, and 10 only.

    Layout: three threshold groups on x-axis; within each group, one box
    per kmax coloured distinctly.  Skips any kmax not present in kmax_data.
    """
    TARGET_KMAX   = [8, 9, 10]
    TARGET_THRESH = [("10%", "precision_at_10%"),
                     ("50%", "precision_at_50%"),
                     ("100%","precision_at_100%")]

    # Filter to kmax values that actually exist
    present_kmax = [k for k in TARGET_KMAX if k in kmax_data]
    if not present_kmax:
        ax.text(0.5, 0.5, "No kmax 8-10 data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="red")
        return

    # Assign colours matching _PALETTE indices for kmax 8, 9, 10
    all_kmax  = sorted(kmax_data.keys())
    kmax_pal  = {k: _PALETTE[all_kmax.index(k) % len(_PALETTE)]
                 for k in present_kmax}

    n_thresh  = len(TARGET_THRESH)
    n_kmax    = len(present_kmax)
    group_w   = 0.7
    box_w     = group_w / n_kmax

    positions_per_thresh = []
    for gi in range(n_thresh):
        centre   = gi                        # group centre at 0, 1, 2
        offsets  = [(j - (n_kmax - 1) / 2) * box_w for j in range(n_kmax)]
        positions_per_thresh.append([centre + o for o in offsets])

    for ki, kmax in enumerate(present_kmax):
        top_df = kmax_data[kmax].head(top_n)
        colour = kmax_pal[kmax]

        vals_per_thresh = []
        for _, col in TARGET_THRESH:
            if col in top_df.columns:
                v = pd.to_numeric(top_df[col], errors="coerce").dropna().values
            else:
                v = np.array([])
            vals_per_thresh.append(v)

        positions = [positions_per_thresh[gi][ki] for gi in range(n_thresh)]

        bp = ax.boxplot(
            vals_per_thresh,
            positions=positions,
            widths=box_w * 0.82,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(facecolor=colour, alpha=0.75, linewidth=0.8),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker=".", markersize=3,
                            markerfacecolor=colour, alpha=0.5),
        )

        # Invisible proxy for legend
        ax.plot([], [], color=colour, lw=6, alpha=0.75, label=f"kmax={kmax}")

    # X-axis: one tick per threshold group
    ax.set_xticks(range(n_thresh))
    ax.set_xticklabels([lbl for lbl, _ in TARGET_THRESH], fontsize=9)
    ax.set_xlabel("Precision Threshold", labelpad=5)
    ax.set_ylabel("Precision (test-only)", labelpad=5)
    ax.set_xlim(-0.5, n_thresh - 0.5)
    ax.set_title(
        f"D  —  Precision Distribution at Key Thresholds  "
        f"[top {top_n} combos, kmax 8–10]",
        fontweight="bold", pad=8, loc="left",
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4, zorder=1)
    ax.legend(title="kmax", title_fontsize=8, loc="upper right", fontsize=8)



# ============================================================================
# PANEL D — FOUR STYLES
# TARGET_KMAX = [8, 9, 10], TARGET_THRESH = p@10%, p@50%, p@100%
# ============================================================================

_D_TARGET_KMAX   = [8, 9, 10]
_D_TARGET_THRESH = [("p@10%",  "precision_at_10%"),
                    ("p@50%",  "precision_at_50%"),
                    ("p@100%", "precision_at_100%")]
_D_THRESH_COLOURS = ["#2C6E91", "#E07B39", "#C94040"]
_D_KMAX_COLOURS   = {8: "#2A9D8F", 9: "#E9C46A", 10: "#E76F51"}


def _d_get_vals(df, col):
    if col not in df.columns:
        return np.array([])
    return pd.to_numeric(df[col], errors="coerce").dropna().values


def _d_finish(ax, title, top_n):
    ax.set_title(
        f"D  —  {title}  [top {top_n} combos, kmax 8–10]",
        fontweight="bold", pad=8, loc="left",
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4, zorder=1)
    ax.set_ylabel("Precision (test-only)", labelpad=5)


def _draw_panel_d_beeswarm(ax, kmax_data: dict, top_n: int) -> None:
    """Individual dots spread within threshold groups, median bar."""
    present = sorted(k for k in _D_TARGET_KMAX if k in kmax_data)
    if not present:
        ax.text(0.5, 0.5, "No kmax 8-10 data", ha="center", va="center",
                transform=ax.transAxes, color="red"); return
    n_k = len(present); n_t = len(_D_TARGET_THRESH)
    slot_w = 0.65 / n_k
    for gi, (lbl, col) in enumerate(_D_TARGET_THRESH):
        for ki, kmax in enumerate(present):
            colour = _D_KMAX_COLOURS[kmax]
            vals   = _d_get_vals(kmax_data[kmax].head(top_n), col)
            if not len(vals): continue
            centre = gi + (ki - (n_k-1)/2) * slot_w
            n = len(vals)
            spread = slot_w * 0.35
            xs = ([centre] if n == 1 else
                  [centre + spread*(i-(n-1)/2)/max(1,(n-1)/2) for i in range(n)])
            ax.scatter(xs, np.sort(vals), color=colour, s=22, alpha=0.8, zorder=4)
            ax.plot([centre-slot_w*0.3, centre+slot_w*0.3],
                    [np.median(vals)]*2, color="black", lw=1.5, zorder=5)
    ax.set_xticks(range(n_t))
    ax.set_xticklabels([l for l,_ in _D_TARGET_THRESH], fontsize=8.5)
    ax.set_xlabel("Threshold", labelpad=4)
    ax.set_xlim(-0.5, n_t-0.5)
    ax.legend(handles=[mpatches.Patch(facecolor=_D_KMAX_COLOURS[k],
              label=f"kmax={k}") for k in present],
              title="kmax", title_fontsize=7.5, loc="lower left", fontsize=7.5)
    _d_finish(ax, "Beeswarm + Median", top_n)


def _draw_panel_d_raincloud(ax, kmax_data: dict, top_n: int) -> None:
    """Half-violin + jittered dots + median bar."""
    from scipy.stats import gaussian_kde
    present = sorted(k for k in _D_TARGET_KMAX if k in kmax_data)
    if not present:
        ax.text(0.5, 0.5, "No kmax 8-10 data", ha="center", va="center",
                transform=ax.transAxes, color="red"); return
    n_k = len(present); n_t = len(_D_TARGET_THRESH)
    slot_w = 0.8 / n_t
    for ki, kmax in enumerate(present):
        for ti, (lbl, col) in enumerate(_D_TARGET_THRESH):
            colour = _D_THRESH_COLOURS[ti]
            vals   = _d_get_vals(kmax_data[kmax].head(top_n), col)
            if not len(vals): continue
            centre = ki + (ti-(n_t-1)/2)*slot_w
            # Half violin
            try:
                kde  = gaussian_kde(vals, bw_method=0.4)
                y_ev = np.linspace(vals.min()-0.01, vals.max()+0.01, 200)
                dens = kde(y_ev); dens = dens/dens.max()*slot_w*0.38
                ax.fill_betweenx(y_ev, centre-dens, centre,
                                 color=colour, alpha=0.45, zorder=2)
            except Exception: pass
            # Dots
            np.random.seed(42)
            jx = centre + np.random.uniform(0.01, slot_w*0.3, size=len(vals))
            ax.scatter(jx, vals, color=colour, s=16, alpha=0.7, zorder=4)
            ax.plot([centre-slot_w*0.1, centre+slot_w*0.35],
                    [np.median(vals)]*2, color="black", lw=1.5, zorder=5)
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f"kmax={k}" for k in present], fontsize=8.5)
    ax.set_xlabel("kmax", labelpad=4); ax.set_xlim(-0.5, n_k-0.5)
    ax.legend(handles=[mpatches.Patch(facecolor=_D_THRESH_COLOURS[i],
              label=_D_TARGET_THRESH[i][0]) for i in range(n_t)],
              title="Threshold", title_fontsize=7.5, loc="lower left", fontsize=7.5)
    _d_finish(ax, "Raincloud", top_n)


def _draw_panel_d_paired(ax, kmax_data: dict, top_n: int) -> None:
    """Each combo tracked as a connected line across kmax 8-10."""
    present = sorted(k for k in _D_TARGET_KMAX if k in kmax_data)
    if not present:
        ax.text(0.5, 0.5, "No kmax 8-10 data", ha="center", va="center",
                transform=ax.transAxes, color="red"); return
    for ti, (lbl, col) in enumerate(_D_TARGET_THRESH):
        colour  = _D_THRESH_COLOURS[ti]
        max_n   = min(top_n, min(
            len(_d_get_vals(kmax_data[k].head(top_n), col)) for k in present))
        for ci in range(max_n):
            ys = []
            for kmax in present:
                v = _d_get_vals(kmax_data[kmax].head(top_n), col)
                ys.append(float(v[ci]) if ci < len(v) else np.nan)
            ax.plot(range(len(present)), ys,
                    color=colour, lw=0.8, alpha=0.35, zorder=2)
            ax.scatter(range(len(present)), ys,
                       color=colour, s=20, alpha=0.6, zorder=3)
        medians = [float(np.median(_d_get_vals(kmax_data[k].head(top_n), col)))
                   if len(_d_get_vals(kmax_data[k].head(top_n), col)) else np.nan
                   for k in present]
        ax.plot(range(len(present)), medians, color=colour, lw=2.2,
                marker="o", ms=6, markeredgecolor="white",
                markeredgewidth=0.8, zorder=5, label=lbl)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([f"kmax={k}" for k in present], fontsize=8.5)
    ax.set_xlabel("kmax", labelpad=4); ax.set_xlim(-0.3, len(present)-0.7)
    ax.legend(title="Threshold", title_fontsize=7.5,
              loc="lower left", fontsize=7.5)
    _d_finish(ax, "Paired Dot + Median", top_n)


def _draw_panel_d_scatter(ax, kmax_data: dict, top_n: int) -> None:
    """
    Grouped boxplot for kmax 7-11, three threshold groups (p@10%, p@50%, p@100%).
    X-axis = thresholds, within each group one box per kmax, coloured by kmax.
    Y-axis fixed 0.80-1.0 for resolution.
    """
    present = sorted(k for k in kmax_data.keys() if 7 <= k <= 11)
    if not present:
        ax.text(0.5, 0.5, "No kmax 7-11 data available",
                ha="center", va="center", transform=ax.transAxes, color="red")
        return

    n_k    = len(present)
    n_t    = len(_D_TARGET_THRESH)
    bw     = 0.7 / n_k   # width of each box

    # Assign kmax colours from global palette
    all_sorted = sorted(kmax_data.keys())
    kmax_pal   = {k: _PALETTE[all_sorted.index(k) % len(_PALETTE)]
                  for k in present}

    for gi, (lbl, col) in enumerate(_D_TARGET_THRESH):
        for ki, kmax in enumerate(present):
            vals   = _d_get_vals(kmax_data[kmax].head(top_n), col)
            if not len(vals):
                continue
            pos    = gi + (ki - (n_k - 1) / 2) * bw
            colour = kmax_pal[kmax]

            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=bw * 0.82,
                patch_artist=True,
                notch=False,
                medianprops=dict(color="black", linewidth=1.8),
                boxprops=dict(facecolor=colour, alpha=0.75, linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker=".", markersize=3,
                                markerfacecolor=colour, alpha=0.5),
            )

    # Legend proxy
    for kmax in present:
        ax.plot([], [], color=kmax_pal[kmax], lw=6,
                alpha=0.75, label=f"kmax={kmax}")

    ax.set_xticks(range(n_t))
    ax.set_xticklabels([lbl for lbl, _ in _D_TARGET_THRESH], fontsize=9)
    ax.set_xlabel("Precision Threshold", labelpad=5)
    ax.set_xlim(-0.5, n_t - 0.5)
    ax.set_ylim(0.80, 1.0)
    ax.set_title(
        f"D  —  Precision Distribution  [top {top_n} combos, kmax 7–11]",
        fontweight="bold", pad=8, loc="left",
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4, zorder=1)
    ax.set_ylabel("Precision (test-only)", labelpad=5)
    ax.legend(title="kmax", title_fontsize=7.5,
              loc="lower left", fontsize=7.5)


_PANEL_D_STYLES = {
    "beeswarm": _draw_panel_d_beeswarm,
    "raincloud": _draw_panel_d_raincloud,
    "paired":    _draw_panel_d_paired,
    "scatter":   _draw_panel_d_scatter,
}

# ============================================================================
# ASSEMBLE
# ============================================================================

def assemble_figure(kmax_data: dict, core_features: set,
                    top_n: int, feat_col: str, sep: str,
                    outdir: Path, dpi: int,
                    panel_d_style: str = "beeswarm") -> None:
    """
    Build the four-panel assembled figure and save as PDF + SVG + PNG.
    panel_d_style: one of beeswarm | raincloud | paired | scatter
    """
    _apply_style()

    n_feats_approx = len(_compute_appearance_counts(
        kmax_data, top_n, feat_col, sep
    ))

    # Dynamic height: feature panel height drives row 2
    feat_h  = max(4.0, n_feats_approx * 0.22)
    hmap_h  = max(3.0, len(kmax_data) * 0.35 + 1.5)
    trend_h = 4.0
    total_h = hmap_h + max(feat_h, 5.0) + trend_h + 1.5  # margins

    fig = plt.figure(figsize=(18, total_h))
    gs  = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[hmap_h, max(feat_h, 5.0), trend_h],
        width_ratios=[1.1, 1],
        hspace=0.45,
        wspace=0.28,
        left=0.07, right=0.97,
        top=0.96, bottom=0.05,
    )

    # A — heatmap spans full width (row 0, both cols)
    ax_A = fig.add_subplot(gs[0, :])
    # B — violin (row 1, left)
    ax_B = fig.add_subplot(gs[1, 0])
    # C — feature ranking (row 1, right)
    ax_C = fig.add_subplot(gs[1, 1])
    # D — precision trend spans full width (row 2, both cols)
    ax_D = fig.add_subplot(gs[2, :])

    print("  Drawing Panel A (heatmap)...")
    _draw_heatmap(ax_A, kmax_data, top_n)

    print("  Drawing Panel B (violin)...")
    _draw_violin(ax_B, kmax_data, top_n)

    print("  Drawing Panel C (feature ranking)...")
    _draw_feature_ranking(ax_C, kmax_data, core_features, top_n, feat_col, sep)

    print(f"  Drawing Panel D ({panel_d_style})...")
    _PANEL_D_STYLES[panel_d_style](ax_D, kmax_data, top_n)

    # Horizontal dividers between rows
    for y_frac in [1 - hmap_h / total_h, trend_h / total_h]:
        fig.add_artist(plt.Line2D(
            [0.03, 0.97], [y_frac, y_frac],
            transform=fig.transFigure,
            color="#cccccc", lw=0.8, ls="--",
        ))

    fig.suptitle(
        "Brute-Force RF — Feature Combination Analysis Across kmax",
        fontsize=13, fontweight="bold", y=0.995,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg", "png"):
        out = outdir / f"assembled_figure_D_{panel_d_style}.{fmt}"
        fig.savefig(out, dpi=dpi, format=fmt)
        print(f"  Saved: {out}")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Assembled publication figure: heatmap, violin, "
                    "feature ranking, precision trend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--kmax-dir",      type=Path, required=True)
    ap.add_argument("--core-features", nargs="+", default=[])
    ap.add_argument("--top-n",         type=int,  default=10)
    ap.add_argument("--min-combos",    type=int,  default=10)
    ap.add_argument("--feat-col",      type=str,  default="features")
    ap.add_argument("--sep",           type=str,  default="|")
    ap.add_argument("--outdir",        type=Path, default=None)
    ap.add_argument("--dpi",           type=int,  default=300)
    args = ap.parse_args()

    outdir = args.outdir or args.kmax_dir

    print("=" * 65)
    print("  ASSEMBLED FIGURE — Panels A, B, C, D")
    print("=" * 65)
    print(f"\nScanning: {args.kmax_dir}")

    kmax_data = _discover(
        args.kmax_dir,
        min_combos=args.min_combos,
        feat_col=args.feat_col,
    )

    if not kmax_data:
        print("ERROR: no valid kmax folders found.", file=sys.stderr)
        return 1

    print(f"\n  kmax loaded  : {sorted(kmax_data.keys())}")
    print(f"  top-N        : {args.top_n}")
    print(f"  core features: {len(args.core_features)}")

    print("\nAssembling figure — Panel D: scatter + median...")
    assemble_figure(
        kmax_data     = kmax_data,
        core_features = set(args.core_features),
        top_n         = args.top_n,
        feat_col      = args.feat_col,
        sep           = args.sep,
        outdir        = outdir,
        dpi           = args.dpi,
        panel_d_style = "scatter",
    )


    print(f"\n{'='*65}")
    print(f"  DONE  —  {outdir}")
    print(f"{'='*65}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
