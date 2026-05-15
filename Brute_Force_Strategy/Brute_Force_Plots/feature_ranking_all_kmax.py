"""
feature_ranking_all_kmax.py — Rank features by weighted frequency across all kmax
==================================================================================

Auto-detects all kmax subfolders in --kmax-dir, loads ranked combo CSVs,
computes rank-weighted feature scores across all kmax values, and plots
a horizontal bar chart with core features highlighted.

Skips any kmax folder with fewer than --min-combos combos (default: 10).

Usage:
    python feature_ranking_all_kmax.py \\
        --kmax-dir   Figures/ \\
        --core-features fold_change fisher_odds_ratio binom_neg_log10_p \\
                        site_enrichment norm_edits log_true_signal \\
                        site_fraction avg_cum_expt_edit_pct site_edit_sd \\
                        mean_expt_max_site_rate std_expt_max_site_rate \\
        --top-n      10 \\
        --outdir     figures/feature_ranking/

Flags:
    --kmax-dir       Root dir containing kmax1/, kmax2/, ... subfolders (required)
    --core-features  Feature names to highlight in blue (space-separated)
    --top-n          Top-N combos per kmax to use (default: 10)
    --min-combos     Skip kmax with fewer than this many combos (default: 10)
    --outdir         Output directory (default: same as --kmax-dir)
    --feat-col       Column containing feature list (default: features)
    --sep            Separator between features (default: |)
    --dpi            Output resolution (default: 300)
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# AUTO-DETECT kmax FOLDERS
# ============================================================================

def _discover_kmax_csvs(kmax_dir: Path, min_combos: int,
                         feat_col: str) -> dict:
    """
    Scan kmax_dir for subdirs matching kmax{N}/ and load their ranked CSVs.
    Returns dict {kmax_int -> DataFrame}, sorted by kmax.
    Skips kmax with fewer than min_combos rows.
    """
    pattern = re.compile(r"^kmax(\d+)$", re.IGNORECASE)
    found   = {}

    for subdir in sorted(kmax_dir.iterdir()):
        if not subdir.is_dir():
            continue
        m = pattern.match(subdir.name)
        if not m:
            continue
        k = int(m.group(1))

        # Find the CSV — accept any file matching kmax*_ranked_combos.csv
        csvs = sorted(subdir.glob("kmax*ranked_combos.csv"))
        if not csvs:
            csvs = sorted(subdir.glob("*.csv"))
        if not csvs:
            print(f"  [skip] kmax={k}: no CSV found in {subdir}")
            continue

        csv_path = csvs[0]
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  [skip] kmax={k}: could not read {csv_path.name}: {exc}")
            continue

        if len(df) < min_combos:
            print(f"  [skip] kmax={k}: only {len(df)} combos < min_combos={min_combos}")
            continue

        if feat_col not in df.columns:
            print(f"  [skip] kmax={k}: column '{feat_col}' not found in {csv_path.name}")
            continue

        found[k] = df
        print(f"  kmax={k:2d}: {len(df):,} combos  ({csv_path.name})")

    return dict(sorted(found.items()))


# ============================================================================
# WEIGHTED SCORE COMPUTATION
# ============================================================================

def compute_weighted_scores(kmax_data: dict, top_n: int,
                             feat_col: str, sep: str) -> pd.DataFrame:
    """
    For each kmax, take top_n combos and compute rank-weighted feature scores.
    Weight for rank R in top-N: (N - R + 1) / N
    rank-1 → 1.0, rank-N → 1/N
    """
    scores      = defaultdict(float)
    counts      = defaultdict(int)
    kmax_scores = {k: defaultdict(float) for k in kmax_data}

    for kmax, df in kmax_data.items():
        top_df = df.head(top_n).reset_index(drop=True)
        n      = len(top_df)

        for idx, row in top_df.iterrows():
            rank   = idx + 1
            weight = (n - rank + 1) / n

            raw  = str(row.get(feat_col, ""))
            feats = [f.strip() for f in raw.split(sep) if f.strip()]

            for feat in feats:
                scores[feat]            += weight
                counts[feat]            += 1
                kmax_scores[kmax][feat] += weight

    rows = []
    for feat in sorted(scores.keys()):
        rec = {
            "feature":          feat,
            "weighted_score":   round(scores[feat], 4),
            "appearance_count": counts[feat],
        }
        for k in sorted(kmax_data.keys()):
            rec[f"kmax{k}_score"] = round(kmax_scores[k].get(feat, 0.0), 4)
        rows.append(rec)

    result = (
        pd.DataFrame(rows)
        .sort_values("weighted_score", ascending=False)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", result.index + 1)
    return result


# ============================================================================
# PLOT
# ============================================================================

def draw_feature_ranking(scores_df: pd.DataFrame, core_features: set,
                          top_n: int, kmax_list: list,
                          outdir: Path, dpi: int) -> None:

    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Georgia", "DejaVu Serif", "Times New Roman"],
        "font.size":         10,
        "axes.linewidth":    1.2,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.frameon":    False,
    })

    n_feats = len(scores_df)
    fig_h   = max(6, n_feats * 0.30 + 2)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    colours = [
        "#2C6E91" if f in core_features else "#B0C4D8"
        for f in scores_df["feature"]
    ]
    y_pos  = np.arange(n_feats)
    values = scores_df["weighted_score"].values

    bars = ax.barh(y_pos, values, color=colours,
                   edgecolor="none", height=0.72, zorder=3)

    max_val = float(values.max()) if len(values) else 1.0
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha="left",
            fontsize=7.5, color="#333333",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(scores_df["feature"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Rank-Weighted Score", labelpad=7)

    kmax_range = f"kmax {min(kmax_list)}–{max(kmax_list)}"
    ax.set_title(
        f"Feature Ranking by Rank-Weighted Frequency\n"
        f"Top {top_n} combos per kmax  [{kmax_range}]",
        fontweight="bold", pad=10,
    )
    ax.grid(axis="x", ls="--", lw=0.6, alpha=0.45, zorder=1)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    n_core = sum(1 for f in scores_df["feature"] if f in core_features)
    legend_elements = [
        mpatches.Patch(facecolor="#2C6E91",
                       label=f"Core feature  (n={n_core})"),
        mpatches.Patch(facecolor="#B0C4D8",
                       label=f"Variable feature  (n={n_feats - n_core})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Rank numbers on right axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"#{int(r)}" for r in scores_df["rank"]],
        fontsize=7.5, color="#888888",
    )
    for spine in ("top", "right", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.tick_params(right=False)

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg", "png"):
        out = outdir / f"feature_ranking_all_kmax.{fmt}"
        fig.savefig(out, dpi=dpi, format=fmt, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rank features by weighted frequency across all kmax.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--kmax-dir",       type=Path, required=True,
                    help="Root dir containing kmax1/, kmax2/, ... subfolders")
    ap.add_argument("--core-features",  nargs="+", default=[],
                    help="Feature names to highlight in blue")
    ap.add_argument("--top-n",          type=int,  default=10,
                    help="Top-N combos per kmax (default: 10)")
    ap.add_argument("--min-combos",     type=int,  default=10,
                    help="Skip kmax with fewer than this many combos (default: 10)")
    ap.add_argument("--outdir",         type=Path, default=None,
                    help="Output directory (default: --kmax-dir)")
    ap.add_argument("--feat-col",       type=str,  default="features",
                    help="Column with feature list (default: features)")
    ap.add_argument("--sep",            type=str,  default="|",
                    help="Separator between features (default: |)")
    ap.add_argument("--dpi",            type=int,  default=300)
    args = ap.parse_args()

    outdir = args.outdir or args.kmax_dir

    print("=" * 65)
    print("  FEATURE RANKING — All kmax")
    print("=" * 65)
    print(f"\nScanning: {args.kmax_dir}")

    kmax_data = _discover_kmax_csvs(
        args.kmax_dir, min_combos=args.min_combos, feat_col=args.feat_col
    )

    if not kmax_data:
        print("ERROR: no valid kmax folders found.", file=sys.stderr)
        return 1

    print(f"\n  kmax values loaded : {sorted(kmax_data.keys())}")
    print(f"  top-N per kmax     : {args.top_n}")

    core_features = set(args.core_features)
    if core_features:
        print(f"  Core features ({len(core_features)}): {sorted(core_features)}")

    print(f"  Weighting: rank-1 = 1.0, rank-{args.top_n} = {1/args.top_n:.3f}")

    print("\nComputing weighted scores...")
    scores_df = compute_weighted_scores(
        kmax_data, top_n=args.top_n,
        feat_col=args.feat_col, sep=args.sep,
    )

    print(f"\n  {len(scores_df)} unique features found")
    print(f"\n  {'Rank':<5} {'Score':>7}  {'Count':>5}  Feature")
    print(f"  {'-'*52}")
    for _, row in scores_df.head(20).iterrows():
        marker = " *" if row["feature"] in core_features else ""
        print(f"  {int(row['rank']):<5} {float(row['weighted_score']):>7.3f}  "
              f"{int(row['appearance_count']):>5}  {row['feature']}{marker}")
    if len(scores_df) > 20:
        print(f"  ... ({len(scores_df) - 20} more in CSV)")
    print("  (* = core feature)")

    csv_path = outdir / "feature_ranking_all_kmax.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    print("\nDrawing plot...")
    draw_feature_ranking(
        scores_df=scores_df,
        core_features=core_features,
        top_n=args.top_n,
        kmax_list=list(kmax_data.keys()),
        outdir=outdir,
        dpi=args.dpi,
    )

    print(f"\n{'='*65}")
    print(f"  DONE  —  {outdir}")
    print(f"{'='*65}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
