"""
kmax_best_combos.py — Fetch best performing combos per kmax from a brute-force run
=======================================================================

For every combo in the results parquet (all kmax values by default, or a
specified subset), reads the per-test-dataset
heatmap files, drops self-evaluation rows (test==train), and averages precision
across all remaining test datasets.

Outputs per kmax:
  - A heatmap PNG per top combo : rows = test datasets, cols = precision thresholds
  - A ranked CSV               : all combos for that kmax sorted by avg test-only precision

Usage:
    python kmax_best_combos.py \\
        --parquet     results_index.parquet \\
        --results-dir /path/to/brute_force_output/ \\
        --outdir      kmax_analysis/ \\
        --top-n       20 \\
        --core-feats  9

    # Specific kmax values only:
    python kmax_best_combos.py \\
        --parquet     results_index.parquet \\
        --results-dir /path/to/brute_force_output/ \\
        --kmax        1 3    # only kmax=1 and kmax=3

Flags:
    --parquet       Path to old results_index.parquet (required)
    --results-dir   Root dir of old brute-force run containing
                    {n_trees}/{n_features}/{combo_hash}/ subfolders (required)
    --outdir        Where to save outputs (default: kmax4_analysis/)
    --top-n         How many top combos to produce heatmaps for (default: 20)
    --rank-by       Summary metric for ranking: avg_precision (mean across all
                    thresholds) or any single threshold e.g. precision_at_10pct
                    (default: avg_precision)
    --n-trees       Filter to a specific tree count (default: use all)
    --core-feats    Number of core features; kmax=4 means n_features=core+4
                    (default: 9)
    --dpi           Heatmap PNG resolution (default: 150)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# CONSTANTS
# ============================================================================

HEATMAP_THRESHOLDS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
HM_COLS            = [f"precision_at_{p}pct" for p in HEATMAP_THRESHOLDS]
HM_LABELS          = [f"{p}%" for p in HEATMAP_THRESHOLDS]

_COMBO_DIR_PATTERNS = (
    "{n_trees}/{n_features}/{combo_hash}",
    "n_trees_{n_trees}/n_features_{n_features}/{combo_hash}",
    "trees_{n_trees}/features_{n_features}/{combo_hash}",
    "{n_trees}_trees/{n_features}_features/{combo_hash}",
    "{combo_hash}",
)
_HEATMAP_FILE_NAMES = (
    "heatmap_data.csv", "heatmap.tsv", "heatmap.csv",
    "heatmap.parquet", "precision_heatmap.tsv", "precision_heatmap.csv",
)


# ============================================================================
# DIRECTORY HELPERS (mirrors rank_top_combos_with_heatmaps.py)
# ============================================================================

def _find_combo_dir(results_dir: Path, n_trees: int,
                    n_features: int, combo_hash: str) -> Optional[Path]:
    for pat in _COMBO_DIR_PATTERNS:
        candidate = results_dir / pat.format(
            n_trees=n_trees, n_features=n_features, combo_hash=combo_hash)
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _find_heatmap_file(train_dir: Path) -> Optional[Path]:
    for name in _HEATMAP_FILE_NAMES:
        p = train_dir / name
        if p.exists():
            return p
    return None


def _enumerate_train_dirs(combo_dir: Path) -> list[Path]:
    if not combo_dir or not combo_dir.exists():
        return []
    skip = {"__pycache__", ".cache"}

    def _children(d):
        return sorted(p for p in d.iterdir()
                      if p.is_dir() and p.name not in skip)

    direct = _children(combo_dir)
    if not direct:
        return []
    for d in direct:
        if _find_heatmap_file(d) is not None:
            return direct
    grandchildren = []
    for wrapper in direct:
        for gc in _children(wrapper):
            if _find_heatmap_file(gc) is not None:
                grandchildren.append(gc)
    return grandchildren


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


# ============================================================================
# HEATMAP DATA LOADING
# ============================================================================

def _load_test_matrix(combo_hash: str, n_trees: int, n_features: int,
                      results_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load and merge all per-train-dataset heatmap files for one combo.

    For each training dataset subfolder:
      - Read heatmap_data.csv
      - Drop the row where test_dataset == train subfolder name (self-eval)
      - Keep remaining test-only rows

    Returns a DataFrame with:
      - index   : test_dataset name
      - columns : precision_at_10pct ... precision_at_100pct
      (averaged across training datasets when multiple exist)

    Returns None if no heatmap files are found.
    """
    combo_dir = _find_combo_dir(results_dir, n_trees, n_features, combo_hash)
    if combo_dir is None:
        return None

    train_dirs = _enumerate_train_dirs(combo_dir)
    if not train_dirs:
        return None

    all_rows: list[pd.DataFrame] = []

    for train_dir in train_dirs:
        hm_file = _find_heatmap_file(train_dir)
        if hm_file is None:
            continue

        try:
            raw = _read_table(hm_file)
        except Exception:
            continue

        # Identify dataset column
        if "test_dataset" in raw.columns:
            ds_col = "test_dataset"
        else:
            ds_col = raw.columns[0]

        # Identify precision columns
        pct_cols = [c for c in HM_COLS if c in raw.columns]
        if not pct_cols:
            pct_cols = [c for c in raw.columns
                        if c != ds_col
                        and pd.api.types.is_numeric_dtype(raw[c])]
        if not pct_cols:
            continue

        # Drop self-evaluation row
        train_name = train_dir.name
        test_rows  = raw[raw[ds_col].astype(str) != train_name].copy()
        test_rows  = test_rows.set_index(ds_col)[pct_cols]
        for c in pct_cols:
            test_rows[c] = pd.to_numeric(test_rows[c], errors="coerce")

        all_rows.append(test_rows)

    if not all_rows:
        return None

    # Average across training datasets per test dataset
    combined = pd.concat(all_rows)
    matrix   = combined.groupby(level=0).mean()

    # Standardise column names to HM_COLS order, fill missing with NaN
    ordered = pd.DataFrame(index=matrix.index)
    for col, lbl in zip(HM_COLS, HM_LABELS):
        ordered[lbl] = matrix[col] if col in matrix.columns else np.nan

    return ordered


# ============================================================================
# HEATMAP DRAWING (same style as rank_top_combos_with_heatmaps.py)
# ============================================================================

def _draw_heatmap(matrix: pd.DataFrame, title: str,
                  out_png: Path, dpi: int = 150) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "precision_red",
        ["#FFFFFF", "#FCEBEB", "#F5B3B2", "#E24B4A", "#A32D2D", "#5A1717"],
    )

    n_rows, n_cols = matrix.shape
    fig_w = max(6, 0.9 * n_cols + 2)
    fig_h = max(3, 0.45 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix.values.astype(float),
                   aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(matrix.columns, rotation=0, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_xlabel("Precision @ K%", fontsize=10)
    ax.set_ylabel("Test dataset",    fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix.values[i, j]
            if pd.isna(val):
                continue
            text_color = "#FFFFFF" if float(val) > 0.65 else "#2C2C2A"
            ax.text(j, i, f"{float(val):.2f}",
                    ha="center", va="center",
                    fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Precision", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _safe_name(text: str, maxlen: int = 60) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in "-_.":
            out.append(ch)
        elif ch in " /|":
            out.append("_")
    s = "".join(out).strip("._")
    return s[:maxlen]



# ============================================================================
# PANEL E — Precision trend across thresholds, one line per kmax
# ============================================================================

def _draw_panel_e(kmax_ranked: dict, top_k: int, outdir: Path,
                  dpi: int = 150) -> None:
    """
    Single-plot Panel E:
      X-axis : precision threshold (10, 20, ... 100)
      Y-axis : mean precision across top-K combos for that kmax
      Lines  : one per kmax, coloured distinctly
      Trend  : numpy polyfit degree-1 trendline per kmax (dashed)

    kmax_ranked : dict mapping kmax (int) -> ranked DataFrame
                  (must have precision_at_10%, precision_at_20%, ... cols
                   and avg_precision col)
    """
    _KMAX_COLOURS = ["#264653", "#2A9D8F", "#E9C46A", "#E76F51",
                     "#9B5DE5", "#F15BB5", "#00BBF9", "#00F5D4"]

    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Georgia", "DejaVu Serif", "Times New Roman"],
        "font.size":         11,
        "axes.linewidth":    1.2,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.frameon":    False,
    })

    x_vals  = list(HEATMAP_THRESHOLDS)          # [10, 20, ... 100]
    x_ticks = x_vals

    fig, ax = plt.subplots(figsize=(11, 5))

    kmax_list = sorted(kmax_ranked.keys())
    pal       = {k: _KMAX_COLOURS[i % len(_KMAX_COLOURS)]
                 for i, k in enumerate(kmax_list)}

    for kmax in kmax_list:
        ranked  = kmax_ranked[kmax]
        top_df  = ranked.head(top_k)
        colour  = pal[kmax]

        # Mean precision at each threshold across top-K combos
        means = []
        sds   = []
        for lbl in HM_LABELS:
            col = f"precision_at_{lbl}"
            if col in top_df.columns:
                vals = pd.to_numeric(top_df[col], errors="coerce").dropna()
                means.append(float(vals.mean()) if len(vals) else np.nan)
                sds.append(float(vals.std())   if len(vals) else np.nan)
            else:
                means.append(np.nan)
                sds.append(np.nan)

        means = np.array(means)
        sds   = np.array(sds)

        # Plot mean line with SD band
        ax.plot(x_vals, means,
                color=colour, lw=2.2, marker="o", ms=5,
                label=f"kmax={kmax}", zorder=3)
        ax.fill_between(x_vals, means - sds, means + sds,
                        color=colour, alpha=0.10, zorder=2)

        # Trendline — linear fit over valid points
        valid = ~np.isnan(means)
        if valid.sum() >= 2:
            x_fit = np.array(x_vals)[valid]
            y_fit = means[valid]
            coef  = np.polyfit(x_fit, y_fit, 1)
            poly  = np.poly1d(coef)
            x_ext = np.linspace(min(x_vals), max(x_vals), 200)
            ax.plot(x_ext, poly(x_ext),
                    color=colour, lw=1.2, ls="--", alpha=0.7, zorder=2)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}%" for x in x_ticks], fontsize=9)
    ax.set_xlabel("Precision Threshold", labelpad=7)
    ax.set_ylabel("Mean Precision (test-only)", labelpad=7)
    ax.set_title(
        f"Panel E — Precision Trend Across Thresholds  "
        f"[top {top_k} combos per kmax, test-only]",
        fontweight="bold", pad=10,
    )
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.legend(title="kmax", title_fontsize=9, loc="upper right")
    ax.grid(axis="y", ls="--", lw=0.6, alpha=0.45, zorder=1)
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.3, zorder=1)

    plt.tight_layout()

    for fmt in ("pdf", "svg", "png"):
        out = outdir / f"PanelE_precision_trend.{fmt}"
        fig.savefig(out, dpi=dpi, format=fmt, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.close(fig)

# ============================================================================
# MAIN LOGIC
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rank best kmax=4 combos by avg test-only precision and "
                    "produce heatmap PNGs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--parquet",     type=Path, required=True,
                    help="Path to main results_index.parquet (kmax 1-3)")
    ap.add_argument("--old-parquet", type=Path, default=None,
                    help="Path to old results_index.parquet containing kmax=4 rows. "
                         "When provided, kmax=4 combos are sourced from here instead.")
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Root dir of old brute-force run")
    ap.add_argument("--outdir",      type=Path, default=Path("kmax4_analysis"),
                    help="Output directory (default: kmax4_analysis/)")
    ap.add_argument("--top-n",       type=int,  default=20,
                    help="How many top combos to produce heatmaps for (default: 20)")
    ap.add_argument("--rank-by",     type=str,  default="avg_precision",
                    help="Ranking metric: avg_precision or precision_at_Xpct "
                         "(default: avg_precision)")
    ap.add_argument("--n-trees",     type=int,  default=None,
                    help="Filter to a specific n_trees value (default: all)")
    ap.add_argument("--core-feats",  type=int,  default=9,
                    help="Number of core features; kmax=N → n_features=core+N "
                         "(default: 9)")
    ap.add_argument("--kmax",        type=int,  nargs="+", default=None,
                    help="Which kmax values to analyse (default: all found). "
                         "e.g. --kmax 1 2 3 4")
    ap.add_argument("--dpi",         type=int,  default=150,
                    help="Heatmap PNG resolution (default: 150)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load parquet -------------------------------------------------------
    print("=" * 65)
    print("  kmax BEST COMBO ANALYSIS")
    print("=" * 65)
    print(f"\nLoading: {args.parquet}")

    try:
        df = pd.read_parquet(args.parquet)
    except Exception as exc:
        print(f"ERROR: could not read parquet: {exc}", file=sys.stderr)
        return 1

    if args.n_trees is not None:
        df = df[df["n_trees"] == args.n_trees].copy()
        print(f"  Filtered to n_trees={args.n_trees}")

    # If old parquet provided, pull kmax=4 rows from it
    if args.old_parquet is not None:
        print(f"\nLoading old parquet: {args.old_parquet}")
        try:
            old_df = pd.read_parquet(args.old_parquet)
        except Exception as exc:
            print(f"ERROR: could not read old parquet: {exc}", file=sys.stderr)
            return 1
        if args.n_trees is not None:
            old_df = old_df[old_df["n_trees"] == args.n_trees].copy()
        kmax4_nf = args.core_feats + 4
        old_kmax4 = old_df[old_df["n_features"] == kmax4_nf].copy()
        if old_kmax4.empty:
            print(f"  [WARN] No kmax=4 rows found in old parquet "
                  f"(n_features={kmax4_nf}). "
                  f"Available: {sorted(old_df['n_features'].unique())}")
        else:
            # Remove any kmax=4 rows from main df to avoid duplicates
            df = df[df["n_features"] != kmax4_nf].copy()
            df = pd.concat([df, old_kmax4], ignore_index=True)
            print(f"  Appended {len(old_kmax4):,} kmax=4 rows from old parquet.")

    # Determine which kmax values to process
    available_kmax = sorted(
        (df["n_features"] - args.core_feats).unique().tolist()
    )
    if args.kmax is not None:
        kmax_to_run = sorted(k for k in args.kmax if k in available_kmax)
        missing = [k for k in args.kmax if k not in available_kmax]
        if missing:
            print(f"  [WARN] kmax values not found in parquet: {missing}")
    else:
        kmax_to_run = available_kmax

    if not kmax_to_run:
        print("ERROR: no kmax values to process.", file=sys.stderr)
        return 1

    print(f"  Available kmax : {available_kmax}")
    print(f"  Processing     : {kmax_to_run}")
    print(f"  Results dir    : {args.results_dir}")
    print(f"  (self-evaluation rows excluded from all precision values)\n")

    rank_col    = args.rank_by
    kmax_ranked = {}   # kmax -> ranked DataFrame (for Panel E)

    # ---- Loop over each kmax ------------------------------------------------
    for kmax in kmax_to_run:
        n_features = args.core_feats + kmax
        df_sub     = df[df["n_features"] == n_features].copy()

        print(f"\n{'='*65}")
        print(f"  kmax={kmax}  (n_features={n_features})  "
              f"--  {len(df_sub):,} combos")
        print(f"{'='*65}")

        # Per-kmax output dirs
        kmax_dir     = args.outdir / f"kmax{kmax}"
        kmax_hm_dir  = kmax_dir / "heatmaps"
        kmax_dir.mkdir(parents=True, exist_ok=True)
        kmax_hm_dir.mkdir(exist_ok=True)

        # -- Load test-only precision for every combo in this kmax -----------
        records   = []
        n_missing = 0

        for idx, (_, row) in enumerate(df_sub.iterrows(), 1):
            combo_hash = str(row["combo_hash"])
            n_trees    = int(row["n_trees"])

            if idx % 200 == 0 or idx == len(df_sub):
                print(f"  Processing {idx}/{len(df_sub)}...", end="\r")

            matrix = _load_test_matrix(combo_hash, n_trees, n_features,
                                       args.results_dir)

            if matrix is None:
                n_missing += 1
                rec = {
                    "combo_hash":        combo_hash,
                    "n_trees":           n_trees,
                    "n_features":        n_features,
                    "variable_features": str(row.get("variable_features", "")),
                    "features":          str(row.get("features", "")),
                    "n_test_datasets":   0,
                    "avg_precision":     np.nan,
                }
                for lbl in HM_LABELS:
                    rec[f"precision_at_{lbl}"] = np.nan
                records.append(rec)
                continue

            avg_per_threshold = matrix.mean()
            overall_avg       = float(avg_per_threshold.mean())

            rec = {
                "combo_hash":        combo_hash,
                "n_trees":           n_trees,
                "n_features":        n_features,
                "variable_features": str(row.get("variable_features", "")),
                "features":          str(row.get("features", "")),
                "n_test_datasets":   len(matrix),
                "avg_precision":     overall_avg,
            }
            for lbl in HM_LABELS:
                rec[f"precision_at_{lbl}"] = (
                    float(avg_per_threshold[lbl])
                    if lbl in avg_per_threshold.index else np.nan
                )
            records.append(rec)

        print()  # clear progress line
        if n_missing:
            print(f"  [WARN] {n_missing}/{len(df_sub)} combos: heatmap files "
                  f"not found. Check --results-dir.")

        # -- Rank -------------------------------------------------------------
        results_df = pd.DataFrame(records)

        rc = rank_col
        if rc not in results_df.columns:
            alt = f"precision_at_{rc}"
            if alt in results_df.columns:
                rc = alt
            else:
                print(f"  [ERROR] --rank-by '{rank_col}' not found for kmax={kmax}. "
                      f"Skipping.", file=sys.stderr)
                continue

        ranked = (
            results_df
            .dropna(subset=[rc])
            .sort_values(rc, ascending=False)
            .reset_index(drop=True)
        )
        ranked.insert(0, "rank", ranked.index + 1)
        kmax_ranked[kmax] = ranked   # store for Panel E

        # -- Print top-N ------------------------------------------------------
        top_n = min(args.top_n, len(ranked))
        print(f"\n  TOP {top_n} combos  (ranked by {rc})")
        print(f"  {'Rank':<5} {'Score':>7}  {'@10%':>6}  {'@20%':>6}  "
              f"{'@50%':>6}  {'@100%':>6}  {'Tests':>5}  Variable Features")
        print(f"  {'-'*63}")

        for _, row in ranked.head(top_n).iterrows():
            vf = str(row["variable_features"])
            if len(vf) > 45:
                vf = vf[:42] + "..."
            print(
                f"  {int(row['rank']):<5} "
                f"{float(row[rc]):>7.4f}  "
                f"{float(row.get('precision_at_10%', 0) or 0):>6.3f}  "
                f"{float(row.get('precision_at_20%', 0) or 0):>6.3f}  "
                f"{float(row.get('precision_at_50%', 0) or 0):>6.3f}  "
                f"{float(row.get('precision_at_100%', 0) or 0):>6.3f}  "
                f"{int(row['n_test_datasets']):>5}  {vf}"
            )
        print(f"  {'='*63}")

        # -- Save ranked CSV --------------------------------------------------
        csv_path = kmax_dir / f"kmax{kmax}_ranked_combos.csv"
        ranked.to_csv(csv_path, index=False)
        print(f"\n  Ranked CSV saved: {csv_path}")

        # -- Draw heatmaps ----------------------------------------------------
        print(f"\n  Drawing heatmaps for top {top_n} combos...")
        n_ok  = 0
        n_err = 0

        for _, row in ranked.head(top_n).iterrows():
            rank_i     = int(row["rank"])
            combo_hash = str(row["combo_hash"])
            n_trees    = int(row["n_trees"])
            vf         = str(row["variable_features"])
            score      = float(row[rc])

            matrix = _load_test_matrix(combo_hash, n_trees, n_features,
                                       args.results_dir)
            if matrix is None or matrix.empty:
                print(f"    [skip] rank #{rank_i}: no heatmap data.")
                n_err += 1
                continue

            out_png = kmax_hm_dir / (
                f"rank{rank_i:02d}__{_safe_name(vf)}__{combo_hash[:10]}.png"
            )
            title = (
                f"kmax={kmax}  |  Rank {rank_i}  |  {rc}={score:.4f}\n"
                f"n_trees={n_trees}  |  variable: {vf}"
            )
            if len(title) > 120:
                title = title[:117] + "..."

            try:
                _draw_heatmap(matrix, title, out_png, dpi=args.dpi)
                print(f"    #{rank_i:02d}  {score:.4f}  → {out_png.name}")
                n_ok += 1
            except Exception as exc:
                print(f"    [warn] rank #{rank_i}: heatmap failed: {exc}")
                n_err += 1

        print(f"\n  Heatmaps saved: {n_ok}  |  Skipped: {n_err}")
        print(f"  Output: {kmax_dir}")

    # ---- Panel E — aggregated trend across all kmax -----------------
    if len(kmax_ranked) > 0:
        print(f"\n[Panel E] Drawing precision trend across thresholds...")
        _draw_panel_e(kmax_ranked, top_k=args.top_n,
                      outdir=args.outdir, dpi=args.dpi)

    # ---- Final summary ------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  ALL DONE")
    print(f"  kmax values processed : {kmax_to_run}")
    print(f"  Output root           : {args.outdir}")
    print(f"{'='*65}\n")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    sys.exit(main())
