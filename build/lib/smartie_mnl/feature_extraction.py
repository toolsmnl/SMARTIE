"""
feature_extraction.py — Feature engineering for the RBP A-to-G prediction pipeline.

Three-condition model
---------------------
    Experiment (exp)   RBP-ADAR fusion: RBP-specific edits + ADAR activity
    Control    (ctrl)  ADAR-only:        baseline ADAR background
    Background (bg)    Cell-line only:   inherent SNPs / spontaneous edits (optional)

Selected feature set (18 features)
------------------------------------
All raw read-count-based quantities use normalised reads (gene reads /
sample total reads per replicate, then averaged) so features are comparable
across replicates and datasets with different sequencing depths.

    Normalisation (2)
        norm_edits              gene edited reads / sample total edited reads (exp)
        site_fraction           gene edit sites / total sample edit sites (exp)

    Differential — exp vs ctrl (2)
        log_true_signal         log2(avg_cum_exp - avg_cum_ctrl); -20 when <= 0
        fold_change             avg_cum_exp / avg_cum_ctrl  (FCM)

    Magnitude (1)
        avg_cum_expt_edit_pct   mean(sum of per-site edit fractions per rep, exp) x100

    Site-level — experiment (5)
        site_enrichment           norm_edits_exp / norm_edits_ctrl
        site_enrichment_edit_rate mean(e_edit_rates) / mean(c_edit_rates)
        mean_expt_max_site_rate   mean of per-gene max per-site edit rate across exp reps
        max_expt_max_site_rate    max of per-gene max per-site edit rate across exp reps
        std_expt_max_site_rate    SD of per-gene max per-site edit rate across exp reps

    Replicate consistency — experiment (2)
        site_edit_sd          mean (across exp reps) of within-gene SD of per-site edit%
        std_expt_n_sites      SD of per-gene number of edit sites across exp reps

    Coverage — experiment (1)
        max_expt_sum_total    max of per-replicate total read coverage at this gene

    Control aggregates (3)
        mean_ctrl_gene_rate   mean per-gene normalised edit rate across ctrl reps
        mean_ctrl_sum_edits   mean of per-replicate raw pooled edit counts in control
        max_ctrl_sum_edits    max of per-replicate raw pooled edit counts in control

    Statistical tests — raw pooled read counts (2)
        binom_neg_log10_p     -log10(binomial p), pooled exp edits vs per-gene ctrl rate
        fisher_odds_ratio     log1p(odds ratio from Fisher exact test, exp vs ctrl)

Public API
----------
    build_feature_matrix(exp_dfs, ctrl_dfs, bg_dfs, gene_names) -> pd.DataFrame
    get_feature_names() -> list[str]
    SELECTED_FEATURES   -> list[str]  (ordered)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

warnings.filterwarnings("ignore")

_EPS = 1e-9

# ── Ordered selected feature list (18 features) ──────────────────────────────
SELECTED_FEATURES: list[str] = [
    "mean_expt_max_site_rate",
    "log_true_signal",
    "std_expt_max_site_rate",
    "avg_cum_expt_edit_pct",
    "norm_edits",
    "site_edit_sd",
    "fold_change",
    "mean_ctrl_sum_edits",
    "fisher_odds_ratio",
    "mean_ctrl_gene_rate",
    "max_expt_max_site_rate",
    "site_enrichment_edit_rate",
    "max_ctrl_sum_edits",
    "std_expt_n_sites",
    "site_enrichment",
    "site_fraction",
    "binom_neg_log10_p",
    "max_expt_sum_total",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    exp_dfs:    list[pd.DataFrame],
    ctrl_dfs:   list[pd.DataFrame],
    bg_dfs:     list[pd.DataFrame] | None = None,
    gene_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the 18-feature matrix for all genes.

    Parameters
    ----------
    exp_dfs    : list of standardised site-level DataFrames for experiment replicates.
                 Each must have columns: gene, edit_count, total_count, edit_rate.
    ctrl_dfs   : same for control replicates
    bg_dfs     : same for background replicates (optional; pass None or [] to omit)
    gene_names : optional gene subset; defaults to union of all genes in all files

    Returns
    -------
    pd.DataFrame  index = gene name, columns = SELECTED_FEATURES (18 columns)
    """
    if not exp_dfs:
        raise ValueError("At least one experiment replicate is required.")
    if not ctrl_dfs:
        raise ValueError("At least one control replicate is required.")

    bg_dfs = bg_dfs or []

    # ── Pre-group each replicate (O(sites), not O(genes × sites)) ────────────
    exp_stats  = [_precompute_gene_stats(df) for df in exp_dfs]
    ctrl_stats = [_precompute_gene_stats(df) for df in ctrl_dfs]

    # ── Gene universe ──────────────────────────────────────────────────────────
    if gene_names is None:
        all_genes: set[str] = set()
        for s in exp_stats + ctrl_stats:
            all_genes.update(s.keys())
        gene_names = sorted(all_genes)

    n_genes = len(gene_names)
    print(f"  Building feature matrix: {n_genes:,} genes | "
          f"{len(exp_dfs)} exp rep(s) | {len(ctrl_dfs)} ctrl rep(s)")

    # ── Sample-level normalisation denominators (one per replicate) ───────────
    # Normalised reads = gene reads / sample total reads for that replicate.
    # Computing per replicate keeps comparisons fair across different depths.
    exp_sample_edits = [float(df["edit_count"].sum()) for df in exp_dfs]
    exp_sample_reads = [float(df["total_count"].sum()) for df in exp_dfs]
    exp_sample_sites = [len(df)                         for df in exp_dfs]
    ctrl_sample_edits= [float(df["edit_count"].sum()) for df in ctrl_dfs]
    ctrl_sample_reads= [float(df["total_count"].sum()) for df in ctrl_dfs]

    # Precompute per-gene site edit rate SD for each experiment replicate
    # (SD of per-site edit% within the gene — used for site_edit_sd feature)
    e_site_sd_by_rep = [
        (df.groupby("Name")["edit_rate"].std() * 100.0).fillna(0.0)
        for df in exp_dfs
    ]

    # ── Build one row per gene ────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []

    for gene in gene_names:
        feat: dict[str, Any] = {}

        # Per-replicate vectors for this gene
        e_stats = [s.get(gene, _ZERO) for s in exp_stats]
        c_stats = [s.get(gene, _ZERO) for s in ctrl_stats]

        e_edit_rates = np.array([s["edit_rate"]            for s in e_stats], float)
        e_cum_rates  = np.array([s["cumulative_edit_rate"] for s in e_stats], float)
        e_n_sites    = np.array([s["n_sites"]              for s in e_stats], float)
        e_coverages  = np.array([s["log_coverage"]         for s in e_stats], float)
        e_edit_sums  = np.array([s["edit_count_sum"]       for s in e_stats], float)
        e_total_sums = np.array([s["total_count_sum"]      for s in e_stats], float)
        e_max_site_rates = np.array([s["max_site_rate"]    for s in e_stats], float)

        c_edit_rates = np.array([s["edit_rate"]            for s in c_stats], float)
        c_cum_rates  = np.array([s["cumulative_edit_rate"] for s in c_stats], float)
        c_n_sites    = np.array([s["n_sites"]              for s in c_stats], float)
        c_edit_sums  = np.array([s["edit_count_sum"]       for s in c_stats], float)
        c_total_sums = np.array([s["total_count_sum"]      for s in c_stats], float)

        # ── Normalised edit fractions per replicate ───────────────────────────
        # norm_edits_r = gene edited reads / sample total edited reads (rep r)
        e_norm_edits = np.array([
            e_edit_sums[i] / (exp_sample_edits[i] + _EPS)
            for i in range(len(exp_dfs))
        ], float)
        c_norm_edits = np.array([
            c_edit_sums[i] / (ctrl_sample_edits[i] + _EPS)
            for i in range(len(ctrl_dfs))
        ], float)

        # norm_total_r = gene total reads / sample total reads (rep r)
        e_norm_total = np.array([
            e_total_sums[i] / (exp_sample_reads[i] + _EPS)
            for i in range(len(exp_dfs))
        ], float)

        # site_fraction_r = gene sites / total sample sites (rep r)
        e_site_frac = np.array([
            e_n_sites[i] / (exp_sample_sites[i] + _EPS)
            for i in range(len(exp_dfs))
        ], float)

        mean_norm_edits_exp  = float(np.mean(e_norm_edits))
        mean_norm_edits_ctrl = float(np.mean(c_norm_edits))
        mean_norm_total_exp  = float(np.mean(e_norm_total))
        mean_e_cum = float(np.mean(e_cum_rates))
        mean_c_cum = float(np.mean(c_cum_rates))

        # ── 1. NORMALISATION ─────────────────────────────────────────────────
        feat["norm_edits"]     = mean_norm_edits_exp
        feat["norm_gene_rate"] = mean_norm_edits_exp / (mean_norm_total_exp + _EPS)
        feat["site_fraction"]  = float(np.mean(e_site_frac))

        # ── 2. DIFFERENTIAL ──────────────────────────────────────────────────
        # All differential features operate on the ×100 scaled cumulative edit
        # percentages so they are consistent with avg_cum_expt_edit_pct and
        # avg_cum_ctrl_edit_pct reported in the output.
        mean_e_pct = mean_e_cum * 100.0
        mean_c_pct = mean_c_cum * 100.0

        diff_pct = max(mean_e_pct - mean_c_pct, 0.0)
        feat["log_true_signal"] = np.log2(diff_pct) if diff_pct > 0 else -20.0

        # fold_change: when ctrl = 0 use 1 as denominator (not epsilon),
        # giving fold_change = avg_cum_expt_edit_pct rather than a huge artefact.
        feat["fold_change"] = mean_e_pct / (mean_c_pct if mean_c_pct > 0 else 1.0)

        feat["signal_baseline"] = (
            np.log2(diff_pct / mean_c_pct)
            if diff_pct > 0 and mean_c_pct > 0
            else (np.log2(diff_pct) if diff_pct > 0 else -20.0)
        )

        # fc_cv: CV of cumulative edit% WITHIN experiment replicates (×100 scale),
        # measuring replicate consistency of editing, not between-condition difference.
        n_exp = len(exp_dfs)
        e_cum_pct = e_cum_rates * 100.0
        feat["fc_cv"] = (
            float(np.std(e_cum_pct) / (mean_e_pct + _EPS))
            if n_exp > 1 else 0.0
        )
        feat["enrichment_score"] = float(np.log2(max(feat["fold_change"], 0.0) + 1.0))

        # ── 3. MAGNITUDE ─────────────────────────────────────────────────────
        feat["avg_cum_expt_edit_pct"] = mean_e_cum * 100.0
        feat["avg_cum_ctrl_edit_pct"] = mean_c_cum * 100.0

        # ── 4. SITE-LEVEL ────────────────────────────────────────────────────
        # site_enrichment: normalised edited reads exp / normalised edited reads ctrl
        feat["site_enrichment"] = mean_norm_edits_exp / (mean_norm_edits_ctrl + _EPS)

        # site_enrichment_edit_rate: same idea but on per-site edit RATES rather
        # than normalised edit counts. Sensitive to per-base editing intensity
        # rather than total signal volume — captures genes where the editing
        # is concentrated rather than just abundant.
        feat["site_enrichment_edit_rate"] = (
            float(np.mean(e_edit_rates)) / (float(np.mean(c_edit_rates)) + _EPS)
        )

        # weighted_site_strength: coverage-weighted edit rate per gene (exp)
        # sum(edit_count) / sum(total_count) x 100, averaged across exp replicates
        ws_vals = [
            e_edit_sums[i] / (e_total_sums[i] + _EPS) * 100.0
            for i in range(len(exp_dfs))
        ]
        feat["weighted_site_strength"] = float(np.mean(ws_vals))

        # ── 5. EXPERIMENT AGGREGATES ─────────────────────────────────────────
        mean_e_rate = float(np.mean(e_edit_rates))
        feat["exp_n_sites_mean"]   = float(np.mean(e_n_sites))
        feat["exp_frac_reps_present"] = float(np.sum(e_n_sites > 0)) / len(exp_dfs)
        feat["exp_replicate_cv"]   = (
            float(np.std(e_edit_rates) / (mean_e_rate + _EPS))
            if n_exp > 1 else 0.0
        )

        # mean/std/max of per-gene max per-site edit rate across experiment replicates.
        # Mean captures the strongest site signal averaged over reps; SD captures how
        # reproducibly that peak shows up; max picks up the single hottest replicate.
        feat["mean_expt_max_site_rate"] = float(np.mean(e_max_site_rates))
        feat["std_expt_max_site_rate"]  = (
            float(np.std(e_max_site_rates)) if n_exp > 1 else 0.0
        )
        feat["max_expt_max_site_rate"]  = float(np.max(e_max_site_rates))

        # std_expt_n_sites: replicate consistency in the number of detected sites.
        # Low SD = same sites called in every rep (real signal); high SD = noisy.
        feat["std_expt_n_sites"] = (
            float(np.std(e_n_sites)) if n_exp > 1 else 0.0
        )

        # max_expt_sum_total: max coverage across exp replicates (a high-coverage
        # rep gives more statistical power; useful for weighting downstream).
        feat["max_expt_sum_total"] = float(np.max(e_total_sums))

        # site_edit_sd: SD of per-site edit% within the gene, averaged across
        # experiment replicates. Low SD + high editing = consistent target;
        # high SD = editing driven by a single outlier site (likely noise).
        feat["site_edit_sd"] = float(np.mean([
            float(sd_ser.get(gene, 0.0)) for sd_ser in e_site_sd_by_rep
        ]))

        # ── 6. CONTROL AGGREGATES ────────────────────────────────────────────
        n_ctrl = len(ctrl_dfs)
        mean_c_rate = float(np.mean(c_edit_rates))
        feat["ctrl_edit_rate_mean"]      = mean_c_rate
        feat["mean_ctrl_gene_rate"]      = mean_c_rate  # mean per-gene normalised edit rate across ctrl reps
        feat["ctrl_cumulative_edit_std"] = float(np.std(c_cum_rates)) if n_ctrl > 1 else 0.0
        feat["ctrl_frac_reps_present"]   = float(np.sum(c_n_sites > 0)) / len(ctrl_dfs)
        feat["ctrl_replicate_cv"]        = (
            float(np.std(c_edit_rates) / (mean_c_rate + _EPS))
            if n_ctrl > 1 else 0.0
        )

        # mean/max of raw pooled edit counts in control replicates. High values
        # mark genes with strong baseline editing that need to be exceeded by
        # experiment to count as a real RBP-specific signal.
        feat["mean_ctrl_sum_edits"] = float(np.mean(c_edit_sums))
        feat["max_ctrl_sum_edits"]  = float(np.max(c_edit_sums))

        # ── 8. STATISTICAL TESTS (raw pooled read counts, exp vs ctrl) ──────────
        # Raw read counts are pooled across ALL replicates before testing.
        # This gives the tests real statistical power (counts in the thousands)
        # which normalised-read scaling cannot provide.
        # Normalised reads are used for magnitude/enrichment features above;
        # here we need the actual counts to make Fisher and binomial meaningful.
        exp_edit_pool  = float(np.sum(e_edit_sums))
        exp_total_pool = float(np.sum(e_total_sums))
        ctrl_edit_pool  = float(np.sum(c_edit_sums))
        ctrl_total_pool = float(np.sum(c_total_sums))

        exp_nonedit  = max(exp_total_pool  - exp_edit_pool,  0.0)
        ctrl_nonedit = max(ctrl_total_pool - ctrl_edit_pool, 0.0)

        # Fisher exact: [[exp edited, exp unedited], [ctrl edited, ctrl unedited]]
        try:
            odds_ratio, fisher_p = fisher_exact(
                [[int(exp_edit_pool),  int(exp_nonedit)],
                 [int(ctrl_edit_pool), int(ctrl_nonedit)]],
                alternative="greater",
            )
        except Exception:
            odds_ratio, fisher_p = 1.0, 1.0

        # Binomial: are exp edited reads significantly above the per-gene ctrl rate?
        ctrl_rate = ctrl_edit_pool / ctrl_total_pool if ctrl_total_pool > 0 else 0.0
        try:
            binom_p = binomtest(
                int(exp_edit_pool),
                int(exp_total_pool),
                p=max(ctrl_rate, _EPS),
                alternative="greater",
            ).pvalue
        except Exception:
            binom_p = 1.0

        feat["fisher_neg_log10_p"] = float(-np.log10(max(fisher_p,  1e-300)))
        feat["binom_neg_log10_p"]  = float(-np.log10(max(binom_p,   1e-300)))
        feat["fisher_odds_ratio"]  = float(np.log1p(max(odds_ratio, 0.0)))

        rows.append(feat)

    df_out = (
        pd.DataFrame(rows, index=gene_names)
        [SELECTED_FEATURES]   # enforce column order
        .fillna(0.0)
    )

    # Replace any infinities produced by near-zero denominators, then clip
    # extreme finite values to a range safe for float32 / sklearn RF models.
    df_out = df_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in df_out.columns:
        p99 = df_out[col].abs().quantile(0.99)
        cap = max(p99 * 5, 1e6)   # floor at 1e6 so we never over-clip small features
        df_out[col] = df_out[col].clip(-cap, cap)

    print(f"  Done. Feature matrix: {df_out.shape[0]:,} genes x {df_out.shape[1]} features")
    return df_out


def get_feature_names() -> list[str]:
    """Return the ordered list of the 18 feature column names."""
    return list(SELECTED_FEATURES)


def load_editing_sites(
    path: Path,
    editing_type: str = "AtoG",
    min_edit_pct: float = 0.0,
    min_reads: int = 0,
) -> pd.DataFrame:
    """
    Load a TRIBE/STAMP editing-site file and return a standardised DataFrame.

    Parameters
    ----------
    path          : path to tab-separated editing-site file
    editing_type  : "AtoG" (default) or "CtoT"
    min_edit_pct  : drop sites with edit% below this value (default: 0 = no filter)
    min_reads     : drop sites with total_count below this value (default: 0 = no filter)

    Returns
    -------
    DataFrame with columns: gene, edit_count, total_count, edit_rate
    (gene names are lowercased and whitespace-stripped)
    """
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Gene name column
    gene_col = next(
        (c for c in ["Name", "name", "gene_id", "Gene_id", "gene", "Gene"]
         if c in df.columns), None
    )
    if gene_col is None:
        raise ValueError(f"{path}: no gene name column found")

    # Edit count column
    if editing_type == "AtoG":
        edit_col = next(
            (c for c in ["Editbase_count", "G_count", "g_count", "editbase_count"]
             if c in df.columns), None
        )
        if edit_col is None:
            raise ValueError(f"{path}: no A-to-G edit count column (expected Editbase_count or G_count)")
    elif editing_type == "CtoT":
        edit_col = next(
            (c for c in ["T_count", "t_count", "edit_count"]
             if c in df.columns), None
        )
        if edit_col is None:
            raise ValueError(f"{path}: no C-to-T edit count column (expected T_count)")
    else:
        raise ValueError(f"editing_type must be 'AtoG' or 'CtoT', got '{editing_type}'")

    # Total count column
    total_col = next(
        (c for c in ["Total_count.1", "Total_count", "total_count"]
         if c in df.columns), None
    )
    if total_col is None:
        raise ValueError(f"{path}: no total count column (expected Total_count)")

    for c in (edit_col, total_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[edit_col, total_col, gene_col])
    df = df[df[total_col] > 0].copy()

    df["edit_count"]  = df[edit_col].astype(float)
    df["total_count"] = df[total_col].astype(float)
    df["edit_rate"]   = df["edit_count"] / df["total_count"]
    df["Name"]        = df[gene_col].astype(str).str.strip().str.lower()

    # Apply site-level filters
    if min_reads > 0:
        df = df[df["total_count"] >= min_reads]
    if min_edit_pct > 0:
        df = df[df["edit_rate"] * 100 >= min_edit_pct]

    return df.copy()


# ---------------------------------------------------------------------------
# Compatibility layer for train_model.py and predict.py
# ---------------------------------------------------------------------------
# These symbols match the API of the older feature_extraction.py so that
# train_model.py and predict.py work without modification.  All feature
# computation still goes through build_feature_matrix(), which carries all
# the corrected logic (×100 scaling, fold_change denominator=1, signal_baseline
# fix, raw read counts for statistical tests).

# Feature column list — alias of SELECTED_FEATURES
FEATURE_COLUMNS: list[str] = SELECTED_FEATURES

# Feature sets — all names map to the single 18-feature schema
FEATURE_SETS: dict[str, list[str]] = {
    "full18":    SELECTED_FEATURES,
    "full12":    SELECTED_FEATURES,   # legacy name → same features
    "full24":    SELECTED_FEATURES,   # legacy name → same features
    "full22":    SELECTED_FEATURES,   # legacy name → same features
    "reduced17": SELECTED_FEATURES,   # legacy name → same features
    "core8":     SELECTED_FEATURES,   # legacy name → same features
}
VALID_FEATURE_SETS: list[str] = list(FEATURE_SETS.keys())


def get_feature_columns(feature_set: str = "full18") -> list[str]:
    """Return the 18-feature column list (ignores feature_set name for compatibility)."""
    return list(SELECTED_FEATURES)


def build_features(
    expt_dfs:   list[pd.DataFrame],
    ctrl_dfs:   list[pd.DataFrame],
    gene_names: list[str],
    **kwargs,
) -> pd.DataFrame:
    """
    Compatibility wrapper around build_feature_matrix().

    Accepts and silently ignores legacy keyword parameters:
        normalization, bg_filter_model, feature_set, min_total_reads, min_fold_change

    Returns a DataFrame with 'Name' as a regular column (not the index),
    matching the output format expected by train_model.py and predict.py.
    """
    ignored_keys = {"normalization", "feature_set", "min_total_reads",
                    "min_fold_change", "bg_filter_model"}
    passed_ignored = {k: v for k, v in kwargs.items() if k in ignored_keys}
    if passed_ignored:
        print(f"  [feature_extraction] Note: the following parameters are not "
              f"used by the updated pipeline and have been ignored: "
              f"{', '.join(passed_ignored)}")

    feat_df = build_feature_matrix(expt_dfs, ctrl_dfs, gene_names=gene_names)

    # Reset index so gene name becomes a 'Name' column (old API format)
    feat_df = feat_df.reset_index().rename(columns={"index": "Name"})
    return feat_df


def learn_background_filter(bg_dfs: list[pd.DataFrame]) -> None:
    """
    Stub for backward compatibility. Background filtering is not implemented
    in the current pipeline version. Returns None.
    """
    print("  [feature_extraction] Note: background filter not supported "
          "in this pipeline version — skipping.")
    return None


def save_background_filter(model: object, path: "Path") -> None:
    """No-op stub for backward compatibility."""
    pass


def load_background_filter(path: "Path") -> None:
    """No-op stub for backward compatibility. Returns None."""
    return None


def apply_background_filter(
    df: pd.DataFrame,
    bg_filter_model: object,
) -> pd.DataFrame:
    """No-op stub for backward compatibility. Returns the DataFrame unchanged."""
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ZERO: dict = {
    "edit_rate":            0.0,
    "edit_count_sum":       0.0,
    "total_count_sum":      0.0,
    "cumulative_edit_rate": 0.0,
    "n_sites":              0,
    "max_site_rate":        0.0,
    "log_coverage":         0.0,
}


def _precompute_gene_stats(df: pd.DataFrame) -> dict[str, dict]:
    """
    Group a single replicate DataFrame by gene and compute per-gene statistics.

    Expects columns: gene, edit_count, total_count, edit_rate.
    Returns dict mapping gene_name -> stats_dict.
    """
    result: dict[str, dict] = {}
    for gene, grp in df.groupby("Name", sort=False):
        edit_sum  = float(grp["edit_count"].sum())
        total_sum = float(grp["total_count"].sum())
        result[gene] = {
            "edit_rate":            edit_sum / total_sum if total_sum > 0 else 0.0,
            "edit_count_sum":       edit_sum,
            "total_count_sum":      total_sum,
            "cumulative_edit_rate": float(grp["edit_rate"].sum()),
            "n_sites":              len(grp),
            "max_site_rate":        float(grp["edit_rate"].max()),
            "log_coverage":         float(np.log1p(total_sum)),
        }
    return result
