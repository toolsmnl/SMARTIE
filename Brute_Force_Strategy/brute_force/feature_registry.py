"""Canonical registry of all 20 features.

Single source of truth for feature names, ordering, and descriptions.
Import CORE_FEATURES, VARIABLE_FEATURES, or ALL_FEATURES from here;
never hard-code feature names elsewhere in the package.

Change log:
    - mean_expt_max_site_rate and std_expt_max_site_rate demoted from core
      back to variable (brute force will now confirm whether they are
      genuinely essential rather than assuming it)
    - 5 weak variable features dropped: mean_ctrl_n_sites, signal_baseline,
      std_ctrl_sum_total, std_ctrl_n_sites, max_expt_sum_total
    - Core: 11 -> 9, Variable: 14 -> 16, Total: 25 -> 25
    - kmax range for brute-force search: 1-10 (58,650 combos)
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

# ---------------------------------------------------------------------------
# Core features (9) -- present in every combination
# ---------------------------------------------------------------------------

CORE_FEATURES: Tuple[str, ...] = (
    "site_edit_sd",             # std dev of per-site edit rates within gene (across sites)
    "avg_cum_expt_edit_pct",    # mean cumulative edit % across expt replicates (absolute)
    "norm_edits",               # mean(gene_edits / sample_total_edits) across expt reps
    "log_true_signal",          # log2(mean_cum_expt - mean_cum_ctrl)
    "site_fraction",            # mean(gene_sites / sample_total_sites) across expt reps
    "fisher_odds_ratio",        # odds ratio from Fisher exact on pooled expt vs ctrl counts
    "binom_neg_log10_p",        # -log10(binomial p), gene expt rate vs global ctrl rate
    "site_enrichment",          # mean_expt_n_sites / mean_ctrl_n_sites
    "fold_change",              # mean_expt_gene_rate / mean_ctrl_gene_rate
)

# ---------------------------------------------------------------------------
# Variable features (16) -- combined combinatorially with the 9 core
# ---------------------------------------------------------------------------

VARIABLE_FEATURES: Tuple[str, ...] = (
    "std_expt_n_sites",          # std of editing site count across expt reps
    "site_enrichment_edit_rate", # mean_cum_edit_pct_expt / mean_cum_edit_pct_ctrl
    "max_expt_n_sites",          # max editing site count across expt reps
    "max_expt_max_site_rate",    # max hottest-site edit rate across expt reps
    "max_ctrl_sum_edits",        # max total edited reads across ctrl reps
    "mean_expt_n_sites",         # mean editing site count across expt reps
    "mean_ctrl_sum_edits",       # mean total edited reads across ctrl reps
    "std_expt_sum_edits",        # std of total edited reads across expt reps
    "mean_ctrl_gene_rate",       # mean gene-level edit rate across ctrl reps
    "mean_expt_max_site_rate",   # mean of hottest-site edit rate across expt reps [demoted]
    "std_expt_max_site_rate",    # std of hottest-site edit rate across expt reps  [demoted]
    "std_ctrl_n_sites",          # std of editing site count across ctrl reps
    "mean_ctrl_n_sites",         # mean editing site count across ctrl reps
    "signal_baseline",           # log2((mean_cum_expt - mean_cum_ctrl) / mean_cum_ctrl)
    "std_ctrl_sum_total",        # std of total read coverage across ctrl reps
    "max_expt_sum_total",        # max total read coverage across expt reps
)

assert len(CORE_FEATURES) == 9, (
    f"Expected 9 core features, got {len(CORE_FEATURES)}"
)

assert len(VARIABLE_FEATURES) == 16, (
    f"Expected 16 variable features, got {len(VARIABLE_FEATURES)}"
)

# ---------------------------------------------------------------------------
# Complete ordered feature list (25)
# ---------------------------------------------------------------------------

ALL_FEATURES: Tuple[str, ...] = CORE_FEATURES + VARIABLE_FEATURES

assert len(ALL_FEATURES) == 25, (
    f"Expected 25 total features, got {len(ALL_FEATURES)}"
)

# Fast membership tests
CORE_FEATURE_SET: FrozenSet[str] = frozenset(CORE_FEATURES)
VARIABLE_FEATURE_SET: FrozenSet[str] = frozenset(VARIABLE_FEATURES)
ALL_FEATURE_SET: FrozenSet[str] = frozenset(ALL_FEATURES)

# Column index lookup: feature name -> index in ALL_FEATURES
FEATURE_INDEX: Dict[str, int] = {f: i for i, f in enumerate(ALL_FEATURES)}

# ---------------------------------------------------------------------------
# Human-readable descriptions (used in docs and query output)
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # Core (9)
    "site_edit_sd":               "Std dev of per-site edit rates within gene, averaged across expt reps",
    "avg_cum_expt_edit_pct":      "Mean cumulative edit % (sum of per-site rates x 100) across expt reps",
    "norm_edits":                 "Mean(gene_edits / sample_total_edits) across expt reps",
    "log_true_signal":            "log2(mean_cum_expt - mean_cum_ctrl), floor -20",
    "site_fraction":              "Mean(gene_n_sites / sample_total_sites) across expt reps",
    "fisher_odds_ratio":          "Odds ratio from Fisher exact on pooled expt vs ctrl counts",
    "binom_neg_log10_p":          "-log10(binomial p), gene expt rate vs global ctrl rate",
    "site_enrichment":            "mean_expt_n_sites / mean_ctrl_n_sites",
    "fold_change":                "mean_expt_gene_rate / mean_ctrl_gene_rate",
    # Variable (11)
    "std_expt_n_sites":           "Std of editing site count across expt reps",
    "site_enrichment_edit_rate":  "mean_cum_edit_pct_expt / mean_cum_edit_pct_ctrl",
    "max_expt_n_sites":           "Max editing site count across expt reps",
    "max_expt_max_site_rate":     "Max hottest-site edit rate across expt reps",
    "max_ctrl_sum_edits":         "Max total edited reads across ctrl reps",
    "mean_expt_n_sites":          "Mean editing site count across expt reps",
    "mean_ctrl_sum_edits":        "Mean total edited reads across ctrl reps",
    "std_expt_sum_edits":         "Std of total edited reads across expt reps",
    "mean_ctrl_gene_rate":        "Mean gene-level edit rate across ctrl reps",
    "mean_expt_max_site_rate":    "Mean of hottest-site edit rate across expt reps [demoted from core]",
    "std_expt_max_site_rate":     "Std of hottest-site edit rate across expt reps [demoted from core]",
    "std_ctrl_n_sites":           "Std of editing site count across ctrl reps",
    "mean_ctrl_n_sites":          "Mean editing site count across ctrl reps",
    "signal_baseline":            "log2((mean_cum_expt - mean_cum_ctrl) / mean_cum_ctrl), floor -20",
    "std_ctrl_sum_total":         "Std of total read coverage across ctrl reps",
    "max_expt_sum_total":         "Max total read coverage across expt reps",
}

assert set(FEATURE_DESCRIPTIONS) == ALL_FEATURE_SET, (
    "FEATURE_DESCRIPTIONS keys do not match ALL_FEATURES"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_combo_features(variable_subset: List[str]) -> List[str]:
    """Return the full ordered feature list for a given variable subset.

    The 9 core features are always prepended in their canonical order.
    Raises ValueError if any name in variable_subset is not a valid variable feature.
    """
    unknown = set(variable_subset) - VARIABLE_FEATURE_SET
    if unknown:
        raise ValueError(
            f"Unknown variable feature(s): {sorted(unknown)}. "
            f"Valid names are in VARIABLE_FEATURES."
        )
    return list(CORE_FEATURES) + list(variable_subset)


def validate_feature_names(names: List[str]) -> None:
    """Raise ValueError if any name is not in ALL_FEATURES."""
    unknown = set(names) - ALL_FEATURE_SET
    if unknown:
        raise ValueError(
            f"Unrecognised feature name(s): {sorted(unknown)}"
        )
