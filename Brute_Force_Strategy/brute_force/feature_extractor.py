"""Feature extractor: parses raw replicate files and computes all features.

The extractor always computes the full set of 53 possible features; the
feature_registry.ALL_FEATURES controls which subset is selected into the
feature matrix.  Extra computed features are silently dropped.

Entry point for callers is ``extract_dataset_features``.  Everything else
in this module is an implementation detail and should not be imported
directly.

Performance notes
-----------------
* Each file is loaded once and immediately aggregated to gene level; the raw
  300k-row DataFrame is discarded before the next file is read.
* Per-replicate gene arrays are stacked into (N_reps, N_genes) matrices so
  that mean/std/max aggregations are single NumPy calls.
* Fisher exact tests are computed in a Python loop (~10k iterations), which
  takes < 1 second and is dominated by network I/O on large files anyway.
* The full 53-column feature matrix is cached to Parquet on first extraction
  and reloaded instantly on subsequent runs (same file paths -> same hash).
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom, fisher_exact

from .exceptions import DataLoadError, FeatureExtractionError
from ._types import DatasetConfig, DatasetFeatures, _ReplicateStats
from ._utils import (
    safe_log2,
    safe_log2_ratio,
    safe_div,
    vec_safe_log2,
    vec_safe_div,
    vec_safe_log10_neg,
)
from .feature_registry import ALL_FEATURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name normalisation
# ---------------------------------------------------------------------------

# Accepted aliases for the gene-name column (first match wins).
_GENE_COL_ALIASES: Tuple[str, ...] = ("Name", "name", "Gene", "gene", "GENE")

# Accepted aliases for edited-read count column.
_EDIT_COL_ALIASES: Tuple[str, ...] = (
    "Editbase_count", "editbase_count", "G_count", "g_count",
)

# Accepted aliases for total-read count column.
_TOTAL_COL_ALIASES: Tuple[str, ...] = (
    "Total_count.1", "total_count.1", "Total_count", "total_count",
)


def _resolve_column(df: pd.DataFrame, aliases: Tuple[str, ...], role: str) -> str:
    """Return the first alias that exists in df.columns, or raise DataLoadError."""
    for alias in aliases:
        if alias in df.columns:
            return alias
    raise DataLoadError(
        f"Could not find '{role}' column. Tried: {list(aliases)}. "
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Single-file loading and aggregation
# ---------------------------------------------------------------------------

def _parse_replicate(path: Path, min_reads: int = 0) -> pd.DataFrame:
    """Load a replicate TSV and return a tidy DataFrame with canonical columns.

    Returns a DataFrame with columns: gene, edit_count, total_count, edit_rate.
    One row per editing site.

    Parameters
    ----------
    path:
        Path to the replicate TSV file.
    min_reads:
        Drop any editing site whose total_count is below this value.
        Applied per site before any gene-level aggregation.
        Default 0 means no filtering.

    Raises DataLoadError on any I/O or parse failure.
    """
    if not path.exists():
        raise DataLoadError(f"Replicate file not found: {path}")
    if path.stat().st_size == 0:
        raise DataLoadError(f"Replicate file is empty: {path}")

    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception as exc:
        raise DataLoadError(f"Failed to parse '{path}': {exc}") from exc

    if df.empty:
        raise DataLoadError(f"Replicate file has no data rows: {path}")

    gene_col  = _resolve_column(df, _GENE_COL_ALIASES,  "gene name")
    edit_col  = _resolve_column(df, _EDIT_COL_ALIASES,  "edited read count")
    total_col = _resolve_column(df, _TOTAL_COL_ALIASES, "total read count")

    # Coerce to numeric; non-numeric rows become NaN and are dropped.
    edit_s  = pd.to_numeric(df[edit_col],  errors="coerce")
    total_s = pd.to_numeric(df[total_col], errors="coerce")

    valid_mask = (
        edit_s.notna()
        & total_s.notna()
        & (total_s > 0)
        & edit_s.ge(0)
        & (edit_s <= total_s)
    )

    # Per-site min_reads filter: drop sites with insufficient coverage.
    # This is intentionally per-site (not per-gene) — a gene with low
    # coverage in one replicate may still be a genuine target; the
    # downstream features (fold change, Fisher, binomial) handle that.
    if min_reads > 0:
        valid_mask = valid_mask & (total_s >= min_reads)

    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        if min_reads > 0:
            logger.debug(
                "%s: dropped %d rows (invalid counts or total_count < %d) (of %d).",
                path.name, n_dropped, min_reads, len(df),
            )
        else:
            logger.debug(
                "%s: dropped %d rows with invalid edit/total counts (of %d).",
                path.name, n_dropped, len(df),
            )

    tidy = pd.DataFrame({
        "gene":        df[gene_col][valid_mask].astype(str).values,
        "edit_count":  edit_s[valid_mask].values.astype(np.float64),
        "total_count": total_s[valid_mask].values.astype(np.float64),
    })
    tidy["edit_rate"] = tidy["edit_count"] / tidy["total_count"]

    logger.debug(
        "%s: %d sites across %d genes.",
        path.name, len(tidy), tidy["gene"].nunique(),
    )
    return tidy


def _aggregate_gene_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-site DataFrame to gene-level statistics.

    Returns a DataFrame indexed by gene name with columns:
        sum_edits, sum_total, n_sites, gene_rate,
        mean_site_rate, median_site_rate, max_site_rate, site_edit_sd,
        cum_edit_pct, norm_edit_frac, site_frac.

    ``norm_edit_frac`` and ``site_frac`` are computed against sample totals
    (sum across all genes in this file).
    """
    sample_total_edits = float(df["edit_count"].sum())
    sample_total_sites = len(df)

    agg = df.groupby("gene", sort=True).agg(
        sum_edits        =("edit_count",  "sum"),
        sum_total        =("total_count", "sum"),
        n_sites          =("edit_count",  "count"),
        mean_site_rate   =("edit_rate",   "mean"),
        median_site_rate =("edit_rate",   "median"),
        max_site_rate    =("edit_rate",   "max"),
        site_edit_sd     =("edit_rate",   "std"),   # NaN for single-site genes
        cum_edit_pct     =("edit_rate",   lambda s: (s * 100.0).sum()),
    )

    agg["site_edit_sd"] = agg["site_edit_sd"].fillna(0.0)
    agg["gene_rate"] = agg["sum_edits"] / agg["sum_total"]

    if sample_total_edits > 0:
        agg["norm_edit_frac"] = agg["sum_edits"] / sample_total_edits
    else:
        agg["norm_edit_frac"] = 0.0

    if sample_total_sites > 0:
        agg["site_frac"] = agg["n_sites"] / sample_total_sites
    else:
        agg["site_frac"] = 0.0

    return agg


# ---------------------------------------------------------------------------
# Gene universe alignment
# ---------------------------------------------------------------------------

def _align_to_universe(
    agg_df: pd.DataFrame,
    universe: np.ndarray,
) -> pd.DataFrame:
    """Reindex agg_df to gene universe, filling absent genes with 0."""
    return agg_df.reindex(universe, fill_value=0.0)


def _build_universe(agg_dfs: List[pd.DataFrame]) -> np.ndarray:
    """Return a sorted array of all gene names appearing in any replicate."""
    all_genes: set[str] = set()
    for df in agg_dfs:
        all_genes.update(df.index.tolist())
    return np.array(sorted(all_genes), dtype=object)


# ---------------------------------------------------------------------------
# Fisher exact p-value helpers
# ---------------------------------------------------------------------------

def _fisher_pvals_per_pair(
    expt_edits_list:  List[np.ndarray],
    expt_totals_list: List[np.ndarray],
    ctrl_edits_list:  List[np.ndarray],
    ctrl_totals_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Fisher p-values for all (expt_i, ctrl_j) pairs per gene.

    Returns (pval_mean, pval_min, pval_std) each of shape (N_genes,).
    Uses two-sided Fisher exact test for conservatism.
    """
    n_genes = len(expt_edits_list[0])

    pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for e_idx in range(len(expt_edits_list)):
        for c_idx in range(len(ctrl_edits_list)):
            pairs.append((
                expt_edits_list[e_idx],
                expt_totals_list[e_idx],
                ctrl_edits_list[c_idx],
                ctrl_totals_list[c_idx],
            ))

    pair_pvals = np.ones((len(pairs), n_genes), dtype=np.float64)

    for pidx, (ee, et, ce, ct) in enumerate(pairs):
        for g in range(n_genes):
            a = int(ee[g])
            b = int(et[g]) - a
            c = int(ce[g])
            d = int(ct[g]) - c
            if b < 0 or d < 0:
                # corrupted value: treat as no evidence
                continue
            if a + b == 0 or c + d == 0:
                continue
            try:
                _, pair_pvals[pidx, g] = fisher_exact(
                    [[a, b], [c, d]], alternative="two-sided"
                )
            except Exception:  # noqa: BLE001
                pass  # leave at 1.0

    pval_mean = pair_pvals.mean(axis=0).astype(np.float32)
    pval_min  = pair_pvals.min(axis=0).astype(np.float32)
    pval_std  = pair_pvals.std(axis=0).astype(np.float32)
    return pval_mean, pval_min, pval_std


def _fisher_odds_ratio_pooled(
    pool_expt_edits: np.ndarray,
    pool_expt_total: np.ndarray,
    pool_ctrl_edits: np.ndarray,
    pool_ctrl_total: np.ndarray,
) -> np.ndarray:
    """Fisher odds ratio on pooled counts, shape (N_genes,).

    Genes with zero coverage on either side get OR = 1.0.
    """
    n_genes = len(pool_expt_edits)
    odds_ratios = np.ones(n_genes, dtype=np.float32)

    for g in range(n_genes):
        a = int(pool_expt_edits[g])
        b = int(pool_expt_total[g]) - a
        c = int(pool_ctrl_edits[g])
        d = int(pool_ctrl_total[g]) - c
        if b < 0 or d < 0:
            continue
        if a + b == 0 or c + d == 0:
            continue
        try:
            result = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            or_val = float(result[0])
            # scipy returns inf when control has zero edited reads (perfect enrichment).
            # Cap to a large finite value to keep the feature meaningful for the RF.
            odds_ratios[g] = min(or_val, 1e6) if np.isfinite(or_val) else 1e6
        except Exception:  # noqa: BLE001
            pass
    return odds_ratios


# ---------------------------------------------------------------------------
# Targets file loading
# ---------------------------------------------------------------------------

def _load_targets(targets_file: Path) -> frozenset:
    """Load known target gene names.  Accepts plain-text (one per line) or
    CSV/TSV with a Name / gene column.

    Raises DataLoadError if the file cannot be read.
    """
    if not targets_file.exists():
        raise DataLoadError(f"Targets file not found: {targets_file}")

    suffix = targets_file.suffix.lower()
    try:
        if suffix in (".csv", ".tsv", ".txt"):
            # Try TSV/CSV first.
            try:
                sep = "\t" if suffix == ".tsv" else ","
                df = pd.read_csv(targets_file, sep=sep, header=0)
                for col in ("Name", "name", "Gene", "gene", "GENE"):
                    if col in df.columns:
                        return frozenset(df[col].dropna().astype(str).str.strip())
                # No recognised header -> treat as one-gene-per-line.
            except Exception:
                pass
        # Fallback: plain text, one gene per line.
        with open(targets_file, encoding="utf-8") as fh:
            genes = {line.strip() for line in fh if line.strip()}
        return frozenset(genes)
    except Exception as exc:
        raise DataLoadError(f"Failed to load targets file '{targets_file}': {exc}") from exc


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _dataset_cache_key(cfg: DatasetConfig, min_reads: int = 0) -> str:
    """Stable sha1 key from sorted absolute file paths and filter parameters.

    min_reads is included so that runs with different per-site read filters
    produce distinct cache files and never silently reuse each other's results.
    """
    all_paths = sorted(str(p.resolve()) for p in (*cfg.expt_files, *cfg.ctrl_files))
    key_str = "|".join(all_paths) + f"|min_reads={min_reads}"
    return hashlib.sha1(key_str.encode()).hexdigest()


def _save_cache(matrix: np.ndarray, genes: np.ndarray, cache_path: Path) -> None:
    """Persist feature matrix to Parquet."""
    df = pd.DataFrame(matrix, columns=list(ALL_FEATURES))
    df.insert(0, "__gene__", genes)
    df.to_parquet(cache_path, index=False, engine="pyarrow")
    logger.debug("Feature cache written: %s", cache_path)


def _load_cache(
    cache_path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load feature matrix from Parquet.  Returns (matrix, genes) or None."""
    if not cache_path.exists():
        return None
    try:
        df = pd.read_parquet(cache_path, engine="pyarrow")
        genes = df["__gene__"].values.astype(object)
        matrix = df.drop(columns=["__gene__"]).values.astype(np.float32)
        logger.debug("Feature cache loaded: %s (%d genes)", cache_path, len(genes))
        return matrix, genes
    except Exception as exc:
        logger.warning("Could not load cache '%s' (will recompute): %s", cache_path, exc)
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_dataset_features(
    cfg: DatasetConfig,
    cache_dir: Optional[Path] = None,
    min_total_reads: int = 0,
    min_reads: int = 0,
) -> DatasetFeatures:
    """Extract all 53 features for every gene in one dataset.

    Parameters
    ----------
    cfg:
        Dataset configuration (file paths, name).
    cache_dir:
        If given, feature matrices are saved/loaded from this directory as
        ``{cache_key}.parquet``.  Speeds up subsequent runs significantly.
    min_total_reads:
        Genes whose pooled expt+ctrl total reads fall below this threshold are
        excluded from the returned feature matrix.
    min_reads:
        Per-site filter applied when loading each raw replicate file.
        Any editing site with total_count < min_reads is dropped before
        gene-level aggregation.  A gene absent from one replicate due to
        this filter is not dropped — it simply contributes 0 reads for that
        replicate, which is handled correctly by fold change, Fisher, and
        binomial features.  Default 0 means no filtering.

    Returns
    -------
    DatasetFeatures with a (N_genes, 53) float32 feature matrix.

    Raises
    ------
    DataLoadError
        If any input file cannot be read or parsed.
    FeatureExtractionError
        If feature computation fails unexpectedly.
    """
    t0 = time.perf_counter()
    logger.info("Extracting features for dataset '%s' ...", cfg.name)

    # -- Cache check ---------------------------------------------------------
    cache_key = _dataset_cache_key(cfg, min_reads=min_reads)
    targets   = _load_targets(cfg.targets_file)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.parquet"
        cached = _load_cache(cache_path)
        if cached is not None:
            matrix, genes = cached
            is_target = np.array([g in targets for g in genes], dtype=bool)
            logger.info(
                "Dataset '%s': loaded from cache (%d genes, %d targets).",
                cfg.name, len(genes), is_target.sum(),
            )
            return DatasetFeatures(
                dataset_name=cfg.name,
                genes=genes,
                feature_matrix=matrix,
                feature_names=list(ALL_FEATURES),
                is_target=is_target,
            )

    # -- Load and aggregate each replicate -----------------------------------
    def _load_and_agg(paths: Tuple[Path, ...], label: str) -> List[pd.DataFrame]:
        aggs = []
        for i, p in enumerate(paths, start=1):
            logger.info("  Loading %s replicate %d/%d: %s", label, i, len(paths), p.name)
            raw = _parse_replicate(p, min_reads=min_reads)
            aggs.append(_aggregate_gene_stats(raw))
            del raw  # release ~40 MB early
        return aggs

    expt_aggs = _load_and_agg(cfg.expt_files, "expt")
    ctrl_aggs = _load_and_agg(cfg.ctrl_files, "ctrl")

    # -- Build gene universe --------------------------------------------------
    all_aggs = expt_aggs + ctrl_aggs
    universe = _build_universe(all_aggs)
    n_genes  = len(universe)
    logger.info("  Gene universe: %d genes.", n_genes)

    # -- Align all replicates to universe ------------------------------------
    expt_aligned = [_align_to_universe(df, universe) for df in expt_aggs]
    ctrl_aligned = [_align_to_universe(df, universe) for df in ctrl_aggs]

    n_expt = len(expt_aligned)
    n_ctrl = len(ctrl_aligned)

    # -- Stack into (N_reps, N_genes) matrices for vectorised ops ------------
    def _stack(dfs: List[pd.DataFrame], col: str) -> np.ndarray:
        return np.stack([df[col].values for df in dfs]).astype(np.float64)

    expt_sum_edits        = _stack(expt_aligned, "sum_edits")
    expt_sum_total        = _stack(expt_aligned, "sum_total")
    expt_n_sites          = _stack(expt_aligned, "n_sites")
    expt_gene_rate        = _stack(expt_aligned, "gene_rate")
    expt_mean_site_rate   = _stack(expt_aligned, "mean_site_rate")
    expt_median_site_rate = _stack(expt_aligned, "median_site_rate")
    expt_max_site_rate    = _stack(expt_aligned, "max_site_rate")
    expt_site_edit_sd     = _stack(expt_aligned, "site_edit_sd")
    expt_cum_edit_pct     = _stack(expt_aligned, "cum_edit_pct")
    expt_norm_edit_frac   = _stack(expt_aligned, "norm_edit_frac")
    expt_site_frac        = _stack(expt_aligned, "site_frac")

    ctrl_sum_edits  = _stack(ctrl_aligned, "sum_edits")
    ctrl_sum_total  = _stack(ctrl_aligned, "sum_total")
    ctrl_n_sites    = _stack(ctrl_aligned, "n_sites")
    ctrl_gene_rate  = _stack(ctrl_aligned, "gene_rate")
    ctrl_cum_edit_pct = _stack(ctrl_aligned, "cum_edit_pct")

    # -- Aggregate across replicates -----------------------------------------

    def _agg(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, std, max) along replicate axis (axis=0)."""
        mean = mat.mean(axis=0).astype(np.float32)
        std  = mat.std(axis=0, ddof=0).astype(np.float32)   # ddof=0 -> 0 for n=1
        mx   = mat.max(axis=0).astype(np.float32)
        return mean, std, mx

    me_sum_edits, se_sum_edits, xe_sum_edits = _agg(expt_sum_edits)
    me_sum_total, se_sum_total, xe_sum_total = _agg(expt_sum_total)
    me_n_sites,   se_n_sites,   xe_n_sites   = _agg(expt_n_sites)
    me_gene_rate, se_gene_rate, xe_gene_rate = _agg(expt_gene_rate)
    me_mean_sr,   se_mean_sr,   xe_mean_sr   = _agg(expt_mean_site_rate)
    me_med_sr,    se_med_sr,    xe_med_sr    = _agg(expt_median_site_rate)
    me_max_sr,    se_max_sr,    xe_max_sr    = _agg(expt_max_site_rate)

    mc_sum_edits, sc_sum_edits, xc_sum_edits = _agg(ctrl_sum_edits)
    mc_sum_total, sc_sum_total, xc_sum_total = _agg(ctrl_sum_total)
    mc_n_sites,   sc_n_sites,   xc_n_sites   = _agg(ctrl_n_sites)
    mc_gene_rate, sc_gene_rate, xc_gene_rate = _agg(ctrl_gene_rate)

    # -- Derived means used in core features ---------------------------------
    mean_cum_expt = expt_cum_edit_pct.mean(axis=0).astype(np.float32)
    mean_cum_ctrl = ctrl_cum_edit_pct.mean(axis=0).astype(np.float32)

    # -- Pooled counts (sum across all reps) for Fisher/binomial -------------
    pool_expt_edits = expt_sum_edits.sum(axis=0)
    pool_expt_total = expt_sum_total.sum(axis=0)
    pool_ctrl_edits = ctrl_sum_edits.sum(axis=0)
    pool_ctrl_total = ctrl_sum_total.sum(axis=0)

    # Global ctrl edit rate: used as the null hypothesis for binomial test.
    global_ctrl_edits = pool_ctrl_edits.sum()
    global_ctrl_total = pool_ctrl_total.sum()
    global_ctrl_rate  = safe_div(float(global_ctrl_edits), float(global_ctrl_total), 1e-9)

    # -- Apply min_total_reads filter ----------------------------------------
    if min_total_reads > 0:
        combined_total = (pool_expt_total + pool_ctrl_total)
        keep_mask = combined_total >= min_total_reads
        n_dropped_cov = int((~keep_mask).sum())
        if n_dropped_cov > 0:
            logger.info(
                "  Dropped %d genes below min_total_reads=%d.",
                n_dropped_cov, min_total_reads,
            )
    else:
        keep_mask = np.ones(n_genes, dtype=bool)

    # -- Statistical tests ----------------------------------------------------
    logger.info("  Computing Fisher exact tests (%d genes x %d pairs) ...",
                n_genes, n_expt * n_ctrl)

    # Per-pair Fisher (variable features)
    pval_mean, pval_min, pval_std = _fisher_pvals_per_pair(
        expt_edits_list  = [expt_sum_edits[i] for i in range(n_expt)],
        expt_totals_list = [expt_sum_total[i] for i in range(n_expt)],
        ctrl_edits_list  = [ctrl_sum_edits[i] for i in range(n_ctrl)],
        ctrl_totals_list = [ctrl_sum_total[i] for i in range(n_ctrl)],
    )

    # Pooled Fisher odds ratio (core feature)
    logger.info("  Computing pooled Fisher odds ratios ...")
    odds_ratio = _fisher_odds_ratio_pooled(
        pool_expt_edits, pool_expt_total,
        pool_ctrl_edits, pool_ctrl_total,
    )

    # Binomial test (vectorised across all genes at once)
    logger.info("  Computing binomial tests ...")
    k = pool_expt_edits.astype(np.float64) - 1.0  # successes (k-1 for sf)
    n = pool_expt_total.astype(np.float64)
    k = np.maximum(k, 0.0)  # clamp so sf is valid when edits=0
    # Keep float64 for clip; vec_safe_log10_neg handles the conversion internally.
    binom_pval_arr = np.clip(binom.sf(k, n, global_ctrl_rate), 1e-300, 1.0).astype(np.float32)

    # -- Assemble all 53 features ---------------------------------------------
    try:
        cols: Dict[str, np.ndarray] = {}

        # -- Core 9 --
        cols["site_edit_sd"]          = expt_site_edit_sd.mean(axis=0).astype(np.float32)
        cols["avg_cum_expt_edit_pct"] = mean_cum_expt
        cols["norm_edits"]            = expt_norm_edit_frac.mean(axis=0).astype(np.float32)
        cols["log_true_signal"]       = vec_safe_log2(
                                            np.maximum(mean_cum_expt - mean_cum_ctrl, 0.0)
                                        )
        cols["site_fraction"]         = expt_site_frac.mean(axis=0).astype(np.float32)
        cols["fisher_odds_ratio"]     = odds_ratio
        cols["binom_neg_log10_p"]     = vec_safe_log10_neg(binom_pval_arr)
        cols["site_enrichment"]       = vec_safe_div(me_n_sites, mc_n_sites)
        cols["fold_change"]           = vec_safe_div(me_gene_rate, mc_gene_rate)

        # -- Category A: expt replicate aggregates (21) --
        cols["mean_expt_sum_edits"]        = me_sum_edits
        cols["std_expt_sum_edits"]         = se_sum_edits
        cols["max_expt_sum_edits"]         = xe_sum_edits
        cols["mean_expt_sum_total"]        = me_sum_total
        cols["std_expt_sum_total"]         = se_sum_total
        cols["max_expt_sum_total"]         = xe_sum_total
        cols["mean_expt_n_sites"]          = me_n_sites
        cols["std_expt_n_sites"]           = se_n_sites
        cols["max_expt_n_sites"]           = xe_n_sites
        cols["mean_expt_gene_rate"]        = me_gene_rate
        cols["std_expt_gene_rate"]         = se_gene_rate
        cols["max_expt_gene_rate"]         = xe_gene_rate
        cols["mean_expt_mean_site_rate"]   = me_mean_sr
        cols["std_expt_mean_site_rate"]    = se_mean_sr
        cols["max_expt_mean_site_rate"]    = xe_mean_sr
        cols["mean_expt_median_site_rate"] = me_med_sr
        cols["std_expt_median_site_rate"]  = se_med_sr
        cols["max_expt_median_site_rate"]  = xe_med_sr
        cols["mean_expt_max_site_rate"]    = me_max_sr
        cols["std_expt_max_site_rate"]     = se_max_sr
        cols["max_expt_max_site_rate"]     = xe_max_sr

        # -- Category B: ctrl replicate aggregates (12) --
        cols["mean_ctrl_sum_edits"]  = mc_sum_edits
        cols["std_ctrl_sum_edits"]   = sc_sum_edits
        cols["max_ctrl_sum_edits"]   = xc_sum_edits
        cols["mean_ctrl_sum_total"]  = mc_sum_total
        cols["std_ctrl_sum_total"]   = sc_sum_total
        cols["max_ctrl_sum_total"]   = xc_sum_total
        cols["mean_ctrl_n_sites"]    = mc_n_sites
        cols["std_ctrl_n_sites"]     = sc_n_sites
        cols["max_ctrl_n_sites"]     = xc_n_sites
        cols["mean_ctrl_gene_rate"]  = mc_gene_rate
        cols["std_ctrl_gene_rate"]   = sc_gene_rate
        cols["max_ctrl_gene_rate"]   = xc_gene_rate

        # -- Category C: Fisher aggregates (4) --
        cols["fisher_pval_mean"] = pval_mean
        cols["fisher_pval_min"]  = pval_min
        cols["fisher_pval_std"]  = pval_std
        cols["log_fisher_avg"]   = vec_safe_log10_neg(pval_mean)

        # -- Category D: enrichment / statistical (2) --
        cols["rate_diff_vs_ctrl"] = (me_gene_rate - mc_gene_rate).astype(np.float32)
        cols["binomial_pval"]     = binom_pval_arr

        # -- Category E: current-branch unique (5) --
        cols["log_norm_edits"]           = vec_safe_log2(
                                               expt_norm_edit_frac.mean(axis=0).astype(np.float32)
                                           )
        cols["site_enrichment_edit_rate"] = vec_safe_div(mean_cum_expt, mean_cum_ctrl)
        cols["signal_baseline"]           = vec_safe_log2(
                                                vec_safe_div(
                                                    np.maximum(mean_cum_expt - mean_cum_ctrl, 0.0),
                                                    np.maximum(mean_cum_ctrl, 0.0),
                                                )
                                            )
        genome_mean_expt = float(mean_cum_expt.mean())
        genome_mean_ctrl = float(mean_cum_ctrl.mean())
        cols["cum_expt_above_mean"] = (mean_cum_expt - genome_mean_expt).astype(np.float32)
        cols["cum_ctrl_above_mean"] = (mean_cum_ctrl - genome_mean_ctrl).astype(np.float32)

    except Exception as exc:
        raise FeatureExtractionError(
            f"Feature computation failed for dataset '{cfg.name}': {exc}"
        ) from exc

    # -- Validate registry features are all present in computed cols ----------
    # Extra computed features beyond ALL_FEATURES are silently dropped;
    # this allows the extractor to compute all 53 features while the registry
    # controls which subset is actually used by the model.
    missing = set(ALL_FEATURES) - set(cols.keys())
    if missing:
        raise FeatureExtractionError(
            f"Feature column mismatch for '{cfg.name}'. "
            f"Missing features required by registry: {sorted(missing)}."
        )
    extra = set(cols.keys()) - set(ALL_FEATURES)
    if extra:
        logger.debug(
            "Dataset '%s': dropping %d computed features not in registry: %s",
            cfg.name, len(extra), sorted(extra),
        )

    # -- Build feature matrix in canonical column order -----------------------
    # Only select features present in ALL_FEATURES (drops extras silently)
    feature_matrix = np.column_stack(
        [cols[f] for f in ALL_FEATURES]
    ).astype(np.float32)

    # -- Apply coverage filter ------------------------------------------------
    universe_filtered = universe[keep_mask]
    feature_matrix    = feature_matrix[keep_mask]

    # -- Replace NaN / Inf with 0 (safety net; should not occur) -------------
    bad = ~np.isfinite(feature_matrix)
    if bad.any():
        n_bad = int(bad.sum())
        logger.warning(
            "Dataset '%s': replaced %d NaN/Inf values in feature matrix with 0.",
            cfg.name, n_bad,
        )
        feature_matrix[bad] = 0.0

    # -- Target labels --------------------------------------------------------
    is_target = np.array([g in targets for g in universe_filtered], dtype=bool)

    logger.info(
        "Dataset '%s': %d genes (%d targets) | %.2f s.",
        cfg.name, len(universe_filtered), int(is_target.sum()),
        time.perf_counter() - t0,
    )

    # -- Persist cache --------------------------------------------------------
    if cache_dir is not None:
        try:
            _save_cache(feature_matrix, universe_filtered, cache_path)
        except Exception as exc:
            logger.warning("Could not write feature cache: %s", exc)

    return DatasetFeatures(
        dataset_name=cfg.name,
        genes=universe_filtered,
        feature_matrix=feature_matrix,
        feature_names=list(ALL_FEATURES),
        is_target=is_target,
    )
