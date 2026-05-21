"""brute_force_multitrain.py -- brute-force combo search with separated train/test datasets.

This is a standalone variant of ``brute_force_main.py``.  It differs in a single
respect: the training and testing dataset pools are given as two separate TSV
metadata files.  For every feature combination the search trains one Random
Forest per training dataset and evaluates that model against every dataset
in the test pool.  Training datasets never appear in the test pool -- any
test-metadata row whose ``name`` matches a training name is dropped at load
time with a warning.

Target leakage
--------------
Target labels are drawn from ``train_features.is_target`` during model fitting
and used only for that purpose.  The test-metadata ``targets_file`` column is
loaded solely so that per-test-dataset precision can be computed *after* the
model has produced its ranking; those labels never enter the estimator.

Dataset-metadata format (one TSV per pool, identical schema)
------------------------------------------------------------
Tab-separated with four required columns:

    name    expt_files                       ctrl_files                       targets_file
    Ataxin  Ataxin_E1.txt,Ataxin_E2.txt      Ataxin_C1.txt,Ataxin_C2.txt      Ataxin_targets.txt
    Hrp48   Hrp48_E1.txt,Hrp48_E2.txt        Hrp48_C1.txt,Hrp48_C2.txt        Hrp48_targets.txt

File paths are resolved relative to the metadata file's own directory.
Multiple replicates go in a single comma-separated cell.

Usage
-----
    python brute_force_multitrain.py \\
        --train-metadata training/train_datasets.tsv \\
        --test-metadata  training/test_datasets.tsv \\
        --trees 100 200 --output-dir results/ --workers 4 --resume
"""


from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message=".*delayed.*sklearn.*")
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from brute_force.exceptions import BruteForceError, ConfigError, DataLoadError
from brute_force.feature_extractor import extract_dataset_features
from brute_force.feature_registry import (
    ALL_FEATURES,
    CORE_FEATURES,
    VARIABLE_FEATURES,
    get_combo_features,
)
from brute_force._types import DatasetConfig, DatasetFeatures
from brute_force.combo_generator import (
    combo_hash,
    describe_combo,
    iter_variable_combos,
    total_combos_in_range,
)
from brute_force.trainer import train_and_rank
from brute_force.evaluator import (
    build_heatmap_df,
    aggregate_combo_summary,
    precision_curve,
    PRECISION_THRESHOLDS,
)
from brute_force.result_store import (
    IndexWriter,
    ProgressTracker,
    build_index_record,
    load_completed_hashes,
    save_combo_results,
)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger("brute_force_multitrain")

# Width of the separator lines used in printed output.
_SEP_WIDTH = 78


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "-") -> str:
    return char * _SEP_WIDTH


def _print_startup_banner(
    train_datasets: List[DatasetFeatures],
    test_datasets:  List[DatasetFeatures],
    trees: List[int],
    k_min: int,
    k_max: int,
    workers: int,
    rf_n_jobs: int,
    output_dir: Path,
    model: str,
    resume: bool,
) -> None:
    total = total_combos_in_range(k_min, k_max)
    tree_str = "  ".join(str(t) for t in trees)

    print(_sep("="))
    print("  BRUTE-FORCE MULTITRAIN  --  separated train/test dataset pools")
    print(_sep("="))
    print(f"  Training datasets ({len(train_datasets)}):")
    for ds in train_datasets:
        print(
            f"    - {ds.dataset_name:<20}  {ds.n_genes:>6,} genes   "
            f"{ds.n_targets:>4} targets"
        )
    print()
    print(f"  Test datasets ({len(test_datasets)}):")
    for ds in test_datasets:
        print(
            f"    - {ds.dataset_name:<20}  {ds.n_genes:>6,} genes   "
            f"{ds.n_targets:>4} targets"
        )
    print()
    print(f"  Trees          : {tree_str}")
    print(f"  Feature range  : k={k_min}..{k_max}  (n_features={k_min+9}..{k_max+9})")
    print(f"  Total combos   : {total:,}  (per tree size)")
    print(f"  Workers        : {workers} combo workers x {rf_n_jobs} RF thread(s)")
    print(f"  Model          : {model.upper()}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Resume mode    : {'on' if resume else 'off'}")
    print(_sep("-"))
    print()


def _print_k_header(n_trees: int, k: int, n_features: int,
                    n_pending: int, n_already_done: int) -> None:
    already = f"  ({n_already_done} already done, skipped)" if n_already_done else ""
    print()
    print(_sep("-"))
    print(
        f"  n_trees={n_trees}  |  n_features={n_features}  (9 core + {k} variable)"
        f"  |  {n_pending:,} combos to process{already}"
    )
    print(_sep("-"))


def _format_combo_line(rec: dict, rank: Optional[int] = None,
                       mark_best: bool = False) -> str:
    var  = rec.get("variable_features", "").replace("|", "  +  ")
    p10  = rec.get("mean_precision_at_10pct",  0.0)
    p20  = rec.get("mean_precision_at_20pct",  0.0)
    p50  = rec.get("mean_precision_at_50pct",  0.0)
    p100 = rec.get("mean_precision_at_100pct", 0.0)
    auc  = rec.get("mean_auc", 0.0)
    star = "  * NEW BEST" if mark_best else ""
    prefix = f"  #{rank:<3}" if rank is not None else "  "
    return (
        f"{prefix}  @10={p10:.3f}  @20={p20:.3f}  @50={p50:.3f}  @100={p100:.3f}"
        f"  AUC={auc:.3f}    {var}{star}"
    )


def _print_top_results(records: List[dict], n_trees: int, n_features: int,
                       top_n: int = 10) -> None:
    if not records:
        return
    sorted_recs = sorted(
        records, key=lambda r: r.get("mean_precision_at_20pct", 0), reverse=True,
    )
    show = sorted_recs[:top_n]

    print()
    print(_sep("="))
    print(
        f"  TOP {min(top_n, len(show))} RESULTS  |  n_trees={n_trees}  "
        f"|  n_features={n_features}  |  ranked by mean precision@20%"
    )
    print(_sep("-"))
    print(
        f"  {'Rank':<5}  {'@10%':>6}  {'@20%':>6}  {'@50%':>6}  {'@100%':>6}  "
        f"{'AUC':>6}    Variable features added"
    )
    print(_sep("-"))
    for i, rec in enumerate(show, start=1):
        var  = rec.get("variable_features", "").replace("|", " + ")
        p10  = rec.get("mean_precision_at_10pct",  0.0)
        p20  = rec.get("mean_precision_at_20pct",  0.0)
        p50  = rec.get("mean_precision_at_50pct",  0.0)
        p100 = rec.get("mean_precision_at_100pct", 0.0)
        auc  = rec.get("mean_auc", 0.0)
        print(
            f"  {i:<5}  {p10:>6.3f}  {p20:>6.3f}  {p50:>6.3f}  {p100:>6.3f}  "
            f"{auc:>6.3f}    {var}"
        )
    print(_sep("="))
    print()


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def load_metadata_file(path: Path, role: str) -> List[DatasetConfig]:
    """Parse one TSV dataset manifest (either the train or test pool).

    ``role`` is used purely for error messages ("train" or "test").  The file
    format is identical to the one used by ``brute_force_main.py`` -- four
    columns: ``name``, ``expt_files``, ``ctrl_files``, ``targets_file``.

    Raises ConfigError if the file is missing, malformed, or references files
    that do not exist.
    """
    if not path.exists():
        raise ConfigError(f"{role.capitalize()}-metadata file not found: {path}")

    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
    except Exception as exc:
        raise ConfigError(
            f"Could not parse {role}-metadata file '{path}': {exc}"
        ) from exc

    required = {"name", "expt_files", "ctrl_files", "targets_file"}
    missing  = required - set(df.columns)
    if missing:
        raise ConfigError(
            f"{role.capitalize()}-metadata file missing columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    base_dir = path.parent
    configs: List[DatasetConfig] = []

    for row_idx, row in df.iterrows():
        name = str(row["name"]).strip()
        if not name:
            raise ConfigError(f"{role.capitalize()}-metadata row {row_idx}: 'name' is empty.")

        def _resolve_files(cell: str, col: str) -> Tuple[Path, ...]:
            parts = [p.strip() for p in str(cell).split(",") if p.strip()]
            if not parts:
                raise ConfigError(
                    f"{role.capitalize()} dataset '{name}': no {col} files specified."
                )
            resolved = []
            for p in parts:
                full = (base_dir / p).resolve()
                if not full.exists():
                    raise ConfigError(
                        f"{role.capitalize()} dataset '{name}': "
                        f"{col} file not found: {full}"
                    )
                resolved.append(full)
            return tuple(resolved)

        expt_files   = _resolve_files(row["expt_files"],   "expt_files")
        ctrl_files   = _resolve_files(row["ctrl_files"],   "ctrl_files")
        targets_path = (base_dir / str(row["targets_file"]).strip()).resolve()
        if not targets_path.exists():
            raise ConfigError(
                f"{role.capitalize()} dataset '{name}': "
                f"targets file not found: {targets_path}"
            )

        configs.append(DatasetConfig(
            name=name,
            expt_files=expt_files,
            ctrl_files=ctrl_files,
            targets_file=targets_path,
        ))

    if not configs:
        raise ConfigError(f"{role.capitalize()}-metadata file contains no data rows.")

    # Reject duplicate names within a single pool -- they would overwrite each
    # other silently in per_train output directories.
    seen = set()
    for cfg in configs:
        if cfg.name in seen:
            raise ConfigError(
                f"Duplicate dataset name '{cfg.name}' in {role}-metadata file."
            )
        seen.add(cfg.name)

    return configs


def _filter_test_configs(
    train_configs: List[DatasetConfig],
    test_configs:  List[DatasetConfig],
) -> List[DatasetConfig]:
    """Drop any test-pool entry whose name also appears in the train pool.

    Emits a printed warning for every dropped entry so the user can confirm
    the exclusion visually.
    """
    train_names = {cfg.name for cfg in train_configs}
    kept:     List[DatasetConfig] = []
    excluded: List[str] = []
    for cfg in test_configs:
        if cfg.name in train_names:
            excluded.append(cfg.name)
        else:
            kept.append(cfg)

    if excluded:
        print()
        print(_sep("-"))
        print(
            f"  NOTE: {len(excluded)} test-pool dataset(s) also appear in the "
            f"train pool and will be excluded from testing:"
        )
        for name in excluded:
            print(f"    - {name}")
        print(_sep("-"))
        for name in excluded:
            logger.warning(
                "Dataset '%s' present in both train and test pools; excluded from test pool.",
                name,
            )

    if not kept:
        raise ConfigError(
            "After excluding datasets shared with the train pool, the test pool is empty. "
            "Add at least one test-only dataset."
        )

    return kept


# ---------------------------------------------------------------------------
# Degenerate-column check
# ---------------------------------------------------------------------------

def _has_degenerate_columns(
    datasets: List[DatasetFeatures],
    feature_subset: List[str],
) -> bool:
    """True if any feature column has zero variance across *every* given dataset.

    A column that is constant in every dataset the RF will ever see carries no
    information and the combo is skipped.  We pass the union of train and test
    datasets here so that a column constant only in the train pool but variable
    in the test pool is not incorrectly flagged.
    """
    for feat in feature_subset:
        all_zero_var = True
        for ds in datasets:
            col = ds.select_features([feat])[:, 0]
            if col.std() > 1e-9:
                all_zero_var = False
                break
        if all_zero_var:
            return True
    return False


# ---------------------------------------------------------------------------
# Single-combo worker
# ---------------------------------------------------------------------------

def _run_combo(
    variable_features: Tuple,
    train_datasets: List[DatasetFeatures],
    test_datasets:  List[DatasetFeatures],
    n_trees: int,
    output_dir: Path,
    rf_n_jobs: int,
    use_xgb: bool,
    use_gpu: bool,
    skip_degenerate: bool,
) -> Optional[dict]:
    """Process one feature combination for the train/test multitrain setup.

    For each training dataset we train exactly one model and evaluate it
    against every test dataset.  Training datasets are NEVER used as the test
    side; that exclusion has already been performed in ``_filter_test_configs``.

    Returns an index record dict on success, or None if the combo was skipped
    or any sub-step raised a BruteForceError.
    """
    full_features  = get_combo_features(list(variable_features))
    chash          = combo_hash(full_features)

    if skip_degenerate and _has_degenerate_columns(
        train_datasets + test_datasets, full_features,
    ):
        logger.debug("Skipping degenerate combo %s.", chash)
        return None

    meta = describe_combo(variable_features)
    meta["n_trees"] = n_trees

    all_results = []                # flat list of every train x test result
    per_train: Dict[str, dict] = {} # same structure save_combo_results expects

    for train_ds in train_datasets:
        train_results_for_ds = []
        for test_ds in test_datasets:
            try:
                result = train_and_rank(
                    train_features = train_ds,
                    test_features  = test_ds,
                    feature_subset = full_features,
                    n_trees        = n_trees,
                    rf_n_jobs      = rf_n_jobs,
                    use_xgb        = use_xgb,
                    use_gpu        = use_gpu,
                )
                train_results_for_ds.append(result)
                all_results.append(result)
            except BruteForceError as exc:
                logger.warning(
                    "combo %s | train=%s | test=%s: %s",
                    chash, train_ds.dataset_name, test_ds.dataset_name, exc,
                )
                return None

        # Heatmap across all test datasets for this training model.
        heatmap_df = build_heatmap_df(train_results_for_ds)

        # Feature importances come from the trained model: identical across
        # every test dataset this model was scored against, so we grab them
        # from the first result.
        importances = train_results_for_ds[0].feature_importances
        imp_df = pd.DataFrame(
            sorted(importances.items(), key=lambda kv: kv[1], reverse=True),
            columns=["feature", "importance"],
        )

        # Per-test-dataset precision at each threshold.
        metrics: dict = {"train_dataset": train_ds.dataset_name}
        for pct in PRECISION_THRESHOLDS:
            key = f"precision_at_{pct}pct"
            metrics[key] = {
                r.test_dataset: round(
                    precision_curve(r.ranked_genes)[key], 6
                )
                for r in train_results_for_ds
            }

        per_train[train_ds.dataset_name] = {
            "heatmap_df":            heatmap_df,
            "feature_importance_df": imp_df,
            "metrics":               metrics,
        }

    # Combo-level summary averages across every (train, test) pair we ran.
    summary = aggregate_combo_summary(all_results)

    try:
        save_combo_results(
            output_dir        = output_dir,
            n_trees           = n_trees,
            combo_hash        = chash,
            combo_metadata    = meta,
            combo_summary     = summary,
            per_train_results = per_train,
        )
    except Exception as exc:
        logger.error("Failed to write results for combo %s: %s", chash, exc)
        return None

    return build_index_record(n_trees, meta, summary)


# ---------------------------------------------------------------------------
# Per-tree-size run
# ---------------------------------------------------------------------------

def run_tree_size(
    n_trees: int,
    train_datasets: List[DatasetFeatures],
    test_datasets:  List[DatasetFeatures],
    output_dir: Path,
    k_min: int,
    k_max: int,
    workers: int,
    rf_n_jobs: int,
    use_xgb: bool,
    use_gpu: bool,
    resume: bool,
    index_writer: IndexWriter,
    batch_size: int,
    skip_degenerate: bool,
) -> None:
    """Run all combinations for one n_trees value."""
    total    = total_combos_in_range(k_min, k_max)
    progress = ProgressTracker(output_dir, total)

    for k in range(k_min, k_max + 1):
        n_features_folder = k + len(CORE_FEATURES)

        completed: set = set()
        if resume:
            completed = load_completed_hashes(output_dir, n_trees, n_features_folder)

        pending = []
        for combo in iter_variable_combos(k):
            full  = get_combo_features(list(combo))
            chash = combo_hash(full)
            if chash in completed:
                progress.increment()
                continue
            pending.append(combo)

        n_already = len(completed)
        _print_k_header(n_trees, k, n_features_folder, len(pending), n_already)

        if not pending:
            print(f"  All {n_already:,} combos already completed -- skipping.\n")
            continue

        pbar = tqdm(
            total=len(pending),
            desc=f"  n_feat={n_features_folder}",
            unit="combo",
            dynamic_ncols=True,
            bar_format=(
                "  {l_bar}{bar}| {n_fmt}/{total_fmt}"
                " [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
            ),
            file=sys.stderr,
        )

        best_prec20 = -1.0
        k_records: List[dict] = []
        t_k_start = time.time()

        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]

            records = Parallel(n_jobs=workers, backend="loky", verbose=0)(
                delayed(_run_combo)(
                    variable_features = combo,
                    train_datasets    = train_datasets,
                    test_datasets     = test_datasets,
                    n_trees           = n_trees,
                    output_dir        = output_dir,
                    rf_n_jobs         = rf_n_jobs,
                    use_xgb           = use_xgb,
                    use_gpu           = use_gpu,
                    skip_degenerate   = skip_degenerate,
                )
                for combo in batch
            )

            new_bests: List[dict] = []
            for rec in records:
                progress.increment()
                pbar.update(1)
                if rec is None:
                    continue
                index_writer.append(rec)
                k_records.append(rec)
                p20 = rec.get("mean_precision_at_20pct", 0.0)
                if p20 > best_prec20:
                    best_prec20 = p20
                    new_bests.append(rec)
                    pbar.set_postfix(best_p20=f"{best_prec20:.3f}", refresh=True)

            for rec in new_bests:
                tqdm.write(_format_combo_line(rec, mark_best=True))

        pbar.close()
        elapsed_k = time.time() - t_k_start
        print(
            f"\n  Completed {len(k_records):,} combos in "
            f"{elapsed_k/60:.1f} min  |  best @20% = {max(best_prec20, 0.0):.3f}"
        )

        _print_top_results(k_records, n_trees, n_features_folder, top_n=10)

    index_writer.flush()
    progress.finalize()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brute_force_multitrain.py",
        description=(
            "Exhaustive feature-combination search with separated train/test "
            "dataset pools.  For every feature combo, one Random Forest is "
            "trained per training dataset and evaluated against every test "
            "dataset.  Any test-pool entry whose name matches a training "
            "dataset is automatically excluded."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset specification -- two TSV files, same schema:

    name    expt_files                    ctrl_files                    targets_file
    Ataxin  Ataxin_E1.txt,Ataxin_E2.txt   Ataxin_C1.txt,Ataxin_C2.txt   Ataxin_targets.txt

File paths are resolved relative to each metadata file's own directory.

targets_file
  * train metadata: used to label positives during model fitting.
  * test  metadata: used ONLY for post-hoc precision evaluation; never
                    passed to the estimator.  May be empty (one-column header
                    with no data) if no known targets exist for that dataset,
                    in which case predictions are still produced but precision
                    cannot be computed.
""",
    )

    # -- Dataset specification -----------------------------------------------
    ds_group = p.add_argument_group("dataset pools (both required)")
    ds_group.add_argument(
        "--train-metadata", type=Path, required=True, metavar="TSV",
        help="TSV manifest of training datasets.",
    )
    ds_group.add_argument(
        "--test-metadata", type=Path, required=True, metavar="TSV",
        help="TSV manifest of test datasets (post-hoc evaluation only).",
    )

    # -- Required run parameters ---------------------------------------------
    p.add_argument(
        "--trees", required=True, nargs="+", type=int, metavar="N",
        help="Number of RF trees to test.  Multiple values run sequentially.",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path, metavar="DIR",
        help="Root output directory.  Created if it does not exist.",
    )

    # -- Parallelism ---------------------------------------------------------
    p.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Parallel combo workers (default: 4).",
    )
    p.add_argument(
        "--rf-threads", type=int, default=None, metavar="N",
        help="n_jobs inside each RF fit (default: cpu_count // workers).",
    )

    # -- Combo range ---------------------------------------------------------
    p.add_argument(
        "--k-min", type=int, default=1, metavar="K",
        help="Min variable features per combo (folder = k+9, default: 1).",
    )
    p.add_argument(
        "--k-max", type=int, default=len(VARIABLE_FEATURES), metavar="K",
        help=f"Max variable features per combo (default: {len(VARIABLE_FEATURES)}).",
    )

    # -- Resume / cache ------------------------------------------------------
    p.add_argument(
        "--resume", action="store_true",
        help="Skip combos that already have a combo_summary.json.",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=None, metavar="DIR",
        help="Directory for feature matrix cache (Parquet).",
    )
    p.add_argument(
        "--min-total-reads", type=int, default=0, metavar="N",
        help="Exclude genes with fewer than N combined expt+ctrl reads.",
    )
    p.add_argument(
        "--min-reads", type=int, default=0, metavar="N",
        help=(
            "Per-site read depth filter applied when loading each raw "
            "replicate file.  Editing sites with total_count < N are "
            "dropped before any feature computation.  Does not drop "
            "genes.  Default: 0 (no filter)."
        ),
    )

    # -- Model ---------------------------------------------------------------
    p.add_argument(
        "--model", choices=["rf", "xgb"], default="rf",
        help="Classifier: 'rf' (Random Forest) or 'xgb' (XGBoost).",
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="Use XGBoost GPU mode (requires --model xgb and a CUDA device).",
    )

    # -- Tuning --------------------------------------------------------------
    p.add_argument(
        "--batch-size", type=int, default=64, metavar="N",
        help="Combos per joblib dispatch batch (default: 64).",
    )
    p.add_argument(
        "--index-flush-every", type=int, default=50, metavar="N",
        help="Flush results_index.parquet every N completed combos (default: 50).",
    )
    p.add_argument(
        "--skip-degenerate", action="store_true", default=True,
        help="Skip combos where any feature column is constant across all datasets.",
    )
    p.add_argument(
        "--no-skip-degenerate", dest="skip_degenerate", action="store_false",
        help="Disable degenerate-combo skipping.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )

    # -- Validate args --------------------------------------------------------
    if args.gpu and args.model != "xgb":
        parser.error("--gpu requires --model xgb.")
    if args.k_min < 1:
        parser.error("--k-min must be >= 1.")
    if args.k_max > len(VARIABLE_FEATURES):
        args.k_max = len(VARIABLE_FEATURES)
        logger.warning("--k-max capped at %d (total variable features).", args.k_max)
    if args.k_min > args.k_max:
        parser.error(f"--k-min ({args.k_min}) must be <= --k-max ({args.k_max}).")

    rf_n_jobs = args.rf_threads
    if rf_n_jobs is None:
        total_cpus = os.cpu_count() or 1
        rf_n_jobs  = max(1, total_cpus // args.workers)

    # -- Load + filter metadata ----------------------------------------------
    try:
        logger.info("Loading train metadata : %s", args.train_metadata)
        train_configs = load_metadata_file(args.train_metadata, role="train")
        logger.info("Loading test  metadata : %s", args.test_metadata)
        test_configs_raw = load_metadata_file(args.test_metadata, role="test")
        test_configs = _filter_test_configs(train_configs, test_configs_raw)
    except ConfigError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    # -- Extract features for every dataset in both pools --------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_all(configs: List[DatasetConfig], role: str) -> List[DatasetFeatures]:
        print()
        print(_sep("-"))
        print(f"  FEATURE EXTRACTION -- {role.upper()} POOL")
        print(_sep("-"))
        feats: List[DatasetFeatures] = []
        for cfg in configs:
            print(f"  Loading {role} dataset: {cfg.name} ...")
            try:
                ds_features = extract_dataset_features(
                    cfg             = cfg,
                    cache_dir       = args.cache_dir,
                    min_total_reads = args.min_total_reads,
                    min_reads       = args.min_reads,
                )
            except (DataLoadError, BruteForceError) as exc:
                print(
                    f"\nERROR: Failed to extract features for {role} "
                    f"dataset '{cfg.name}': {exc}",
                    file=sys.stderr,
                )
                raise
            feats.append(ds_features)
            src = "(from cache)" if args.cache_dir else ""
            print(
                f"    [OK] {ds_features.dataset_name:<20}  "
                f"{ds_features.n_genes:>6,} genes   "
                f"{ds_features.n_targets:>4} targets  {src}"
            )
        return feats

    try:
        train_features = _extract_all(train_configs, role="train")
        test_features  = _extract_all(test_configs,  role="test")
    except (DataLoadError, BruteForceError):
        return 1

    if not train_features:
        print("ERROR: Train pool produced no extractable datasets.", file=sys.stderr)
        return 1
    if not test_features:
        print("ERROR: Test pool produced no extractable datasets.", file=sys.stderr)
        return 1

    # -- Startup banner -------------------------------------------------------
    _print_startup_banner(
        train_datasets = train_features,
        test_datasets  = test_features,
        trees          = args.trees,
        k_min          = args.k_min,
        k_max          = args.k_max,
        workers        = args.workers,
        rf_n_jobs      = rf_n_jobs,
        output_dir     = args.output_dir,
        model          = args.model,
        resume         = args.resume,
    )

    # -- Run all tree sizes ---------------------------------------------------
    index_writer = IndexWriter(
        output_dir  = args.output_dir,
        flush_every = args.index_flush_every,
    )

    t_start = time.time()
    for tree_idx, n_trees in enumerate(args.trees):
        if len(args.trees) > 1:
            print(f"\n{'='*_SEP_WIDTH}")
            print(f"  TREE SIZE {tree_idx+1}/{len(args.trees)} -- n_trees={n_trees}")
            print(f"{'='*_SEP_WIDTH}")

        run_tree_size(
            n_trees         = n_trees,
            train_datasets  = train_features,
            test_datasets   = test_features,
            output_dir      = args.output_dir,
            k_min           = args.k_min,
            k_max           = args.k_max,
            workers         = args.workers,
            rf_n_jobs       = rf_n_jobs,
            use_xgb         = (args.model == "xgb"),
            use_gpu         = args.gpu,
            resume          = args.resume,
            index_writer    = index_writer,
            batch_size      = args.batch_size,
            skip_degenerate = args.skip_degenerate,
        )

    index_writer.close()
    elapsed = time.time() - t_start

    print(_sep("="))
    print(f"  ALL DONE  --  {elapsed/60:.1f} min total")
    print(f"  Results index : {args.output_dir / 'results_index.parquet'}")
    print(
        f"  Query results : python -m brute_force.query "
        f"--index {args.output_dir}/results_index.parquet --top 20"
    )
    print(_sep("="))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
