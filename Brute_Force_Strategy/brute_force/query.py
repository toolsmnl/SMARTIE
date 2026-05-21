"""CLI query tool for filtering the completed results index.

Usage
-----
    python -m brute_force.query --help
    python -m brute_force.query \\
        --index results/results_index.parquet \\
        --recall-at 20 --min-recall 0.75 \\
        --must-contain fold_change \\
        --trees 200 \\
        --sort-by mean_recall_at_20pct \\
        --top 20

All filter flags are AND-ed together.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .exceptions import QueryError
from .feature_registry import ALL_FEATURE_SET, FEATURE_DESCRIPTIONS


# ---------------------------------------------------------------------------
# Core filter function
# ---------------------------------------------------------------------------

def query_index(
    index_path: Path,
    precision_filters: Optional[List[tuple]] = None,
    importance_filters: Optional[List[tuple]] = None,
    must_contain: Optional[List[str]] = None,
    must_not_contain: Optional[List[str]] = None,
    trees: Optional[int] = None,
    n_features: Optional[int] = None,
    sort_by: str = "mean_precision_at_20pct",
    top: Optional[int] = None,
) -> pd.DataFrame:
    """Load and filter the results index.

    Parameters
    ----------
    index_path:
        Path to results_index.parquet.
    precision_filters:
        List of (pct: int, min_precision: float) tuples.
        E.g. [(20, 0.75), (10, 0.60)] keeps rows where
        mean_precision_at_20pct >= 0.75 AND mean_precision_at_10pct >= 0.60.
    importance_filters:
        List of (feature_name: str, min_importance: float) tuples.
        Filters on the ``mean_feature_importance`` dict serialised into
        the per-combo summary (stored flat in the index as ``imp_{name}``
        columns if present, otherwise skipped with a warning).
    must_contain:
        Combo must include all these feature names.
    must_not_contain:
        Combo must include none of these feature names.
    trees:
        Filter to exactly this n_trees value.
    n_features:
        Filter to exactly this n_features value.
    sort_by:
        Column to sort descending by.
    top:
        Return only the top N rows.

    Returns
    -------
    Filtered and sorted DataFrame.

    Raises
    ------
    QueryError
        If the index file is missing or unreadable.
    """
    if not index_path.exists():
        raise QueryError(
            f"Results index not found: {index_path}\n"
            "Run the pipeline first to generate results."
        )

    try:
        df = pd.read_parquet(index_path, engine="pyarrow")
    except Exception as exc:
        raise QueryError(f"Could not read index '{index_path}': {exc}") from exc

    if df.empty:
        return df

    # -- Exact-value filters --------------------------------------------------
    if trees is not None:
        df = df[df["n_trees"] == trees]
    if n_features is not None:
        df = df[df["n_features"] == n_features]

    # -- Precision filters ----------------------------------------------------
    for pct, min_val in (precision_filters or []):
        col = f"mean_precision_at_{pct}pct"
        if col not in df.columns:
            raise QueryError(
                f"Column '{col}' not found in index.  "
                f"Available precision columns: "
                f"{[c for c in df.columns if 'precision' in c]}"
            )
        df = df[df[col] >= min_val]

    # -- Feature-presence filters ---------------------------------------------
    if must_contain:
        for feat in must_contain:
            if feat not in ALL_FEATURE_SET:
                raise QueryError(f"Unknown feature name: '{feat}'")
            df = df[df["features"].str.contains(feat, regex=False)]

    if must_not_contain:
        for feat in must_not_contain:
            if feat not in ALL_FEATURE_SET:
                raise QueryError(f"Unknown feature name: '{feat}'")
            df = df[~df["features"].str.contains(feat, regex=False)]

    # -- Feature importance filters -------------------------------------------
    for feat, min_imp in (importance_filters or []):
        imp_col = f"imp_{feat}"
        if imp_col not in df.columns:
            print(
                f"Warning: importance column '{imp_col}' not in index "
                f"(feature importance columns start with 'imp_'); skipping.",
                file=sys.stderr,
            )
            continue
        df = df[df[imp_col] >= min_imp]

    # -- Sort -----------------------------------------------------------------
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    else:
        print(
            f"Warning: sort column '{sort_by}' not in index; "
            f"returning unsorted.",
            file=sys.stderr,
        )

    if top is not None:
        df = df.head(top)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m brute_force.query",
        description="Filter and search the brute-force combo results index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Best 20 combos at 20% precision threshold with 200 trees
  python -m brute_force.query \\
      --index results/results_index.parquet \\
      --precision-at 20 --min-precision 0.75 --trees 200 --top 20

  # Combos containing fold_change with high AUC
  python -m brute_force.query \\
      --index results/results_index.parquet \\
      --must-contain fold_change \\
      --sort-by mean_auc --top 50 --output top50.csv

  # Combos where log_true_signal has high importance
  python -m brute_force.query \\
      --index results/results_index.parquet \\
      --feature-importance log_true_signal --min-importance 0.10 \\
      --sort-by mean_precision_at_20pct
""",
    )
    p.add_argument(
        "--index", required=True, type=Path,
        metavar="FILE",
        help="Path to results_index.parquet.",
    )
    p.add_argument(
        "--precision-at", type=int, action="append", dest="precision_thresholds",
        metavar="PCT",
        help="Precision threshold % (repeat for multiple, paired with --min-precision).",
    )
    p.add_argument(
        "--min-precision", type=float, action="append", dest="min_precisions",
        metavar="V",
        help="Minimum precision value for the corresponding --precision-at threshold.",
    )
    p.add_argument(
        "--feature-importance", type=str, action="append", dest="imp_features",
        metavar="FEAT",
        help="Feature name for importance filter (paired with --min-importance).",
    )
    p.add_argument(
        "--min-importance", type=float, action="append", dest="min_importances",
        metavar="V",
        help="Minimum importance for the corresponding --feature-importance feature.",
    )
    p.add_argument(
        "--must-contain", nargs="+", metavar="FEAT",
        help="Only show combos that include ALL of these features.",
    )
    p.add_argument(
        "--must-not-contain", nargs="+", metavar="FEAT",
        help="Exclude combos that include ANY of these features.",
    )
    p.add_argument(
        "--trees", type=int, metavar="N",
        help="Filter to a specific n_trees value.",
    )
    p.add_argument(
        "--n-features", type=int, metavar="N",
        help="Filter to a specific total feature count.",
    )
    p.add_argument(
        "--sort-by", default="mean_precision_at_20pct", metavar="COL",
        help="Sort descending by this column (default: mean_precision_at_20pct).",
    )
    p.add_argument(
        "--top", type=int, metavar="N",
        help="Return only the top N results.",
    )
    p.add_argument(
        "--output", type=Path, metavar="FILE",
        help="Save results as CSV.  Prints to stdout if omitted.",
    )
    p.add_argument(
        "--list-features", action="store_true",
        help="Print all valid feature names with descriptions and exit.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.list_features:
        print(f"{'Feature name':<40} Description")
        print("-" * 80)
        for name, desc in FEATURE_DESCRIPTIONS.items():
            print(f"  {name:<38} {desc}")
        return 0

    # Validate paired args.
    precision_filters: List[tuple] = []
    if args.precision_thresholds or args.min_precisions:
        thresholds = args.precision_thresholds or []
        min_vals   = args.min_precisions or []
        if len(thresholds) != len(min_vals):
            parser.error(
                "--precision-at and --min-precision must be provided in equal numbers."
            )
        precision_filters = list(zip(thresholds, min_vals))

    importance_filters: List[tuple] = []
    if args.imp_features or args.min_importances:
        feats     = args.imp_features or []
        min_imps  = args.min_importances or []
        if len(feats) != len(min_imps):
            parser.error(
                "--feature-importance and --min-importance must be provided in equal numbers."
            )
        importance_filters = list(zip(feats, min_imps))

    try:
        result = query_index(
            index_path          = args.index,
            precision_filters   = precision_filters,
            importance_filters  = importance_filters,
            must_contain       = args.must_contain,
            must_not_contain   = args.must_not_contain,
            trees              = args.trees,
            n_features         = args.n_features,
            sort_by            = args.sort_by,
            top                = args.top,
        )
    except QueryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if result.empty:
        print("No results match the given filters.", file=sys.stderr)
        return 0

    print(f"Found {len(result)} matching combos.", file=sys.stderr)

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Saved to: {args.output}", file=sys.stderr)
    else:
        print(result.to_string(index=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
