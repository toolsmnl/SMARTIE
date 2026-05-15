"""Dataclasses and type aliases used throughout the brute_force package.

Keeping types in one module prevents circular imports and makes the data
contracts between modules explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dataset configuration (parsed from datasets.tsv)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """Paths and metadata for one RBP dataset."""
    name: str
    expt_files: Tuple[Path, ...]
    ctrl_files: Tuple[Path, ...]
    targets_file: Path

    def __post_init__(self) -> None:
        if not self.expt_files:
            raise ValueError(f"Dataset '{self.name}': at least one expt file required.")
        if not self.ctrl_files:
            raise ValueError(f"Dataset '{self.name}': at least one ctrl file required.")


# ---------------------------------------------------------------------------
# Per-replicate gene-level statistics (intermediate; not exposed to callers)
# ---------------------------------------------------------------------------

@dataclass
class _ReplicateStats:
    """Gene-level statistics extracted from a single replicate file.

    All arrays are aligned to the shared gene universe (same length, same order).
    Genes not detected in this replicate have 0 in all numeric arrays.
    """
    # shape (N_genes,) for all arrays
    sum_edits: np.ndarray       # total edited reads per gene
    sum_total: np.ndarray       # total read coverage per gene
    n_sites: np.ndarray         # number of editing sites detected
    gene_rate: np.ndarray       # sum_edits / sum_total
    mean_site_rate: np.ndarray  # mean of per-site edit rates within gene
    median_site_rate: np.ndarray
    max_site_rate: np.ndarray
    site_edit_sd: np.ndarray    # std of per-site edit rates within gene
    cum_edit_pct: np.ndarray    # sum of (per-site edit rate x 100)
    norm_edit_frac: np.ndarray  # sum_edits / sample_total_edits
    site_frac: np.ndarray       # n_sites / sample_total_sites

    # scalars
    sample_total_edits: float
    sample_total_sites: int


# ---------------------------------------------------------------------------
# Fully computed dataset features (output of feature_extractor)
# ---------------------------------------------------------------------------

@dataclass
class DatasetFeatures:
    """All 53 features for every gene in one dataset, ready for training."""
    dataset_name: str
    genes: np.ndarray            # shape (N_genes,) dtype object (gene names)
    feature_matrix: np.ndarray   # shape (N_genes, 53) dtype float32
    feature_names: List[str]     # length 53, matches columns of feature_matrix
    is_target: np.ndarray        # shape (N_genes,) dtype bool

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def n_targets(self) -> int:
        return int(self.is_target.sum())

    def select_features(self, names: List[str]) -> np.ndarray:
        """Return a (N_genes, len(names)) sub-matrix for the requested features."""
        idx = [self.feature_names.index(n) for n in names]
        return self.feature_matrix[:, idx]


# ---------------------------------------------------------------------------
# Training outputs
# ---------------------------------------------------------------------------

@dataclass
class RankedGene:
    gene: str
    score: float          # RF predicted probability of being a target
    is_target: bool


@dataclass
class TrainResult:
    """Output of training a model on one dataset and predicting on another."""
    train_dataset: str
    test_dataset: str
    feature_names: List[str]
    n_trees: int
    ranked_genes: List[RankedGene]   # descending by score
    feature_importances: Dict[str, float]
    n_genes_train: int
    n_targets_train: int


# ---------------------------------------------------------------------------
# Evaluation outputs
# ---------------------------------------------------------------------------

# Precision at each threshold: key = "precision_at_Xpct", value in [0, 1]
PrecisionCurve = Dict[str, float]

@dataclass
class ComboResult:
    """Aggregated results for one (n_trees, feature_combo) pair."""
    n_trees: int
    feature_names: List[str]            # full list including core
    variable_features: List[str]        # only the variable portion
    combo_hash: str
    # per training dataset -> per test dataset -> recall curve
    per_train: Dict[str, Dict[str, RecallCurve]] = field(default_factory=dict)
    # per training dataset -> feature importances
    feature_importances: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # aggregated across all (train, test) pairs
    summary: Dict[str, float] = field(default_factory=dict)
