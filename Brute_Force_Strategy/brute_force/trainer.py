"""Model training and gene ranking.

Wraps scikit-learn RandomForestClassifier (default) and optionally XGBoost.
The caller decides how many n_jobs to give the RF based on the outer
parallelism level (see OPTIMIZATIONS.md Sec.3.3).
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Falling back to prediction using DMatrix.*",
    category=UserWarning,
)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from ._types import DatasetFeatures, RankedGene, TrainResult
from .exceptions import BruteForceError

logger = logging.getLogger(__name__)

# XGBoost is optional; import lazily so the rest of the pipeline works
# without it installed.
_xgb_available: Optional[bool] = None

def _try_import_xgb():
    global _xgb_available
    if _xgb_available is None:
        try:
            import xgboost  # noqa: F401
            _xgb_available = True
        except ImportError:
            _xgb_available = False
    return _xgb_available


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_and_rank(
    train_features: DatasetFeatures,
    test_features:  DatasetFeatures,
    feature_subset: List[str],
    n_trees: int,
    rf_n_jobs: int = 1,
    use_xgb: bool = False,
    use_gpu: bool = False,
) -> TrainResult:
    """Train on ``train_features`` and rank genes in ``test_features``.

    Parameters
    ----------
    train_features:
        Features and labels for the training dataset.
    test_features:
        Features (and labels for evaluation) for the test dataset.
        Can be the same object as ``train_features`` for training-set evaluation.
    feature_subset:
        Ordered list of feature names to use (must be a subset of ALL_FEATURES).
    n_trees:
        Number of trees in the Random Forest.
    rf_n_jobs:
        Parallelism inside the RF fit; tune alongside outer combo-level
        parallelism to avoid over-subscription.
    use_xgb:
        Use XGBoost instead of Random Forest.
    use_gpu:
        Pass ``device='cuda'`` to XGBoost (ignored for RF).

    Returns
    -------
    TrainResult with ranked genes for the test dataset.

    Raises
    ------
    BruteForceError
        If training fails (e.g. no positive labels in training set).
    """
    # -- Build training matrix ------------------------------------------------
    X_train = train_features.select_features(feature_subset).astype(np.float32)
    y_train = train_features.is_target.astype(np.int32)

    n_pos = int(y_train.sum())
    n_neg = int((~train_features.is_target).sum())

    if n_pos == 0:
        raise BruteForceError(
            f"Dataset '{train_features.dataset_name}' has 0 positive (target) "
            f"labels in its feature matrix.  Cannot train."
        )
    if n_neg == 0:
        raise BruteForceError(
            f"Dataset '{train_features.dataset_name}' has 0 negative labels. "
            f"Cannot train."
        )

    # Replace any remaining NaN/Inf with 0 (safety net).
    if not np.isfinite(X_train).all():
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # -- Fit model -----------------------------------------------------------
    model = _fit_model(
        X_train, y_train, n_trees, rf_n_jobs, use_xgb, use_gpu,
        train_dataset=train_features.dataset_name,
    )

    # -- Score test dataset ---------------------------------------------------
    X_test = test_features.select_features(feature_subset).astype(np.float32)
    if not np.isfinite(X_test).all():
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    scores = _predict_proba(model, X_test, use_xgb, use_gpu)

    # -- Build ranked gene list -----------------------------------------------
    order = np.argsort(scores)[::-1]
    ranked_genes: List[RankedGene] = [
        RankedGene(
            gene=str(test_features.genes[i]),
            score=float(scores[i]),
            is_target=bool(test_features.is_target[i]),
        )
        for i in order
    ]

    # -- Feature importances -------------------------------------------------
    importances = _get_importances(model, feature_subset, use_xgb)

    return TrainResult(
        train_dataset=train_features.dataset_name,
        test_dataset=test_features.dataset_name,
        feature_names=list(feature_subset),
        n_trees=n_trees,
        ranked_genes=ranked_genes,
        feature_importances=importances,
        n_genes_train=len(y_train),
        n_targets_train=n_pos,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    n_trees: int,
    n_jobs: int,
    use_xgb: bool,
    use_gpu: bool,
    train_dataset: str,
):
    """Return a fitted classifier."""
    if use_xgb:
        if not _try_import_xgb():
            raise BruteForceError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )
        import xgboost as xgb

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        scale_pos = n_neg / max(n_pos, 1)

        params: dict = {
            "n_estimators":     n_trees,
            "tree_method":      "hist",
            "scale_pos_weight": scale_pos,
            "eval_metric":      "logloss",
            "random_state":     42,
            "n_jobs":           n_jobs,
        }
        if use_gpu:
            params["device"] = "cuda"

        clf = xgb.XGBClassifier(**params)
        clf.fit(X, y)
        logger.debug(
            "XGB fit: dataset=%s, n_trees=%d, scale_pos=%.2f",
            train_dataset, n_trees, scale_pos,
        )
    else:
        # Balanced class weights handle the typical 1:100+ target/background ratio.
        clf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=n_jobs,
        )
        clf.fit(X, y)
        logger.debug(
            "RF fit: dataset=%s, n_trees=%d, n_pos=%d, n_neg=%d",
            train_dataset, n_trees, int(y.sum()), len(y) - int(y.sum()),
        )
    return clf


def _predict_proba(model, X: np.ndarray, use_xgb: bool, use_gpu: bool = False) -> np.ndarray:
    """Return predicted probability of class 1 (shape N_genes,)."""
    if use_xgb and use_gpu:
        try:
            import cupy as cp
            import xgboost as xgb
            dmatrix = xgb.DMatrix(cp.array(X))
            scores = model.get_booster().predict(dmatrix)
            return scores.astype(np.float32)
        except ImportError:
            # cupy not installed — fall through to standard predict_proba
            pass
    proba = model.predict_proba(X)
    # Both RF and XGB return shape (N, 2) with [:, 1] = P(positive).
    return proba[:, 1].astype(np.float32)


def _get_importances(model, feature_names: List[str], use_xgb: bool) -> Dict[str, float]:
    """Extract feature importances as a name -> value dict."""
    if use_xgb:
        raw = model.get_booster().get_score(importance_type="gain")
        # XGB names features f0, f1, ...; map back to our names.
        importances = {}
        for i, name in enumerate(feature_names):
            key = f"f{i}"
            importances[name] = float(raw.get(key, 0.0))
    else:
        importances = {
            name: float(imp)
            for name, imp in zip(feature_names, model.feature_importances_)
        }

    # Normalise so importances sum to 1.0 (RF already does this, XGB may not).
    total = sum(importances.values())
    if total > 0.0:
        importances = {k: v / total for k, v in importances.items()}
    return importances
