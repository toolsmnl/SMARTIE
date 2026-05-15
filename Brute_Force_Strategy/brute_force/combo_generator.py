"""Lazy C(14, k) combination iterator with deterministic hashing.

No combination is ever materialised in memory as a list; callers receive
an iterator.  The hash function is stable across Python versions and
machines (sha1 on sorted feature names).
"""

from __future__ import annotations

import hashlib
import itertools
import math
from typing import Iterator, List, Tuple

from .feature_registry import VARIABLE_FEATURES


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------

def combo_hash(feature_names: List[str]) -> str:
    """Return a stable 12-character hex digest for a set of feature names.

    Input order does not matter; names are sorted before hashing so the same
    logical feature set always maps to the same directory name.
    """
    key = "|".join(sorted(feature_names))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def total_combos(k: int) -> int:
    """Return C(11, k) -- the number of variable-feature combinations of size k."""
    n = len(VARIABLE_FEATURES)
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def total_combos_in_range(k_min: int, k_max: int) -> int:
    """Total combinations across all k in [k_min, k_max] inclusive."""
    return sum(total_combos(k) for k in range(k_min, k_max + 1))


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------

def iter_variable_combos(k: int) -> Iterator[Tuple[str, ...]]:
    """Yield all C(44, k) tuples of variable feature names for a given k.

    Each yielded tuple is in the canonical order defined by VARIABLE_FEATURES
    (i.e. the same relative order, not sorted alphabetically).  This makes
    the combo_hash of each tuple deterministic.
    """
    if k < 0 or k > len(VARIABLE_FEATURES):
        return
    yield from itertools.combinations(VARIABLE_FEATURES, k)


def iter_all_combos(
    k_min: int = 1,
    k_max: int = len(VARIABLE_FEATURES),
) -> Iterator[Tuple[int, Tuple[str, ...]]]:
    """Yield (k, variable_feature_tuple) for all k in [k_min, k_max].

    Combinations are yielded in ascending k order.  Within each k, they
    follow the canonical VARIABLE_FEATURES ordering (lexicographic over
    index positions, not names).

    Parameters
    ----------
    k_min:
        Minimum number of variable features in a combo (folder = k_min + 9).
    k_max:
        Maximum number of variable features (folder = k_max + 9).
        Defaults to 44 (all variable features).
    """
    n = len(VARIABLE_FEATURES)
    k_min = max(0, k_min)
    k_max = min(n, k_max)

    for k in range(k_min, k_max + 1):
        for combo in itertools.combinations(VARIABLE_FEATURES, k):
            yield k, combo


# ---------------------------------------------------------------------------
# Combo metadata helpers
# ---------------------------------------------------------------------------

def describe_combo(variable_features: Tuple[str, ...]) -> dict:
    """Return a dict suitable for writing to combo_metadata.json."""
    from .feature_registry import CORE_FEATURES, get_combo_features

    full_features = get_combo_features(list(variable_features))
    return {
        "combo_hash":        combo_hash(full_features),
        "n_features":        len(full_features),
        "core_features":     list(CORE_FEATURES),
        "variable_features": list(variable_features),
        "all_features":      full_features,
    }
