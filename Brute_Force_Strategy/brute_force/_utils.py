"""Shared low-level math helpers used across the package.

All scalar and array operations here are pure functions with no side effects.
Floor values match the convention used in the original feature_extraction.py.
"""

from __future__ import annotations

import math

import numpy as np

_LOG2_FLOOR = -20.0
_LOG10_FLOOR = 1e-300  # smallest value passed to log10 (not floored to -20)


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def safe_log2(x: float, floor: float = _LOG2_FLOOR) -> float:
    """log2(x), returning floor when x <= 0."""
    if x <= 0.0:
        return floor
    return math.log2(x)


def safe_log2_ratio(num: float, denom: float, floor: float = _LOG2_FLOOR) -> float:
    """log2(num / denom), returning floor when either is <= 0."""
    if denom <= 0.0 or num <= 0.0:
        return floor
    return math.log2(num / denom)


def safe_div(num: float, denom: float, fallback: float = 0.0) -> float:
    """num / denom, returning fallback when denom == 0."""
    return num / denom if denom != 0.0 else fallback


# ---------------------------------------------------------------------------
# Vectorised helpers  (operate on 1-D float32/float64 numpy arrays)
# ---------------------------------------------------------------------------

def vec_safe_log2(arr: np.ndarray, floor: float = _LOG2_FLOOR) -> np.ndarray:
    """Element-wise log2, flooring non-positive values."""
    out = np.full(len(arr), floor, dtype=np.float32)
    mask = arr > 0.0
    out[mask] = np.log2(arr[mask]).astype(np.float32)
    return out


def vec_safe_div(
    num: np.ndarray,
    denom: np.ndarray,
    fallback: float = 0.0,
) -> np.ndarray:
    """Element-wise num / denom, substituting fallback where denom == 0."""
    num64   = np.asarray(num,   dtype=np.float64)
    denom64 = np.asarray(denom, dtype=np.float64)
    out  = np.full(len(num64), fallback, dtype=np.float64)
    mask = denom64 != 0.0
    out[mask] = num64[mask] / denom64[mask]
    return out.astype(np.float32)


def vec_safe_log10_neg(arr: np.ndarray, floor: float = 300.0) -> np.ndarray:
    """-log10(arr), returning floor for arr <= 0.

    Works in float64 internally so that float32 inputs whose small positive
    values underflowed to 0.0 are treated as zero (get the floor) rather than
    triggering a divide-by-zero warning.
    """
    arr64 = np.asarray(arr, dtype=np.float64)
    out   = np.full(len(arr64), floor, dtype=np.float32)
    mask  = arr64 > 0.0
    out[mask] = (-np.log10(arr64[mask])).astype(np.float32)
    return out
