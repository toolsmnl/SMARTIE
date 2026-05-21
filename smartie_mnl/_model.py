"""
Locate and load the bundled pre-trained SMARTIE model.
"""

from __future__ import annotations

import pickle
from pathlib import Path


_BUNDLED_MODEL = Path(__file__).parent / "models" / "SMARTIE.pkl"


def get_bundled_model_path() -> Path:
    """Return the path to the bundled SMARTIE.pkl shipped with the package."""
    if not _BUNDLED_MODEL.exists():
        raise FileNotFoundError(
            f"Bundled model not found at {_BUNDLED_MODEL}. "
            "Re-install the package or upload a model file manually."
        )
    return _BUNDLED_MODEL


def load_bundled_model():
    """Load and return the bundled pre-trained SMARTIE scikit-learn pipeline."""
    with get_bundled_model_path().open("rb") as fh:
        return pickle.load(fh)
