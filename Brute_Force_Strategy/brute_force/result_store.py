"""Persistent result storage: atomic writes, resume logic, and the flat index.

All public functions are thread-safe (they use file-level atomic rename or
a threading.Lock around the in-memory index buffer).

Directory layout written by this module
----------------------------------------
results/
  results_index.parquet          <- rolling flat index, one row per combo
  cache/                         <- feature matrix cache (managed by feature_extractor)
  {n_trees}/
    {n_features}/
      {combo_hash}/
        combo_metadata.json
        combo_summary.json
        train_{dataset}/
          heatmap.png
          heatmap_data.csv
          feature_importance.png
          feature_importance.csv
          metrics.json
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .exceptions import ResultStoreError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def combo_dir(
    output_dir: Path,
    n_trees: int,
    n_features: int,
    combo_hash: str,
) -> Path:
    return output_dir / str(n_trees) / str(n_features) / combo_hash


def train_subdir(combo_path: Path, dataset_name: str) -> Path:
    return combo_path / f"train_{dataset_name}"


# ---------------------------------------------------------------------------
# Resume: scan for completed combos
# ---------------------------------------------------------------------------

def load_completed_hashes(
    output_dir: Path,
    n_trees: int,
    n_features: int,
) -> Set[str]:
    """Return the set of combo hashes that already have a combo_summary.json.

    This is the single O(n_existing) scan done at startup; subsequent checks
    are O(1) set lookups.
    """
    folder = output_dir / str(n_trees) / str(n_features)
    if not folder.exists():
        return set()

    completed: Set[str] = set()
    for entry in folder.iterdir():
        if entry.is_dir() and (entry / "combo_summary.json").exists():
            completed.add(entry.name)

    logger.debug(
        "n_trees=%d n_features=%d: %d completed combos found.",
        n_trees, n_features, len(completed),
    )
    return completed


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------

def _write_json_atomic(path: Path, data: dict) -> None:
    """Write a dict as JSON to path atomically (write-temp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=_json_default)
        os.replace(tmp_path, path)  # atomic on POSIX; best-effort on Windows
    except Exception as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise ResultStoreError(f"Failed to write JSON '{path}': {exc}") from exc


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Per-combo result writing
# ---------------------------------------------------------------------------

def save_combo_results(
    output_dir: Path,
    n_trees: int,
    combo_hash: str,
    combo_metadata: dict,
    combo_summary: dict,
    per_train_results: Dict[str, dict],
    generate_plots: bool = False,
) -> None:
    """Write all files for one combo atomically.

    Results are staged in a temporary directory first; the directory is
    renamed to the final combo path only when all writes succeed.  A crash
    mid-write therefore leaves no partial directory that could confuse the
    resume scan.

    Parameters
    ----------
    per_train_results:
        Mapping from training dataset name to a dict with keys:
        ``heatmap_df`` (pd.DataFrame), ``feature_importance_df`` (pd.DataFrame),
        ``metrics`` (dict).
    generate_plots:
        Generate PNG heatmap and importance plots per combo.  Disabled by
        default because running matplotlib inside hundreds of parallel
        workers exhausts memory quickly.  Use the post-processing query
        tool to generate plots for the top-N combos after the run.
    """
    n_features  = combo_metadata["n_features"]
    final_path  = combo_dir(output_dir, n_trees, n_features, combo_hash)
    parent      = final_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Use a temp dir in the same parent so rename is atomic on most OSes.
    tmp_dir = Path(tempfile.mkdtemp(dir=parent, prefix=f"{combo_hash}.tmp."))
    try:
        _write_json_atomic(tmp_dir / "combo_metadata.json", combo_metadata)
        _write_json_atomic(tmp_dir / "combo_summary.json",  combo_summary)

        for dataset_name, res in per_train_results.items():
            sub = tmp_dir / f"train_{dataset_name}"
            sub.mkdir(parents=True, exist_ok=True)

            # Heatmap data
            heatmap_df: pd.DataFrame = res["heatmap_df"]
            heatmap_df.to_csv(sub / "heatmap_data.csv")

            # Feature importance data
            imp_df: pd.DataFrame = res["feature_importance_df"]
            imp_df.to_csv(sub / "feature_importance.csv", index=False)

            # Metrics JSON
            _write_json_atomic(sub / "metrics.json", res["metrics"])

            if generate_plots:
                try:
                    _save_heatmap_plot(heatmap_df, sub / "heatmap.png")
                    _save_importance_plot(imp_df, sub / "feature_importance.png")
                except Exception as exc:
                    logger.warning(
                        "Plot generation failed for combo %s / %s: %s",
                        combo_hash, dataset_name, exc,
                    )

        # Atomic rename: only after all writes succeed.
        if final_path.exists():
            shutil.rmtree(final_path)  # overwrite if re-running
        os.replace(str(tmp_dir), str(final_path))

    except Exception:
        # Clean up staging dir on failure.
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Plot generation (runs inside worker process)
# ---------------------------------------------------------------------------

def _save_heatmap_plot(df: pd.DataFrame, path: Path) -> None:
    """Save the precision@k heatmap as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 0.9), max(4, len(df) * 0.6 + 1)))
    # Rename columns to short labels: precision_at_20pct -> @20%
    display_cols = [
        c.replace("precision_at_", "@").replace("pct", "%")
        for c in df.columns
    ]
    plot_df = df.copy()
    plot_df.columns = display_cols

    sns.heatmap(
        plot_df,
        ax=ax,
        vmin=0.0, vmax=1.0,
        cmap="YlOrRd",
        annot=True, fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Precision"},
    )
    ax.set_xlabel("Prediction depth (k = top X% of targets)")
    ax.set_ylabel("Test dataset")
    ax.set_title("Precision@k heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_importance_plot(imp_df: pd.DataFrame, path: Path) -> None:
    """Save the feature importance bar chart as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # imp_df has columns: feature, importance (sorted descending already).
    n = len(imp_df)
    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.35 + 1)))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#4878d0")
    ax.invert_yaxis()
    ax.set_xlabel("Mean decrease in impurity (normalised)")
    ax.set_title("Feature importances")
    ax.set_xlim(0, max(imp_df["importance"].max() * 1.1, 0.01))
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rolling results index
# ---------------------------------------------------------------------------

class IndexWriter:
    """Thread-safe rolling writer for results_index.parquet.

    Records are buffered in memory and flushed to disk every
    ``flush_every`` records or on explicit flush() / close().
    """

    def __init__(self, output_dir: Path, flush_every: int = 50) -> None:
        self._path        = output_dir / "results_index.parquet"
        self._flush_every = flush_every
        self._buffer: List[dict] = []
        self._lock        = threading.Lock()

    def append(self, record: dict) -> None:
        """Add one record.  Flushes automatically when buffer is full."""
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._flush_every:
                self._flush_locked()

    def flush(self) -> None:
        """Force-flush the current buffer to disk."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush remaining records and release resources."""
        self.flush()

    def _flush_locked(self) -> None:
        """Must be called with self._lock held."""
        if not self._buffer:
            return
        new_df = pd.DataFrame(self._buffer)
        self._buffer = []

        if self._path.exists():
            try:
                existing = pd.read_parquet(self._path, engine="pyarrow")
                combined = pd.concat([existing, new_df], ignore_index=True)
            except Exception as exc:
                logger.warning(
                    "Could not read existing index ('%s'); creating fresh: %s",
                    self._path, exc,
                )
                combined = new_df
        else:
            combined = new_df

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, suffix=".tmp.parquet"
        )
        os.close(tmp_fd)
        try:
            combined.to_parquet(tmp_path, index=False, engine="pyarrow")
            os.replace(tmp_path, self._path)
        except Exception as exc:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            logger.error("Failed to flush results index: %s", exc)


def build_index_record(
    n_trees: int,
    combo_metadata: dict,
    combo_summary: dict,
) -> dict:
    """Flatten combo metadata and summary into a single index row."""
    record: dict = {
        "n_trees":      n_trees,
        "n_features":   combo_metadata["n_features"],
        "combo_hash":   combo_metadata["combo_hash"],
        "features":     "|".join(combo_metadata["all_features"]),
        "variable_features": "|".join(combo_metadata["variable_features"]),
    }

    # Flatten top-level summary metrics.
    for k, v in combo_summary.items():
        if isinstance(v, (int, float)):
            record[k] = v

    return record


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Thread-safe counter that writes progress.json periodically."""

    def __init__(
        self,
        output_dir: Path,
        total: int,
        flush_interval_s: float = 60.0,
    ) -> None:
        self._output_dir      = output_dir
        self._total           = total
        self._completed       = 0
        self._started_at      = time.time()
        self._last_flush      = 0.0
        self._flush_interval  = flush_interval_s
        self._lock            = threading.Lock()

    def increment(self) -> None:
        with self._lock:
            self._completed += 1
            now = time.time()
            if now - self._last_flush >= self._flush_interval:
                self._write_locked(now)
                self._last_flush = now

    def finalize(self) -> None:
        with self._lock:
            self._write_locked(time.time())

    def _write_locked(self, now: float) -> None:
        elapsed   = now - self._started_at
        remaining = self._total - self._completed
        eta       = (elapsed / self._completed * remaining) if self._completed else None

        data = {
            "total":       self._total,
            "completed":   self._completed,
            "elapsed_s":   round(elapsed, 1),
            "eta_s":       round(eta, 0) if eta is not None else None,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        try:
            _write_json_atomic(self._output_dir / "progress.json", data)
        except Exception as exc:
            logger.debug("Progress write failed: %s", exc)
