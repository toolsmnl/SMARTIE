"""
utils.py — shared helpers for the smartie Streamlit app.

Handles:
  - Saving uploaded files to temp dirs
  - Loading & caching the pipeline modules (feature_extraction, train_model, etc.)
  - Capturing stdout/stderr from pipeline calls so we can show progress in the UI
  - Zip-bundling output directories for download
  - Displaying plots produced by the pipeline
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import traceback
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import streamlit as st


# ── Pipeline module loader ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline_modules():
    """Import the three pipeline modules and return them.
    Cached so import only happens once per Streamlit session.

    XGBoost requires OpenMP (libomp on Mac, libgomp on Linux, vcomp on Windows).
    If it is missing the import fails with a native library error.  We catch that
    here so the rest of the app still loads, and show a clear install instruction
    rather than a raw traceback.
    """
    try:
        from smartie_mnl import feature_extraction as fe
    except ImportError as exc:
        st.error(
            f"❌ Could not import feature_extraction: {exc}. "
            "Make sure the package is installed: pip install smartie-mnl"
        )
        st.stop()

    try:
        from smartie_mnl import train_model as tm
    except Exception as exc:
        _msg = str(exc)
        if "libxgboost" in _msg or "libomp" in _msg or "XGBoost" in _msg or "vcomp" in _msg:
            st.error(
                "❌ **XGBoost could not load its native library (OpenMP missing).**\n\n"
                "Fix for your operating system:\n"
                "- **macOS:** `brew install libomp`\n"
                "- **Linux:** `sudo apt install libgomp1` (Debian/Ubuntu) "
                "or `sudo yum install libgomp` (RHEL/CentOS)\n"
                "- **Windows:** install the "
                "[Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)\n\n"
                "After installing, restart SMARTIE."
            )
        else:
            st.error(f"❌ Could not import train_model: {exc}")
        st.stop()

    try:
        from smartie_mnl import cross_dataset_heatmap as cd
    except Exception as exc:
        st.error(f"❌ Could not import cross_dataset_heatmap: {exc}")
        st.stop()

    return fe, tm, cd


# ── File upload helpers ────────────────────────────────────────────────────────

def save_uploads(uploaded_files, dest_dir: Path) -> list[Path]:
    """Save a list of Streamlit UploadedFile objects into dest_dir."""
    saved = []
    for uf in uploaded_files:
        p = dest_dir / uf.name
        p.write_bytes(uf.read())
        uf.seek(0)          # reset so callers can re-read if needed
        saved.append(p)
    return saved


def save_single_upload(uploaded_file, dest_dir: Path) -> Path:
    """Save one Streamlit UploadedFile into dest_dir."""
    return save_uploads([uploaded_file], dest_dir)[0]


@contextmanager
def temp_workspace():
    """Context manager that creates a temp directory and cleans it up on exit."""
    d = tempfile.mkdtemp(prefix="smartie_")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextmanager
def output_workspace(user_dir: str | None):
    """Use a user-specified persistent directory, or fall back to a temp workspace."""
    if user_dir:
        p = Path(user_dir)
        p.mkdir(parents=True, exist_ok=True)
        yield p
    else:
        with temp_workspace() as p:
            yield p


def savefig(fig, path_stem: Path, dpi: int = 180):
    """Save a matplotlib figure as both PNG and PDF."""
    for fmt in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{fmt}", dpi=dpi, bbox_inches="tight")


# ── Stdout capture ─────────────────────────────────────────────────────────────

class _StreamCapture(io.StringIO):
    """StringIO that also forwards writes to a Streamlit container in real time."""

    def __init__(self, container):
        super().__init__()
        self._container = container
        self._text = ""

    def write(self, s: str):
        if s and s.strip():
            self._text += s + "\n"
            self._container.code(self._text, language="")
        super().write(s)

    def flush(self):
        pass


@contextmanager
def capture_output(container):
    """Redirect stdout/stderr to a Streamlit code block in real time."""
    cap = _StreamCapture(container)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = cap
    sys.stderr = cap
    try:
        yield cap
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


# ── Plot display helpers ───────────────────────────────────────────────────────

def show_plots_in_dir(outdir: Path, title: str = "Plots", glob: str = "**/*.png"):
    """Find all PNGs in outdir (recursively) and display them in the UI."""
    plots = sorted(outdir.glob(glob))
    if not plots:
        st.info("No plots were generated in this run.")
        return
    st.markdown(f"### {title}")
    # Display in two-column grid
    cols = st.columns(2)
    for i, p in enumerate(plots):
        with cols[i % 2]:
            st.image(str(p), caption=p.stem.replace("_", " ").title(), use_container_width=True)


def show_top_predictions(predictions_tsv: Path, n: int = 30):
    """Show the top-N gene predictions as a nicely formatted table."""
    import pandas as pd
    if not predictions_tsv.exists():
        return
    df = pd.read_csv(predictions_tsv, sep="\t")
    # Try to find the probability/score column
    score_col = next(
        (c for c in ["rf_probability", "final_probability", "cv_probability", "score"]
         if c in df.columns),
        None,
    )
    gene_col = next(
        (c for c in ["gene", "Name", "name"] if c in df.columns),
        df.columns[0],
    )
    if score_col:
        df = df.sort_values(score_col, ascending=False)
        show_df = df[[gene_col, score_col]].head(n).reset_index(drop=True)
        show_df.index += 1
        show_df.columns = ["Gene", "RF Probability"]
        show_df["RF Probability"] = show_df["RF Probability"].map("{:.4f}".format)
    else:
        show_df = df.head(n).reset_index(drop=True)
        show_df.index += 1
    st.dataframe(show_df, use_container_width=True, height=min(600, 35 * n + 38))


# ── Download helpers ──────────────────────────────────────────────────────────

def zip_directory(src: Path) -> bytes:
    """Zip an entire directory tree and return the bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in src.rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(src))
    return buf.getvalue()


def download_button(outdir: Path, label: str = "⬇️ Download all results (ZIP)"):
    """Render a Streamlit download button that zips and downloads outdir."""
    if not outdir.exists():
        return
    zip_bytes = zip_directory(outdir)
    st.download_button(
        label=label,
        data=zip_bytes,
        file_name=f"{outdir.name}_results.zip",
        mime="application/zip",
        use_container_width=True,
    )


# ── Parameter panel helpers ───────────────────────────────────────────────────

def editing_type_selector(key: str = "editing_type") -> str:
    return st.selectbox(
        "Editing type",
        ["AtoG", "CtoT"],
        index=0,
        key=key,
        help=(
            "**AtoG** — TRIBE experiments using ADAR (most common).  \n"
            "**CtoT** — STAMP experiments using APOBEC."
        ),
    )


def min_reads_slider(key: str = "min_reads", default: int = 20) -> int:
    return st.slider(
        "Minimum reads per site",
        min_value=0,
        max_value=200,
        value=default,
        step=5,
        key=key,
        help=(
            "Drop editing sites covered by fewer than this many reads. "
            "**20** is a sensible default. Increase to 50–100 for noisier datasets."
        ),
    )


def min_edit_pct_slider(key: str = "min_edit_pct") -> float:
    return st.slider(
        "Minimum edit % per site",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        key=key,
        help=(
            "Drop sites where fewer than this percentage of reads show editing. "
            "Set to **5** to remove near-zero events that may be sequencing noise."
        ),
    )
