"""
smartie — Streamlit web interface
=====================================
Launch with:
    SMARTIE
or directly:
    streamlit run /path/to/smartie/app/app.py
"""

import os
import sys
from pathlib import Path

# Ensure the app directory is on the path so pages/ can be imported
_APP_DIR = Path(__file__).parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

st.set_page_config(
    page_title="smartie",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("## 🧬 smartie")
    st.markdown(
        "Machine-learning pipeline for predicting RNA-binding protein targets "
        "from **TRIBE** and **STAMP** editing data."
    )
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["🏠 Home", "🔬 Predict Targets", "🎓 Train Your Model", "📊 Cross-Dataset Testing"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Output settings**")

    # ── Base directory row: path text + browse button ────────────────────────
    col_path, col_browse = st.columns([5, 1])
    with col_path:
        base_dir = st.text_input(
            "Base directory",
            value=st.session_state.get("output_base_dir_input", os.getcwd()),
            label_visibility="collapsed",
            key="output_base_dir_input",
            help="Folder where results are saved. Defaults to the terminal's working directory.",
        )
    with col_browse:
        if st.button("📂", help="Browse for a folder", use_container_width=True):
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                try:
                    root.wm_attributes("-topmost", True)
                except Exception:
                    pass
                chosen = filedialog.askdirectory(
                    initialdir=base_dir or os.getcwd(),
                    title="Select output base folder",
                )
                root.destroy()
                if chosen:
                    st.session_state["output_base_dir_input"] = chosen
                    st.rerun()
            except Exception as exc:
                st.warning(f"Folder picker unavailable: {exc}")

    # ── Output folder name ───────────────────────────────────────────────────
    folder_name = st.text_input(
        "Output folder name",
        value=st.session_state.get("output_folder_name_input", "smartie_output"),
        key="output_folder_name_input",
        placeholder="smartie_output",
        help="A subfolder with this name will be created inside the base directory.",
    )

    # ── Resolve and display the final path ───────────────────────────────────
    _base = (base_dir or os.getcwd()).strip()
    _name = (folder_name or "smartie_output").strip()
    _final = str(Path(_base) / _name)
    st.session_state["output_dir"] = _final
    st.caption(f"`{_final}`")

    st.markdown("---")
    st.markdown(
        "**Need help?** See the "
        "[README on GitHub](https://github.com/toolsmnl/SMARTIE#readme)."
    )

if page == "🏠 Home":
    import pages.home as p;     p.show()
elif page == "🔬 Predict Targets":
    import pages.predict as p;  p.show()
elif page == "🎓 Train Your Model":
    import pages.train as p;    p.show()
elif page == "📊 Cross-Dataset Testing":
    import pages.crosstest as p; p.show()
