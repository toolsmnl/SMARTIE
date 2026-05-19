"""
smartie — Streamlit web interface
=====================================
Launch with:
    SMARTIE
or directly:
    streamlit run /path/to/smartie/app/app.py
"""

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
    st.markdown(
        "**Need help?** See the "
        "[README on GitHub](https://github.com/your-lab/smartie#readme)."
    )

if page == "🏠 Home":
    import pages.home as p;     p.show()
elif page == "🔬 Predict Targets":
    import pages.predict as p;  p.show()
elif page == "🎓 Train Your Model":
    import pages.train as p;    p.show()
elif page == "📊 Cross-Dataset Testing":
    import pages.crosstest as p; p.show()
