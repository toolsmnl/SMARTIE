"""
Home page — welcome screen and quick orientation.
"""

import streamlit as st


def show():
    st.title("🧬 smartie")
    st.subheader("RNA-binding protein target prediction from TRIBE & STAMP data")

    st.markdown(
        """
        **TRIBE** (Targets of RNA-Binding Proteins Identified By Editing) and **STAMP**
        (Surveying Targets by APOBEC-Mediated Profiling) fuse an RNA-binding protein (RBP)
        to a base-editing enzyme. Sites that are edited above background reveal where the
        RBP binds. `SMARTIE` automates the full analysis using a machine-learning model
        trained on 18 biologically motivated features.
        """
    )

    st.markdown("---")

    # ── Three workflow cards ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="background:#f0f7f0;border-radius:12px;padding:24px;height:260px;">
            <h3>🔬 Predict Targets</h3>
            <p>
            Upload your experiment and control editing files, set a few parameters,
            and get a ranked list of predicted RBP targets — with plots — in under a minute.
            </p>
            <p><b>Best for:</b> running a pre-trained model on your new data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background:#f0f4f7;border-radius:12px;padding:24px;height:260px;">
            <h3>🎓 Train Your Model</h3>
            <p>
            Have a list of known targets for your RBP? Train a custom model on your own data.
            The trained model is saved and can be reused on new experiments.
            </p>
            <p><b>Best for:</b> building a model tailored to your specific RBP and cell line.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div style="background:#f7f4f0;border-radius:12px;padding:24px;height:260px;">
            <h3>📊 Cross-Dataset Testing</h3>
            <p>
            Test whether a model trained on one RBP generalises to another.
            Generates precision heatmaps, ROC/PR curves, and rank-overlap plots.
            </p>
            <p><b>Best for:</b> comparing multiple models across multiple datasets.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("## How it works")

    steps = [
        ("1️⃣  Upload editing files",
         "Tab-separated files from your TRIBE or STAMP pipeline — one file per replicate. "
         "Both experiment (RBP-ADAR fusion) and control (ADAR only) are required."),
        ("2️⃣  Set parameters",
         "Choose your editing type (A-to-G or C-to-T), read-depth filters, and model options "
         "using sliders and drop-downs — no command line needed."),
        ("3️⃣  Run the analysis",
         "Click **Run**. Progress appears in real time. The pipeline computes 18 features per gene, "
         "trains (or applies) a Random Forest classifier, and writes all outputs."),
        ("4️⃣  Explore results",
         "Browse the ranked gene list and plots directly in the browser, then download "
         "everything as a ZIP file for your records."),
    ]

    for title, desc in steps:
        with st.expander(title, expanded=True):
            st.write(desc)

    st.markdown("---")
    st.markdown("## Input file format")
    st.markdown(
        "Your editing files should be **tab-separated** with these columns "
        "(column order does not matter):"
    )

    import pandas as pd
    example = pd.DataFrame({
        "Chr":            ["chr3R", "chr3R", "chrX"],
        "Edit_coord":     [16204186, 16204250, 9812344],
        "Name":           ["Atx2", "Atx2", "lola"],
        "Type":           ["EXON", "UTR3", "EXON"],
        "Editbase_count": [45, 12, 88],
        "Total_count":    [720, 310, 1040],
    })
    st.dataframe(example, use_container_width=True, hide_index=True)

    st.info(
        "💡 **For A-to-G (TRIBE/ADAR):** the edited-read column should be named "
        "`Editbase_count` or `G_count`.  \n"
        "**For C-to-T (STAMP/APOBEC):** name it `T_count`."
    )

    st.markdown("---")
    st.markdown(
        "Questions? See the full documentation in the "
        "[README on GitHub](https://github.com/toolsmnl/SMARTIE#readme)."
    )
