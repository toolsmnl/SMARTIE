"""
Tutorial page — step-by-step guide with downloadable example data from GitHub.
"""

import streamlit as st

# GitHub repository base URL
_REPO = "https://github.com/toolsmnl/SMARTIE"
_RAW  = "https://raw.githubusercontent.com/toolsmnl/SMARTIE/main"
_TUTORIAL_DIR = f"{_REPO}/tree/main/Tutorial"
_ARCHIVE_URL  = f"{_REPO}/blob/main/Tutorial/Example_Data.7z?raw=true"
_YOUTUBE      = "https://www.youtube.com/channel/UCWI3TLG4mFfU-enSNWL_yGw"
_PREPRINT     = "https://doi.org/10.64898/2026.05.18.726004"


def show():
    st.title("📖 Tutorial")
    st.markdown(
        "This tutorial walks through a complete SMARTIE run using the **Ataxin-2** example dataset "
        "([Singh et al., 2021](https://elifesciences.org/articles/60326)). "
        "Video walkthroughs are also available on the "
        f"[MNLTools YouTube channel]({_YOUTUBE})."
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_app, tab_cli, tab_data = st.tabs([
        "🖥️  GUI walkthrough",
        "💻  Command-line walkthrough",
        "📦  Download example data",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — GUI walkthrough
    # ══════════════════════════════════════════════════════════════════════════
    with tab_app:
        st.markdown("## Running SMARTIE through the web interface")
        st.markdown(
            "Follow the steps below. All files needed are in the "
            f"[Tutorial folder on GitHub]({_TUTORIAL_DIR}) — "
            "see the **Download example data** tab to get them."
        )

        st.markdown("---")

        with st.expander("### Step 1 — Install SMARTIE", expanded=True):
            st.markdown(
                "Open a terminal (Terminal on macOS/Linux; PowerShell or Command Prompt on Windows) "
                "and run:"
            )
            st.code("pip install git+https://github.com/toolsmnl/SMARTIE.git", language="bash")
            st.markdown(
                "For users new to Python, **pipx** is recommended — it keeps SMARTIE in its own "
                "isolated environment:"
            )
            st.code(
                "# macOS\nbrew install pipx\npipx install git+https://github.com/toolsmnl/SMARTIE.git",
                language="bash",
            )
            st.code(
                "# Ubuntu / Debian\nsudo apt install pipx\npipx install git+https://github.com/toolsmnl/SMARTIE.git",
                language="bash",
            )
            st.info(
                "**Windows note:** if `SMARTIE` is not recognised after installing, run:\n"
                "```\npython -m smartie_mnl.app.launcher\n```"
            )

        with st.expander("### Step 2 — Launch the application", expanded=True):
            st.code("SMARTIE", language="bash")
            st.markdown(
                "A browser tab opens automatically at **http://localhost:8501**. "
                "If the browser does not open, paste that URL into your browser manually."
            )

        with st.expander("### Step 3 — Run a prediction", expanded=True):
            st.markdown(
                "1. Click **🔬 Predict Targets** in the sidebar.\n"
                "2. Under **Experiment replicates**, upload `RBP_Exp_1.txt` and `RBP_Exp_2.txt`.\n"
                "3. Under **Control replicates**, upload `Ctrl_1.txt` and `Ctrl_2.txt`.\n"
                "4. Under **Targets file** *(optional)*, upload `Targets.csv` to enable the "
                "EPAR heatmap and Venn diagram.\n"
                "5. Leave all parameters at their defaults.\n"
                "6. Click **▶️ Run Prediction** and wait for the progress log to complete."
            )
            st.success(
                "✅ When finished, the ranked predictions table, EPAR heatmap, and Venn diagram "
                "are shown below the log. Click **⬇️ Download all results (ZIP)** to save everything."
            )

        with st.expander("### Expected outputs", expanded=False):
            st.markdown(
                "After a successful run you should see:"
            )
            st.code(
                "smartie_predictions.zip\n"
                "├── predictions.tsv          # ranked gene list\n"
                "├── gene_features.tsv        # features per gene\n"
                "└── plots/\n"
                "    ├── epar_heatmap.png/.pdf\n"
                "    ├── epar_values.tsv\n"
                "    └── venn_diagram.png/.pdf",
                language="text",
            )
            st.markdown(
                "A reference copy of the expected outputs is available as "
                f"[`smartie_predictions.zip`]({_REPO}/blob/main/Tutorial/smartie_predictions.zip?raw=true) "
                "in the Tutorial folder. If your plots match, the installation is working correctly."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — CLI walkthrough
    # ══════════════════════════════════════════════════════════════════════════
    with tab_cli:
        st.markdown("## Running SMARTIE from the command line")
        st.markdown(
            "The command-line interface provides the same functionality as the GUI "
            "and is suited for scripting and batch analysis."
        )

        st.markdown("---")

        with st.expander("### Step 1 — Install", expanded=True):
            st.code("pip install git+https://github.com/toolsmnl/SMARTIE.git", language="bash")
            st.markdown("Verify the installation:")
            st.code("smartie-test --help", language="bash")

        with st.expander("### Step 2 — Prepare a metadata TSV", expanded=True):
            st.markdown(
                "Create a tab-separated file (e.g. `metadata.tsv`) describing your dataset. "
                "Replicate files within a column are separated by semicolons."
            )
            st.code(
                "label\texpt_files\tctrl_files\ttargets_file\n"
                "Example_RBP\tExample_Data/RBP_Exp_1.txt;Example_Data/RBP_Exp_2.txt"
                "\tExample_Data/Ctrl_1.txt;Example_Data/Ctrl_2.txt"
                "\tExample_Data/Targets.csv",
                language="text",
            )

        with st.expander("### Step 3 — Run the prediction", expanded=True):
            st.code(
                "smartie-test \\\n"
                "    --metadata Example_Data/metadata.tsv \\\n"
                "    --outdir   results/example/",
                language="bash",
            )
            st.markdown("**Key options:**")
            import pandas as pd
            opts = pd.DataFrame([
                ["--min-reads N", "20", "Drop sites with fewer than N total reads"],
                ["--min-edit-pct F", "0.0", "Drop sites below F% editing"],
                ["--editing-type", "AtoG", "AtoG for TRIBE/ADAR; CtoT for STAMP/APOBEC"],
                ["--outdir DIR", "outputs/cross_test", "Where results are written"],
            ], columns=["Flag", "Default", "Description"])
            st.dataframe(opts, hide_index=True, use_container_width=True)

        with st.expander("### Expected output structure", expanded=False):
            st.code(
                "results/example/\n"
                "└── Example_RBP/\n"
                "    ├── predictions.tsv\n"
                "    ├── gene_features.tsv\n"
                "    └── plots/\n"
                "        ├── epar_heatmap.png\n"
                "        ├── epar_heatmap.pdf\n"
                "        ├── epar_values.tsv\n"
                "        ├── venn_diagram.png\n"
                "        └── venn_diagram.pdf",
                language="text",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Download example data
    # ══════════════════════════════════════════════════════════════════════════
    with tab_data:
        st.markdown("## Example dataset")
        st.markdown(
            "The example dataset is based on the **Ataxin-2** TRIBE experiment "
            "([Singh et al., 2021](https://elifesciences.org/articles/60326)). "
            "It contains all files needed to complete both the GUI and CLI tutorials."
        )

        st.markdown("---")

        # File table
        import pandas as pd
        files_df = pd.DataFrame([
            ["RBP_Exp_1.txt", "Experiment replicate 1"],
            ["RBP_Exp_2.txt", "Experiment replicate 2"],
            ["Ctrl_1.txt",    "Control replicate 1"],
            ["Ctrl_2.txt",    "Control replicate 2"],
            ["Targets.csv",   "Known Ataxin-2 target genes (for EPAR / Venn evaluation)"],
        ], columns=["File", "Role"])
        st.dataframe(files_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("### Download")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📦 All files (archive)")
            st.markdown(
                "Download `Example_Data.7z` — contains all 5 files above in a single archive. "
                "Extract with [7-Zip](https://www.7-zip.org/) (Windows/Linux) "
                "or [The Unarchiver](https://theunarchiver.com/) (macOS)."
            )
            st.link_button(
                "⬇️ Download Example_Data.7z",
                url=_ARCHIVE_URL,
                use_container_width=True,
            )

        with col2:
            st.markdown("#### 📁 Individual files")
            st.markdown("Click any file to download it directly from GitHub:")
            for fname, role in [
                ("RBP_Exp_1.txt", "Experiment rep 1"),
                ("RBP_Exp_2.txt", "Experiment rep 2"),
                ("Ctrl_1.txt",    "Control rep 1"),
                ("Ctrl_2.txt",    "Control rep 2"),
                ("Targets.csv",   "Known targets"),
            ]:
                url = f"{_REPO}/blob/main/Tutorial/{fname}?raw=true"
                st.link_button(
                    f"⬇️ {fname}  —  {role}",
                    url=url,
                    use_container_width=True,
                )

        st.markdown("---")
        st.markdown("### Reference outputs")
        st.markdown(
            "A reference copy of the expected results is also available. "
            "Compare your outputs against these to confirm the installation is working."
        )
        st.link_button(
            "⬇️ Download reference smartie_predictions.zip",
            url=f"{_REPO}/blob/main/Tutorial/smartie_predictions.zip?raw=true",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### Video tutorials")
        st.markdown(
            f"Step-by-step video walkthroughs are available on the "
            f"[MNLTools YouTube channel]({_YOUTUBE})."
        )
        st.link_button(
            "▶️ Watch on YouTube",
            url=_YOUTUBE,
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### Citation")
        st.info(
            "If SMARTIE contributes to your work, please cite:\n\n"
            "> Koppaka O, Kumar U, Ahuja G, Yadav R, Bakthavachalu B (2026). "
            "*SMARTIE: A Machine-Learning approach for investigating RBP-RNA interactions identified by Editing.* "
            f"bioRxiv 2026.05.18.726004. {_PREPRINT}"
        )
