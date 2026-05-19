"""
Train page — train a custom Random Forest model on the user's own editing data.

Workflow:
  1. Upload experiment + control replicates
  2. Upload a known-targets file
  3. Configure training parameters
  4. Run training
  5. Show diagnostics, plots, download trained model + results
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))  # app dir
from utils import (
    capture_output,
    editing_type_selector,
    load_pipeline_modules,
    min_edit_pct_slider,
    min_reads_slider,
    save_uploads,
    save_single_upload,
    show_plots_in_dir,
    temp_workspace,
)


def show():
    st.title("🎓 Train Your Model")
    st.markdown(
        "Train a custom Random Forest classifier on your own TRIBE or STAMP data. "
        "You need a list of genes your RBP is known to bind — these act as the positive "
        "training examples. The trained model can then be applied to new experiments."
    )

    fe, tm, cd = load_pipeline_modules()

    # ── Step 1: Upload editing files ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 1 — Upload editing files")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Experiment replicates** *(RBP-ADAR or RBP-APOBEC fusion)*")
        expt_files = st.file_uploader(
            "Experiment files",
            accept_multiple_files=True,
            type=["txt", "tsv", "csv"],
            key="train_expt",
            label_visibility="collapsed",
            help="One file per replicate. Two or more replicates are recommended.",
        )
        if expt_files:
            st.success(f"✅ {len(expt_files)} experiment file(s)")

    with col2:
        st.markdown("**Control replicates** *(ADAR-only or APOBEC-only)*")
        ctrl_files = st.file_uploader(
            "Control files",
            accept_multiple_files=True,
            type=["txt", "tsv", "csv"],
            key="train_ctrl",
            label_visibility="collapsed",
        )
        if ctrl_files:
            st.success(f"✅ {len(ctrl_files)} control file(s)")

    # ── Step 2: Known targets ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 2 — Upload known targets")
    st.markdown(
        "Upload a file listing genes your RBP is **known to bind**. "
        "These are used as positive training examples. Sources: published CLIP-seq, "
        "PAR-CLIP, eCLIP, or biochemical pulldown data."
    )

    targets_file = st.file_uploader(
        "Known targets file",
        type=["txt", "csv"],
        key="train_targets",
        label_visibility="collapsed",
        help=(
            "**Text file:** one gene symbol per line.  \n"
            "**CSV:** must have a column named 'Name' containing gene symbols."
        ),
    )
    if targets_file:
        # Preview
        try:
            content = targets_file.read().decode()
            targets_file.seek(0)
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            st.success(f"✅ {len(lines)} genes in targets file")
            with st.expander(f"Preview (first 10 genes)"):
                st.write(", ".join(lines[:10]) + ("..." if len(lines) > 10 else ""))
        except Exception:
            st.warning("Could not preview targets file, but it will still be used.")

    # ── Step 3: Parameters ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 3 — Training parameters")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        editing_type = editing_type_selector("train_editing_type")
    with col_b:
        min_reads = min_reads_slider("train_min_reads", default=20)
    with col_c:
        min_edit_pct = min_edit_pct_slider("train_min_edit_pct")

    col_d, col_e = st.columns(2)
    with col_d:
        min_total_reads = st.slider(
            "Minimum total reads per gene",
            min_value=0, max_value=1000, value=50, step=10,
            key="train_min_total_reads",
            help=(
                "Gene-level filter: exclude very lowly expressed genes. "
                "50 is a safe default; raise to 100–200 for cleaner results."
            ),
        )
    with col_e:
        min_fold_change = st.slider(
            "Minimum fold-change over control",
            min_value=1.0, max_value=10.0, value=1.0, step=0.5,
            key="train_min_fc",
            help=(
                "Only include genes with at least this fold-change (experiment / control). "
                "1.5–2.0 helps focus the model on genuinely enriched genes."
            ),
        )

    with st.expander("⚙️ Advanced options"):
        normalization = st.selectbox(
            "Normalisation method",
            ["simple", "median_of_ratios"],
            index=0,
            key="train_norm",
            help=(
                "**simple:** divide by sample total reads.  \n"
                "**median_of_ratios:** DESeq2-style normalisation — more robust when "
                "replicates have different sequencing depths."
            ),
        )

    # ── Step 4: Run ─────────────────────────────────────────────────────────
    st.markdown("---")
    ready = bool(expt_files and ctrl_files and targets_file)
    if not ready:
        missing = []
        if not expt_files: missing.append("experiment files")
        if not ctrl_files: missing.append("control files")
        if not targets_file: missing.append("known targets file")
        st.info(f"📂 Still needed: {', '.join(missing)}.")

    if st.button(
        "▶️ Train Model",
        disabled=not ready,
        use_container_width=True,
        type="primary",
    ):
        _run_training(
            fe=fe, tm=tm,
            expt_files=expt_files,
            ctrl_files=ctrl_files,
            targets_file=targets_file,
            editing_type=editing_type,
            min_reads=min_reads,
            min_edit_pct=min_edit_pct,
            min_total_reads=min_total_reads,
            min_fold_change=min_fold_change,
            normalization=normalization,
        )


def _run_training(
    fe, tm,
    expt_files, ctrl_files, targets_file,
    editing_type, min_reads, min_edit_pct,
    min_total_reads, min_fold_change,
    normalization,
):
    with temp_workspace() as workspace:
        data_dir = workspace / "data"
        out_dir  = workspace / "outputs"
        data_dir.mkdir(); out_dir.mkdir()

        expt_paths = save_uploads(expt_files, data_dir)
        ctrl_paths = save_uploads(ctrl_files, data_dir)
        tgt_path   = save_single_upload(targets_file, data_dir)

        st.markdown("---")
        st.markdown("### Training in progress…")
        st.caption(
            "This typically takes 20–90 seconds depending on dataset size. "
            "Do not close this tab."
        )
        log_container = st.empty()

        try:
            with capture_output(log_container):
                # Load data
                print("Loading editing data...")
                expt_dfs = []
                for p in expt_paths:
                    df = fe.load_editing_sites(
                        p, editing_type=editing_type,
                        min_edit_pct=min_edit_pct, min_reads=min_reads,
                    )
                    print(f"  [expt] {p.name}: {len(df):,} sites")
                    expt_dfs.append(df)

                ctrl_dfs = []
                for p in ctrl_paths:
                    df = fe.load_editing_sites(
                        p, editing_type=editing_type,
                        min_edit_pct=min_edit_pct, min_reads=min_reads,
                    )
                    print(f"  [ctrl] {p.name}: {len(df):,} sites")
                    ctrl_dfs.append(df)

                # Targets
                targets_set = tm.load_targets(str(tgt_path))
                print(f"\nKnown targets: {len(targets_set)}")

                # Gene universe
                gene_names = sorted(
                    set().union(*(df["Name"].unique() for df in expt_dfs + ctrl_dfs))
                )
                print(f"Unique genes:  {len(gene_names):,}")

                # Features
                print("\nBuilding features...")
                features_df = fe.build_features(
                    expt_dfs, ctrl_dfs, gene_names,
                    normalization=normalization,
                    min_total_reads=min_total_reads,
                    min_fold_change=min_fold_change,
                )
                features_df.to_csv(out_dir / "gene_features.tsv", sep="\t", index=False)

                # Resolve model selection (use RF only for simplicity from UI)
                selected_models = tm.resolve_model_selection(["rf"], None)

                # Train
                print("\nTraining model...")
                tm.train_all_models(
                    features_df=features_df,
                    targets_set=targets_set,
                    outdir=out_dir,
                    model_names=selected_models,
                    feature_weights=None,
                    label="smartie_model",
                    make_plots=True,
                        )
                print("\nTraining complete.")

        except Exception as exc:
            st.error(f"❌ Training error: {exc}")
            with st.expander("Full traceback"):
                import traceback
                st.code(traceback.format_exc())
            return

        # ── Results ───────────────────────────────────────────────────────────
        st.success("✅ Training complete!")
        st.markdown("---")
        st.markdown("### Results")

        # Top predictions
        pred_path = out_dir / "predictions.tsv"
        if pred_path.exists():
            st.markdown("#### Top 50 predicted targets")
            st.caption(
                "Genes are ranked by the model's probability that they are RBP targets. "
                "Known targets are highlighted with `is_target = 1`."
            )
            df_pred = pd.read_csv(pred_path, sep="\t")
            score_col = next(
                (c for c in ["final_probability", "cv_probability"] if c in df_pred.columns),
                df_pred.columns[-1],
            )
            df_pred = df_pred.sort_values(score_col, ascending=False).head(50).reset_index(drop=True)
            df_pred.index += 1
            st.dataframe(df_pred, use_container_width=True, height=500)

        # Precision at top-K
        metrics_path = out_dir / "topk_metrics.tsv"
        if metrics_path.exists():
            st.markdown("#### Precision at top-K")
            st.caption(
                "Of the top-K genes predicted by the model, what fraction are known targets? "
                "Higher is better."
            )
            df_metrics = pd.read_csv(metrics_path, sep="\t")
            st.dataframe(df_metrics, hide_index=True, use_container_width=True)

        # Feature importance
        imp_path = out_dir / "feature_importance.tsv"
        if imp_path.exists():
            st.markdown("#### Feature importance")
            df_imp = pd.read_csv(imp_path, sep="\t").head(18)
            st.dataframe(df_imp, hide_index=True, use_container_width=True)

        # Plots
        show_plots_in_dir(out_dir, title="Training plots", glob="**/*.png")

        # Download trained model separately for convenience
        model_pkl = out_dir / "rf_model.pkl"
        if model_pkl.exists():
            st.markdown("---")
            st.markdown("### Download your trained model")
            st.markdown(
                "Save this `rf_model.pkl` file — you will upload it in the "
                "**Predict** page to apply it to new experiments."
            )
            st.download_button(
                "⬇️ Download rf_model.pkl",
                data=model_pkl.read_bytes(),
                file_name="rf_model.pkl",
                mime="application/octet-stream",
                use_container_width=True,
            )

        # Download all
        st.markdown("---")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in out_dir.rglob("*"):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(out_dir))
        st.download_button(
            "⬇️ Download all training results (ZIP)",
            data=zip_buf.getvalue(),
            file_name="smartie_training.zip",
            mime="application/zip",
            use_container_width=True,
        )
