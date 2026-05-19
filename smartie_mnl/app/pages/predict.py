"""
Predict page — apply a trained RF model to new editing data.
"""

from __future__ import annotations

import io
import pickle
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    capture_output,
    editing_type_selector,
    load_pipeline_modules,
    min_edit_pct_slider,
    min_reads_slider,
    save_uploads,
    save_single_upload,
    show_top_predictions,
    temp_workspace,
)


def show():
    st.title("🔬 Predict Targets")
    st.markdown(
        "Apply a trained SMARTIE model to your editing data and get a ranked list "
        "of predicted RBP target genes."
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
            key="pred_expt",
            label_visibility="collapsed",
        )
        if expt_files:
            st.success(f"✅ {len(expt_files)} experiment file(s) uploaded")

    with col2:
        st.markdown("**Control replicates** *(ADAR-only or APOBEC-only)*")
        ctrl_files = st.file_uploader(
            "Control files",
            accept_multiple_files=True,
            type=["txt", "tsv", "csv"],
            key="pred_ctrl",
            label_visibility="collapsed",
        )
        if ctrl_files:
            st.success(f"✅ {len(ctrl_files)} control file(s) uploaded")

    # ── Step 2: Upload model ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 2 — Upload your trained model")
    model_file = st.file_uploader(
        "rf_model.pkl",
        type=["pkl"],
        key="pred_model",
        label_visibility="collapsed",
    )
    if model_file:
        st.success("✅ Model uploaded")

    # Optional validation targets
    st.markdown("***(Optional)*** Upload a known-targets file to evaluate predictions with EPAR")
    targets_file = st.file_uploader(
        "Known targets (optional)",
        type=["txt", "csv"],
        key="pred_targets",
        label_visibility="collapsed",
        help=(
            "A text file (one gene per line) or CSV with a 'Name' column. "
            "If provided, an EPAR heatmap and Venn diagram will be generated."
        ),
    )

    # ── Step 3: Parameters ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 3 — Parameters")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        editing_type = editing_type_selector("pred_editing_type")
    with col_b:
        min_reads = min_reads_slider("pred_min_reads", default=20)
    with col_c:
        min_edit_pct = min_edit_pct_slider("pred_min_edit_pct")

    top_n = st.slider(
        "Genes to display in ranked table",
        min_value=10, max_value=500, value=50, step=10,
        key="pred_top_n",
    )


    # ── Step 4: Run ─────────────────────────────────────────────────────────
    st.markdown("---")
    ready = bool(expt_files and ctrl_files and model_file)
    if not ready:
        st.info("📂 Upload experiment files, control files, and a model file to enable Run.")

    if st.button("▶️ Run Prediction", disabled=not ready,
                 use_container_width=True, type="primary"):
        _run_prediction(
            fe=fe, tm=tm,
            expt_files=expt_files,
            ctrl_files=ctrl_files,
            model_file=model_file,
            targets_file=targets_file,
            editing_type=editing_type,
            min_reads=min_reads,
            min_edit_pct=min_edit_pct,
            top_n=top_n,
        )


def _run_prediction(
    fe, tm,
    expt_files, ctrl_files, model_file, targets_file,
    editing_type, min_reads, min_edit_pct, top_n,
):
    with temp_workspace() as workspace:
        data_dir = workspace / "data"
        out_dir  = workspace / "outputs"
        data_dir.mkdir(); out_dir.mkdir()

        expt_paths = save_uploads(expt_files, data_dir)
        ctrl_paths = save_uploads(ctrl_files, data_dir)
        rf_model   = pickle.loads(model_file.read())

        targets_set: set[str] = set()
        if targets_file:
            tgt_path = save_single_upload(targets_file, data_dir)
            targets_set = tm.load_targets(str(tgt_path))
            st.info(f"📋 Loaded {len(targets_set)} known targets for EPAR evaluation.")

        st.markdown("---")
        st.markdown("### Running…")
        log_container = st.empty()

        try:
            with capture_output(log_container):
                print("Loading editing data...")
                expt_dfs, ctrl_dfs = [], []
                for p in expt_paths:
                    df = fe.load_editing_sites(p, editing_type=editing_type,
                                               min_edit_pct=min_edit_pct, min_reads=min_reads)
                    print(f"  [expt] {p.name}: {len(df):,} sites")
                    expt_dfs.append(df)
                for p in ctrl_paths:
                    df = fe.load_editing_sites(p, editing_type=editing_type,
                                               min_edit_pct=min_edit_pct, min_reads=min_reads)
                    print(f"  [ctrl] {p.name}: {len(df):,} sites")
                    ctrl_dfs.append(df)

                gene_names = sorted(
                    set().union(*(df["Name"].unique() for df in expt_dfs + ctrl_dfs))
                )
                print(f"\nFound {len(gene_names):,} unique genes")

                print("\nBuilding features...")
                features_df = fe.build_features(expt_dfs, ctrl_dfs, gene_names)
                features_df.to_csv(out_dir / "gene_features.tsv", sep="\t", index=False)

                print("\nApplying model...")
                feature_cols = [c for c in features_df.columns if c != "Name"]
                import numpy as np
                X = features_df[feature_cols].fillna(0).values
                probabilities = rf_model.predict_proba(X)[:, 1]

                results_df = pd.DataFrame({
                    "gene": features_df["Name"].values,
                    "rf_probability": probabilities,
                }).sort_values("rf_probability", ascending=False).reset_index(drop=True)
                results_df["rank"] = range(1, len(results_df) + 1)

                if targets_set:
                    results_df["is_known_target"] = (
                        results_df["gene"].str.lower().isin(targets_set)
                    ).astype(int)

                results_df.to_csv(out_dir / "predictions.tsv", sep="\t", index=False)

                print("\nGenerating plots...")
                _make_plots(results_df, targets_set, out_dir)
                print("\nDone.")

        except Exception as exc:
            st.error(f"❌ Pipeline error: {exc}")
            with st.expander("Full traceback"):
                import traceback
                st.code(traceback.format_exc())
            return

        # ── Results ──────────────────────────────────────────────────────────
        st.success("✅ Prediction complete!")
        st.markdown("---")
        st.markdown("### Results")

        st.markdown(f"#### Top {top_n} predicted targets")
        show_top_predictions(out_dir / "predictions.tsv", n=top_n)

        # Show plots
        plots_dir = out_dir / "plots"
        pngs = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []
        if pngs:
            st.markdown("#### Plots")
            if len(pngs) == 1:
                st.image(str(pngs[0]), caption=pngs[0].stem.replace("_", " ").title(),
                         use_container_width=True)
            else:
                cols = st.columns(len(pngs))
                for col, p in zip(cols, pngs):
                    with col:
                        st.image(str(p), caption=p.stem.replace("_", " ").title(),
                                 use_container_width=True)

        # Download
        st.markdown("---")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in out_dir.rglob("*"):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(out_dir))
        st.download_button(
            "⬇️ Download all results (ZIP)",
            data=zip_buf.getvalue(),
            file_name="smartie_predictions.zip",
            mime="application/zip",
            use_container_width=True,
        )


# ── Plot generation ───────────────────────────────────────────────────────────

def _make_plots(results_df: pd.DataFrame, targets_set: set, out_dir: Path):
    """
    With validation targets:   EPAR heatmap + Venn diagram
    Without validation targets: Score distribution + Editing enrichment scatter
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    has_targets = bool(targets_set) and "is_known_target" in results_df.columns

    if has_targets:
        _plot_epar_heatmap(results_df, targets_set, plots_dir)
        _plot_venn(results_df, targets_set, plots_dir)
    else:
        _plot_score_distribution(results_df, plots_dir)
        _plot_enrichment_scatter(results_df, out_dir, plots_dir)


def _plot_epar_heatmap(results_df: pd.DataFrame, targets_set: set, plots_dir: Path):
    """
    EPAR heatmap with cividis_r colormap.
    n= annotations placed inside each cell below the EPAR value to avoid collision.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    deciles   = list(range(10, 110, 10))
    n_targets = len(targets_set)
    epar_vals = []
    n_refs    = []

    for d in deciles:
        n_ref = max(1, min(round(n_targets * d / 100), len(results_df)))
        top_genes = set(results_df.head(n_ref)["gene"].str.lower())
        epar_vals.append(len(top_genes & targets_set) / n_ref * 100)
        n_refs.append(n_ref)

    df_heat = pd.DataFrame(
        {"EPAR (%)": epar_vals},
        index=[f"{d}%" for d in deciles],
    )

    # Wider figure to give colorbar room and avoid clipping
    fig, ax = plt.subplots(figsize=(4.5, 7))

    sns.heatmap(
        df_heat,
        ax=ax,
        annot=False,          # we draw custom annotations below
        cmap="cividis_r",
        vmin=0,
        vmax=100,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "EPAR (%)", "shrink": 0.6, "pad": 0.02},
    )

    # Custom two-line annotation: EPAR value + n= on separate lines inside each cell
    for i, (val, n_ref) in enumerate(zip(epar_vals, n_refs)):
        # Determine text colour for readability against the colormap
        text_color = "white" if val < 50 else "black"
        ax.text(
            0.5, i + 0.38, f"{val:.1f}",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=text_color,
            transform=ax.transData,
        )
        ax.text(
            0.5, i + 0.68, f"(n={n_ref})",
            ha="center", va="center",
            fontsize=8, color=text_color, alpha=0.85,
            transform=ax.transData,
        )

    ax.set_title(
        "Equivalent Precision\nAgainst Reference (EPAR)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("EPAR (%)", fontsize=11)
    ax.set_ylabel("Decile of reference target list", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10, rotation=0)

    fig.tight_layout()
    fig.savefig(plots_dir / "epar_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    df_heat.to_csv(plots_dir / "epar_values.tsv", sep="\t")



def _plot_venn(results_df: pd.DataFrame, targets_set: set, plots_dir: Path):
    """
    Venn diagram using the full reference target list size as N.
    Top-N SMARTIE predictions (where N = len(targets_set)) vs all known targets.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(targets_set)                                      # use full reference list size
    top_genes    = set(results_df.head(n)["gene"].str.lower())
    overlap      = top_genes & targets_set
    only_smartie = top_genes - targets_set
    only_known   = targets_set - top_genes

    try:
        from matplotlib_venn import venn2
        fig, ax = plt.subplots(figsize=(6, 5))
        v = venn2(
            subsets=(len(only_smartie), len(only_known), len(overlap)),
            set_labels=(f"SMARTIE\ntop {n}", "Known\ntargets"),
            ax=ax,
            set_colors=("#4CAF50", "#2196F3"),
            alpha=0.6,
        )
        if v.get_label_by_id("11"):
            v.get_label_by_id("11").set_fontsize(13)
            v.get_label_by_id("11").set_fontweight("bold")
    except ImportError:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
        ax.add_patch(plt.Circle((3.5, 3.5), 2.8, color="#4CAF50", alpha=0.4))
        ax.add_patch(plt.Circle((6.5, 3.5), 2.8, color="#2196F3", alpha=0.4))
        ax.text(2.2, 3.5, str(len(only_smartie)), ha="center", va="center",
                fontsize=16, fontweight="bold", color="#2E7D32")
        ax.text(5.0, 3.5, str(len(overlap)), ha="center", va="center",
                fontsize=18, fontweight="bold", color="#333333")
        ax.text(7.8, 3.5, str(len(only_known)), ha="center", va="center",
                fontsize=16, fontweight="bold", color="#1565C0")
        ax.text(2.2, 0.5, f"SMARTIE only\n(top {n})", ha="center",
                fontsize=10, color="#2E7D32")
        ax.text(5.0, 0.2, f"Overlap\n({len(overlap)/max(1,n)*100:.0f}%)",
                ha="center", fontsize=10, color="#333")
        ax.text(7.8, 0.5, "Known\ntargets only", ha="center",
                fontsize=10, color="#1565C0")

    ax.set_title(
        f"SMARTIE top {n} predictions vs known targets\n"
        f"(overlap: {len(overlap)}/{n} = {len(overlap)/max(1,n)*100:.1f}%)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "venn_diagram.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_score_distribution(results_df: pd.DataFrame, plots_dir: Path):
    """
    RF probability score distribution.
    Colour-coded: high confidence (>0.7) green, mid (0.4–0.7) orange, low (<0.4) grey.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    scores = results_df["rf_probability"].values
    bins   = np.linspace(0, 1, 41)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot three segments with different colours
    for lo, hi, color, label in [
        (0.0, 0.4, "#B0BEC5", "Low confidence  (<0.4)"),
        (0.4, 0.7, "#FFB74D", "Mid confidence  (0.4–0.7)"),
        (0.7, 1.0, "#4CAF50", "High confidence (>0.7)"),
    ]:
        mask = (scores >= lo) & (scores < hi + 0.001)
        ax.hist(scores[mask], bins=bins, color=color, edgecolor="white",
                alpha=0.9, label=label, linewidth=0.5)

    ax.axvline(0.5, color="#E53935", linestyle="--", linewidth=1.5,
               alpha=0.8, label="0.5 threshold")
    ax.set_xlabel("RF Probability Score", fontsize=12)
    ax.set_ylabel("Number of genes", fontsize=12)
    ax.set_title("SMARTIE Score Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.8)

    n_high = (scores >= 0.7).sum()
    ax.annotate(
        f"{n_high} genes\nwith score > 0.7",
        xy=(0.85, n_high), xytext=(0.75, n_high * 1.3 + 5),
        arrowprops=dict(arrowstyle="->", color="grey"),
        fontsize=9, color="#2E7D32",
    )

    fig.tight_layout()
    fig.savefig(plots_dir / "score_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_enrichment_scatter(results_df: pd.DataFrame, out_dir: Path, plots_dir: Path):
    """
    Scatter: average experiment edit rate vs average control edit rate per gene,
    coloured by RF probability score. Genes with high scores that sit above the
    diagonal are true enriched targets.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    feat_path = out_dir / "gene_features.tsv"
    if not feat_path.exists():
        return

    import pandas as pd
    feats = pd.read_csv(feat_path, sep="\t")

    # Average experiment and control rates across replicates
    e_cols = [c for c in feats.columns if "e1__gene_rate" in c or "e2__gene_rate" in c]
    b_cols = [c for c in feats.columns if "bg1__gene_rate" in c or "bg2__gene_rate" in c]

    if not e_cols or not b_cols:
        return

    feats["avg_expt_rate"] = feats[e_cols].mean(axis=1)
    feats["avg_ctrl_rate"] = feats[b_cols].mean(axis=1)

    # Merge with RF scores
    merged = feats.merge(
        results_df[["gene", "rf_probability"]].rename(columns={"gene": "Name"}),
        on="Name", how="inner",
    )
    merged = merged[
        (merged["avg_expt_rate"] > 0) | (merged["avg_ctrl_rate"] > 0)
    ].copy()

    # Log-transform for readability
    merged["log_expt"] = np.log10(merged["avg_expt_rate"].clip(lower=1e-6))
    merged["log_ctrl"] = np.log10(merged["avg_ctrl_rate"].clip(lower=1e-6))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(
        merged["log_ctrl"], merged["log_expt"],
        c=merged["rf_probability"],
        cmap="RdYlGn", vmin=0, vmax=1,
        alpha=0.6, s=18, linewidths=0,
    )

    # Diagonal (equal rates)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, label="Equal rates")
    ax.set_xlim(lims); ax.set_ylim(lims)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("SMARTIE RF score", fontsize=10)

    ax.set_xlabel("log₁₀ Control edit rate", fontsize=11)
    ax.set_ylabel("log₁₀ Experiment edit rate", fontsize=11)
    ax.set_title("Editing enrichment — experiment vs control\n(coloured by SMARTIE score)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(plots_dir / "enrichment_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
