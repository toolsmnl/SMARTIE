"""
Cross-dataset testing page.

Allows the user to test a trained model against multiple datasets defined
in a metadata TSV, or to supply datasets manually via the UI without
needing a metadata file.

Two sub-modes:
  A) Manual upload — user uploads datasets one by one through the UI.
     Good for 1–3 datasets.
  B) Metadata file — user uploads a pre-filled TSV. Good for many datasets.
"""

from __future__ import annotations

import io
import pickle
import sys
import tempfile
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
    output_workspace,
    savefig,
    save_uploads,
    save_single_upload,
    show_plots_in_dir,
)


def show():
    st.title("📊 Cross-Dataset Testing")
    st.markdown(
        "Test a trained model on one or more datasets to see how well its predictions "
        "generalise. Generates precision heatmaps, ROC curves, PR curves, and more."
    )

    fe, tm, cd = load_pipeline_modules()

    # ── Mode selector ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### How would you like to specify the datasets?")
    mode = st.radio(
        "Dataset input mode",
        ["📁 Upload datasets manually (1–3 datasets)", "📋 Upload a metadata TSV file"],
        label_visibility="collapsed",
        key="cdt_mode",
    )

    # ── Upload model ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Upload your trained model")
    model_file = st.file_uploader(
        "rf_model.pkl",
        type=["pkl"],
        key="cdt_model",
        label_visibility="collapsed",
        help="The trained model produced by the Train page or smartie-train.",
    )
    if model_file:
        st.success("✅ Model uploaded")

    # ── Dataset input ─────────────────────────────────────────────────────────
    datasets: list[dict] = []

    if mode.startswith("📁"):
        datasets = _manual_dataset_input()
    else:
        datasets = _metadata_dataset_input()

    # ── Parameters ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Parameters")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        editing_type = editing_type_selector("cdt_editing_type")
    with col_b:
        min_reads = min_reads_slider("cdt_min_reads", default=20)
    with col_c:
        min_edit_pct = min_edit_pct_slider("cdt_min_edit_pct")

    # ── Run ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    ready = bool(model_file and datasets and all(d.get("expt") and d.get("ctrl") for d in datasets))
    if not ready:
        if not model_file:
            st.info("📂 Upload a trained model file to continue.")
        elif not datasets:
            st.info("📂 Add at least one complete dataset (experiment + control files).")
        else:
            incomplete = [d["label"] for d in datasets if not d.get("expt") or not d.get("ctrl")]
            st.info(f"📂 These datasets are missing files: {', '.join(incomplete)}")

    if st.button(
        "▶️ Run Cross-Dataset Testing",
        disabled=not ready,
        use_container_width=True,
        type="primary",
    ):
        _run_cross_test(
            fe=fe, tm=tm, cd=cd,
            model_file=model_file,
            datasets=datasets,
            editing_type=editing_type,
            min_reads=min_reads,
            min_edit_pct=min_edit_pct,
            out_dir_setting=st.session_state.get("output_dir") or None,
        )


# ── Manual dataset input UI ──────────────────────────────────────────────────

def _manual_dataset_input() -> list[dict]:
    """Render UI for manually uploading up to 3 datasets. Returns list of dataset dicts."""

    st.markdown("---")
    st.markdown("### Upload datasets")
    st.markdown(
        "Fill in one dataset at a time. Each dataset needs a name, "
        "experiment files, and control files."
    )

    n_datasets = st.selectbox(
        "How many datasets?",
        [1, 2, 3],
        key="cdt_n_datasets",
    )

    datasets = []
    for i in range(n_datasets):
        with st.expander(f"Dataset {i+1}", expanded=True):
            label = st.text_input(
                "Dataset name",
                value=f"Dataset_{i+1}",
                key=f"cdt_label_{i}",
                help="A short label shown in the plots (e.g. 'Ataxin-2', 'Thor', 'FUS').",
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Experiment files**")
                expt = st.file_uploader(
                    f"Expt {i+1}",
                    accept_multiple_files=True,
                    type=["txt", "tsv", "csv"],
                    key=f"cdt_expt_{i}",
                    label_visibility="collapsed",
                )
                if expt: st.caption(f"✅ {len(expt)} file(s)")
            with col2:
                st.markdown("**Control files**")
                ctrl = st.file_uploader(
                    f"Ctrl {i+1}",
                    accept_multiple_files=True,
                    type=["txt", "tsv", "csv"],
                    key=f"cdt_ctrl_{i}",
                    label_visibility="collapsed",
                )
                if ctrl: st.caption(f"✅ {len(ctrl)} file(s)")

            targets = st.file_uploader(
                "Known targets (optional — needed for precision metrics)",
                type=["txt", "csv"],
                key=f"cdt_targets_{i}",
                help="Without this, no precision or ROC metrics will be shown for this dataset.",
            )

            datasets.append({
                "label": label,
                "expt": expt,
                "ctrl": ctrl,
                "targets": targets,
            })

    return datasets


def _metadata_dataset_input() -> list[dict]:
    """Render UI for metadata TSV upload. Returns list of dataset dicts (paths will be set later)."""

    st.markdown("---")
    st.markdown("### Upload metadata TSV")
    st.markdown(
        "Upload a tab-separated file describing your datasets. "
        "All editing files referenced in it must be accessible — "
        "for a local install they can be on your computer's file system; "
        "for the web app, upload editing files separately."
    )

    with st.expander("Metadata file format (click to expand)"):
        st.markdown(
            """
The metadata TSV must have a header row with these columns (tab-separated):

| Column | Required | Description |
|---|---|---|
| `label` | ✅ | Short name for this dataset (e.g. Ataxin-2) |
| `expt_files` | ✅ | Semicolon-separated paths to experiment files |
| `ctrl_files` | ✅ | Semicolon-separated paths to control files |
| `targets_file` | ✅ | Path to known-targets file |
| `background_files` | ❌ | Semicolon-separated paths to background (gDNA) files |

Example:
```
label	expt_files	ctrl_files	targets_file
Ataxin-2	data/Atx_1.txt;data/Atx_2.txt	data/Adar_1.txt;data/Adar_2.txt	targets/atx.txt
Thor	data/Thor_1.txt;data/Thor_2.txt	data/BG_1.txt;data/BG_2.txt	targets/thor.txt
```
            """
        )

    meta_file = st.file_uploader(
        "Metadata TSV",
        type=["tsv", "txt", "csv"],
        key="cdt_meta",
        label_visibility="collapsed",
    )

    if meta_file:
        try:
            content = meta_file.read().decode()
            meta_file.seek(0)
            sep = "\t" if "\t" in content.splitlines()[0] else ","
            df = pd.read_csv(io.StringIO(content), sep=sep)
            st.success(f"✅ Metadata loaded: {len(df)} datasets")
            st.dataframe(df, hide_index=True, use_container_width=True)
            # Return placeholder — actual file loading happens at run time
            return [{"label": "__metadata__", "meta_file": meta_file, "meta_df": df}]
        except Exception as exc:
            st.error(f"Could not parse metadata file: {exc}")

    return []


# ── Run pipeline ─────────────────────────────────────────────────────────────

def _run_cross_test(
    fe, tm, cd,
    model_file, datasets,
    editing_type, min_reads, min_edit_pct,
    out_dir_setting=None,
):
    with output_workspace(out_dir_setting) as workspace:
        data_dir = workspace / "data"
        out_dir  = workspace / "outputs"
        data_dir.mkdir(); out_dir.mkdir()

        # Load model
        rf_model = pickle.loads(model_file.read())

        st.markdown("---")
        st.markdown("### Running cross-dataset testing…")
        log_container = st.empty()

        all_results: dict[str, dict] = {}

        try:
            with capture_output(log_container):
                for ds in datasets:
                    if ds.get("label") == "__metadata__":
                        # Metadata TSV mode — files must be on local disk
                        meta_df = ds["meta_df"]
                        for _, row in meta_df.iterrows():
                            label = row["label"]
                            ds_out = out_dir / label
                            ds_out.mkdir(exist_ok=True)
                            result = _predict_one_dataset_from_row(
                                fe=fe, rf_model=rf_model, row=row,
                                editing_type=editing_type,
                                min_reads=min_reads, min_edit_pct=min_edit_pct,
                                out_dir=ds_out,
                            )
                            if result:
                                all_results[label] = result
                    else:
                        # Manual mode
                        label = ds["label"]
                        ds_dir = data_dir / label
                        ds_dir.mkdir(exist_ok=True)
                        ds_out = out_dir / label
                        ds_out.mkdir(exist_ok=True)

                        expt_paths = save_uploads(ds["expt"], ds_dir)
                        ctrl_paths = save_uploads(ds["ctrl"], ds_dir)
                        targets_path = None
                        if ds.get("targets"):
                            targets_path = save_single_upload(ds["targets"], ds_dir)

                        result = _predict_one_dataset(
                            fe=fe, rf_model=rf_model,
                            label=label,
                            expt_paths=expt_paths,
                            ctrl_paths=ctrl_paths,
                            targets_path=targets_path,
                            editing_type=editing_type,
                            min_reads=min_reads, min_edit_pct=min_edit_pct,
                            out_dir=ds_out,
                        )
                        if result:
                            all_results[label] = result

                # Summary plots
                if len(all_results) >= 2:
                    print("\nGenerating comparison plots...")
                    _make_comparison_plots(all_results, out_dir)

        except Exception as exc:
            st.error(f"❌ Error: {exc}")
            with st.expander("Full traceback"):
                import traceback
                st.code(traceback.format_exc())
            return

        # ── Results ──────────────────────────────────────────────────────────
        st.success("✅ Cross-dataset testing complete!")
        st.markdown("---")

        # Per-dataset predictions
        st.markdown("### Per-dataset top predictions")
        for label, res in all_results.items():
            with st.expander(f"📄 {label}", expanded=len(all_results) == 1):
                pred_path = out_dir / label / "predictions.tsv"
                if pred_path.exists():
                    df_pred = pd.read_csv(pred_path, sep="\t")
                    score_col = next(
                        (c for c in ["rf_probability", "final_probability"] if c in df_pred.columns),
                        df_pred.columns[-1],
                    )
                    df_show = df_pred.sort_values(score_col, ascending=False).head(30).reset_index(drop=True)
                    df_show.index += 1
                    st.dataframe(df_show, use_container_width=True, height=400)

        # Summary metrics table
        if all_results:
            st.markdown("### Summary metrics")
            summary_rows = []
            for label, res in all_results.items():
                row = {"Dataset": label}
                row.update({k: f"{v:.3f}" if isinstance(v, float) else v
                            for k, v in res.items() if k != "label"})
                summary_rows.append(row)
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

        # Comparison plots
        show_plots_in_dir(out_dir, title="Comparison plots", glob="comparison/*.png")

        # Per-dataset plots
        for label in all_results:
            show_plots_in_dir(out_dir / label, title=f"Plots — {label}", glob="**/*.png")

        # Download all
        st.markdown("---")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in out_dir.rglob("*"):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(out_dir))
        st.download_button(
            "⬇️ Download all results (ZIP)",
            data=zip_buf.getvalue(),
            file_name="smartie_cross_test.zip",
            mime="application/zip",
            use_container_width=True,
        )


def _predict_one_dataset(
    fe, rf_model,
    label, expt_paths, ctrl_paths, targets_path,
    editing_type, min_reads, min_edit_pct,
    out_dir: Path,
) -> dict | None:
    """Run prediction for a single dataset. Returns a metrics dict."""
    import numpy as np

    print(f"\n{'─'*50}")
    print(f"Dataset: {label}")

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
    print(f"  Genes: {len(gene_names):,}")

    features_df = fe.build_features(expt_dfs, ctrl_dfs, gene_names)
    feature_cols = [c for c in features_df.columns if c != "Name"]
    X = features_df[feature_cols].fillna(0).values
    probabilities = rf_model.predict_proba(X)[:, 1]

    results_df = pd.DataFrame({
        "gene": features_df["Name"].values,
        "rf_probability": probabilities,
    }).sort_values("rf_probability", ascending=False)
    results_df["rank"] = range(1, len(results_df) + 1)

    metrics = {}
    if targets_path:
        from train_model import load_targets
        targets_set = load_targets(str(targets_path))
        results_df["is_known_target"] = results_df["gene"].str.lower().isin(targets_set).astype(int)
        y_true  = results_df["is_known_target"].values
        y_score = results_df["rf_probability"].values
        if y_true.sum() > 0:
            from sklearn.metrics import roc_auc_score, average_precision_score
            metrics["ROC-AUC"]  = roc_auc_score(y_true, y_score)
            metrics["PR-AUC"]   = average_precision_score(y_true, y_score)
            for k in [10, 20, 50, 100]:
                if k <= len(results_df):
                    n_hit = results_df.head(k)["is_known_target"].sum()
                    metrics[f"P@{k}"] = n_hit / k
            print(f"  ROC-AUC: {metrics['ROC-AUC']:.3f}  PR-AUC: {metrics['PR-AUC']:.3f}")

    results_df.to_csv(out_dir / "predictions.tsv", sep="\t", index=False)
    _make_dataset_plots(results_df, out_dir, targets_set=targets_set)
    return metrics


def _predict_one_dataset_from_row(
    fe, rf_model, row,
    editing_type, min_reads, min_edit_pct,
    out_dir: Path,
) -> dict | None:
    """Same as _predict_one_dataset but reads file paths from a metadata row."""
    from train_model import parse_files as _parse_files

    expt_paths = [Path(p) for p in _parse_files(str(row["expt_files"]))]
    ctrl_paths = [Path(p) for p in _parse_files(str(row["ctrl_files"]))]
    targets_path = Path(str(row["targets_file"])) if pd.notna(row.get("targets_file")) else None

    return _predict_one_dataset(
        fe=fe, rf_model=rf_model,
        label=row["label"],
        expt_paths=expt_paths,
        ctrl_paths=ctrl_paths,
        targets_path=targets_path,
        editing_type=editing_type,
        min_reads=min_reads,
        min_edit_pct=min_edit_pct,
        out_dir=out_dir,
    )



def _make_dataset_plots(results_df, out_dir: Path, targets_set: set = None, venn_cutoff: int = 200):
    """
    With targets:    EPAR heatmap + Venn diagram
    Without targets: Score distribution + enrichment scatter
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    has_targets = bool(targets_set) and "is_known_target" in results_df.columns

    if has_targets:
        _epar_heatmap(results_df, targets_set, plots_dir)
        _venn(results_df, targets_set, plots_dir, venn_cutoff)
    else:
        _score_distribution(results_df, plots_dir)


def _epar_heatmap(results_df, targets_set: set, plots_dir: Path):
    """EPAR heatmap — deciles 10%–100% of reference list."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    deciles   = list(range(10, 110, 10))
    n_targets = len(targets_set)
    epar_vals = []

    for d in deciles:
        n_pred    = max(1, min(round(n_targets * d / 100), len(results_df)))
        top_genes = set(results_df.head(n_pred)["gene"].str.lower())
        epar_vals.append(len(top_genes & targets_set) / n_pred * 100)

    df_heat = pd.DataFrame({"EPAR (%)": epar_vals}, index=[f"{d}%" for d in deciles])

    fig, ax = plt.subplots(figsize=(3.5, 6))
    sns.heatmap(
        df_heat, ax=ax, annot=True, fmt=".1f", cmap="YlOrRd",
        vmin=0, vmax=100, linewidths=0.8, linecolor="white",
        cbar_kws={"label": "EPAR (%)", "shrink": 0.7},
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title("Equivalent Precision\nAgainst Reference (EPAR)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Decile of reference target list", fontsize=11)
    ax.tick_params(axis="y", rotation=0)

    for i, d in enumerate(deciles):
        n_ref = max(1, round(n_targets * d / 100))
        ax.text(1.35, i + 0.5, f"n={n_ref}", va="center", ha="left",
                fontsize=8, color="gray", transform=ax.get_yaxis_transform())

    fig.tight_layout()
    savefig(fig, plots_dir / "epar_heatmap")
    plt.close(fig)
    df_heat.to_csv(plots_dir / "epar_values.tsv", sep="\t")


def _venn(results_df, targets_set: set, plots_dir: Path, venn_cutoff: int):
    """Venn diagram — top-N SMARTIE predictions vs known targets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_genes    = set(results_df.head(venn_cutoff)["gene"].str.lower())
    overlap      = top_genes & targets_set
    only_smartie = top_genes - targets_set
    only_known   = targets_set - top_genes

    try:
        from matplotlib_venn import venn2
        fig, ax = plt.subplots(figsize=(6, 5))
        venn2(
            subsets=(len(only_smartie), len(only_known), len(overlap)),
            set_labels=(f"SMARTIE\ntop {venn_cutoff}", "Known\ntargets"),
            ax=ax, set_colors=("#4CAF50", "#2196F3"), alpha=0.6,
        )
    except ImportError:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
        ax.add_patch(plt.Circle((3.5, 3.5), 2.8, color="#4CAF50", alpha=0.4))
        ax.add_patch(plt.Circle((6.5, 3.5), 2.8, color="#2196F3", alpha=0.4))
        ax.text(2.2, 3.5, str(len(only_smartie)), ha="center", va="center",
                fontsize=16, fontweight="bold", color="#2E7D32")
        ax.text(5.0, 3.5, str(len(overlap)), ha="center", va="center",
                fontsize=18, fontweight="bold", color="#333")
        ax.text(7.8, 3.5, str(len(only_known)), ha="center", va="center",
                fontsize=16, fontweight="bold", color="#1565C0")
        ax.text(2.2, 0.5, f"SMARTIE only\n(top {venn_cutoff})", ha="center",
                fontsize=10, color="#2E7D32")
        ax.text(5.0, 0.2, f"Overlap\n({len(overlap)/max(1,venn_cutoff)*100:.0f}%)",
                ha="center", fontsize=10, color="#333")
        ax.text(7.8, 0.5, "Known only", ha="center", fontsize=10, color="#1565C0")

    ax.set_title(
        f"SMARTIE top {venn_cutoff} vs known targets\n"
        f"(overlap: {len(overlap)}/{venn_cutoff} = {len(overlap)/max(1,venn_cutoff)*100:.1f}%)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    savefig(fig, plots_dir / "venn_diagram")
    plt.close(fig)


def _score_distribution(results_df, plots_dir: Path):
    """Score distribution coloured by confidence band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scores = results_df["rf_probability"].values
    bins   = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for lo, hi, color, label in [
        (0.0, 0.4, "#B0BEC5", "Low (<0.4)"),
        (0.4, 0.7, "#FFB74D", "Mid (0.4–0.7)"),
        (0.7, 1.0, "#4CAF50", "High (>0.7)"),
    ]:
        mask = (scores >= lo) & (scores < hi + 0.001)
        ax.hist(scores[mask], bins=bins, color=color, edgecolor="white",
                alpha=0.9, label=label, linewidth=0.5)

    ax.axvline(0.5, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("RF Probability Score", fontsize=12)
    ax.set_ylabel("Number of genes", fontsize=12)
    ax.set_title("SMARTIE Score Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, plots_dir / "score_distribution")
    plt.close(fig)


def _make_comparison_plots(all_results: dict, out_dir: Path):
    """
    Multi-dataset comparison: EPAR heatmap with rows=RBPs, columns=deciles.
    Only generated when validation targets exist for at least 2 datasets.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    comp_dir = out_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)

    # Collect EPAR values per dataset
    epar_rows = {}
    decile_labels = [f"{d}%" for d in range(10, 110, 10)]

    for label, res in all_results.items():
        epar_path = out_dir / label / "plots" / "epar_values.tsv"
        if epar_path.exists():
            df_e = pd.read_csv(epar_path, sep="\t", index_col=0)
            epar_rows[label] = df_e["EPAR (%)"].values

    if len(epar_rows) < 2:
        # Not enough datasets with targets for a comparison heatmap
        return

    df_heat = pd.DataFrame(epar_rows, index=decile_labels).T  # rows=RBPs, cols=deciles

    fig, ax = plt.subplots(figsize=(max(8, len(decile_labels) * 0.9),
                                    max(3, len(epar_rows) * 0.7 + 1.5)))
    sns.heatmap(
        df_heat, ax=ax,
        annot=True, fmt=".1f",
        cmap="YlOrRd",
        vmin=0, vmax=100,
        linewidths=0.6, linecolor="white",
        cbar_kws={"label": "EPAR (%)", "shrink": 0.8},
        annot_kws={"size": 9, "weight": "bold"},
    )
    ax.set_title("EPAR across datasets and deciles",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Decile of reference target list", fontsize=11)
    ax.set_ylabel("Dataset / RBP", fontsize=11)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    savefig(fig, comp_dir / "epar_comparison_heatmap")
    plt.close(fig)
    df_heat.to_csv(comp_dir / "epar_comparison_data.tsv", sep="\t")
