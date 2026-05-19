# SMARTIE

SMARTIE (**S**ystematic **M**achine-learning **A**pproach for **R**BP **T**arget **I**dentification from **E**diting data) is a machine-learning ranker, based on the XGBoost algorithm, that ranks genes as likely targets of an RNA-binding protein (RBP) from TRIBE / STAMP data. The package ships a pretrained model (`SMARTIE.pkl`) that predicts RNA targets from TRIBE / STAMP data of any sample source.

SMARTIE can be used in two ways:

- **The SMARTIE application** — a graphical, point-and-click web interface. Recommended for users without a bioinformatics background.
- **The command-line interface** — `smartie-test`, `smartie-train`, and related commands, intended for scripting and batch analysis.

Both interfaces run the same underlying pipeline and produce identical results.

---

## Table of contents

### Part 1 — The SMARTIE application (graphical interface)

1. [Requirements](#1-requirements)
2. [Installation and prediction with the SMARTIE application](#2-installation-and-prediction-with-the-smartie-application)
3. [Output directory layout](#3-output-directory-layout)

### Part 2 — Command-line interface

4. [Installation and prediction from the command line](#4-installation-and-prediction-from-the-command-line)

### Advanced usage

Training a custom model, exhaustive feature selection, and troubleshooting are documented separately in [`docs/ADVANCED.md`](docs/ADVANCED.md).

---

# Part 1 — The SMARTIE application (graphical interface)

## 1. Requirements

Two prerequisites must be satisfied before running SMARTIE: the **input files** must conform to a supported format, and a small set of **Python dependencies** must be available. The dependencies are installed automatically during package installation, as described below.

### 1.1 Input file format

SMARTIE operates on **editing-site files** produced by a TRIBE or STAMP pipeline. One file is required per replicate, for two sample groups:

- **Experiment replicates** — RBP-ADAR (or RBP-APOBEC) fusion samples.
- **Control replicates** — ADAR-only (or APOBEC-only) baseline samples.

Each file must be a **tab-separated text file** with one row per editing site. A raw TRIBE/STAMP output file is wide: it carries per-base counts for both the RNA sample and the matched gDNA/wtRNA reference. The full column set is:

`Chr`, `Edit_coord`, `Name`, `Type`, `A_count`, `T_count`, `C_count`, `G_count`, `Total_count`, `A_count_gDNA/wtRNA`, `T_count_gDNA/wtRNA`, `C_count_gDNA/wtRNA`, `G_count_gDNA/wtRNA`, `Total_count_gDNA/wtRNA`, `Editbase_count`, `Total_count`, `Editbase_count_gDNA/wtRNA`, `Total_count_gDNA/wtRNA`

A representative example is shown below. Because the file is wide, the example is collapsed by default — expand it and scroll horizontally to view all columns.

<details>
<summary><b>Show a full raw file example</b> (scroll horizontally to see all columns)</summary>

<br>

```
Chr    Edit_coord  Name             Type    A_count  T_count  C_count  G_count  Total_count  A_count_gDNA/wtRNA  T_count_gDNA/wtRNA  C_count_gDNA/wtRNA  G_count_gDNA/wtRNA  Total_count_gDNA/wtRNA  Editbase_count  Total_count  Editbase_count_gDNA/wtRNA  Total_count_gDNA/wtRNA
chrY   1438171     Su(Ste):CR42410  INTRON  0        0        0        1        1            30                  0                   0                   0                   30                      1               1            0                          30
chr2R  9840430     CG1665           EXON    15       0        0        1        16           31                  0                   0                   0                   31                      1               16           0                          31
chr2R  17975371    dpr13            EXON    0        14       22       0        36           0                   58                  0                   0                   58                      22              36           0                          58
chr2R  17975979    dpr13            EXON    0        0        22       0        22           0                   54                  0                   0                   54                      22              22           0                          54
chr2R  17976517    dpr13            EXON    0        31       1        0        32           0                   37                  0                   0                   37                      1               32           0                          37
chr2R  17977621    dpr13            EXON    0        18       12       0        30           0                   38                  0                   0                   38                      12              30           0                          38
```

</details>

No reformatting of this file is required — SMARTIE reads it as produced by the TRIBE/STAMP pipeline. It uses only three quantities from each row: the **gene name** (`Name`), the **edit count** (`Editbase_count`), and the **total read count**. Note that the column header `Total_count` appears twice; when the file is read, the second occurrence is interpreted as `Total_count.1`, and SMARTIE uses this RNA-sample total. All remaining columns, including the per-base and gDNA/wtRNA columns, are ignored.

The following column names are accepted for each required quantity, so files from other pipelines generally do not need to be renamed either:

| Column      | Accepted names                                                          |
| ----------- | ----------------------------------------------------------------------- |
| Gene name   | `Name`, `name`, `Gene`, `gene`, `gene_id`, `Gene_id`                    |
| Edit count  | `Editbase_count`, `editbase_count`, `G_count`, `g_count` (A-to-G)       |
|             | `T_count`, `t_count` (C-to-T)                                           |
| Total count | `Total_count`, `total_count`, `Total_count.1`                           |

All other columns are ignored. Sites with `Total_count == 0`, and rows containing non-numeric counts, are discarded.

A **targets file** is required if precision and ROC plots are desired. This is a plain-text file listing known target gene names, one per line:

```
DDX3X
PUM1
PUM2
NUDT21
```

If no known targets are available, ranked predictions are still produced; only the precision and ROC plots are omitted. A CSV file containing a `Name` column is also accepted.

### 1.2 Dependencies

SMARTIE is a Python package and runs on **Linux, macOS, and Windows** (natively or via WSL) under **Python 3.10 or newer**.

Dependencies do not need to be installed manually. They are declared within the package and are resolved automatically by `pip` during installation (see [Section 2](#2-installation-and-prediction-with-the-smartie-application)). For reference, the package also includes a `requirements.txt` file listing the same dependencies:

```
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
streamlit
xgboost
lightgbm
```

If the dependencies need to be installed separately — for example, into a pre-existing environment — the following command may be used:

```bash
pip install -r requirements.txt
```

In the standard installation procedure this step is performed automatically.

---

## 2. Installation and prediction with the SMARTIE application

The SMARTIE application is a web interface that runs locally on the user's own machine. The package is installed once, a single command launches the application, and a browser window opens automatically; no further use of the terminal is required.

### Step 1 — Install SMARTIE

In a terminal (Terminal on macOS/Linux, or Command Prompt / PowerShell on Windows), run:

```bash
pip install git+https://github.com/toolsmnl/SMARTIE.git
```

This installs SMARTIE, resolves all dependencies automatically, and bundles the pretrained `SMARTIE.pkl` model. Installation is required only once.

For users new to Python, `pipx` is recommended, as it installs SMARTIE within an isolated environment:

```bash
# macOS
brew install pipx

# Ubuntu / Debian
sudo apt install pipx

# Then, on any operating system:
pipx install git+https://github.com/toolsmnl/SMARTIE.git
```

### Step 2 — Launch the application

In the terminal, run:

```bash
SMARTIE
```

A browser tab opens automatically at **http://localhost:8501**.

> **Windows:** if a *"command not recognized"* error appears, use the following equivalent command:
> ```bash
> python -m smartie_mnl.app.launcher
> ```

### Step 3 — Run a prediction

The application provides the following pages, accessible from the sidebar:

| Page                      | Function                                                                                       |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| **Home**                  | Overview, input-format reference, and orientation.                                            |
| **Predict Targets**       | Upload editing-site files to obtain a ranked gene list and accompanying plots.                 |
| **Train Your Model**      | Upload data and known targets to train a custom model (see [`docs/ADVANCED.md`](docs/ADVANCED.md)). |
| **Cross-Dataset Testing** | Evaluate a model across several datasets simultaneously.                                       |

To predict targets using the bundled model:

1. Select **Predict Targets** in the sidebar.
2. Upload the **experiment replicate files** (RBP-ADAR fusion samples).
3. Upload the **control replicate files** (ADAR-only samples).
4. *(Optional)* Upload a **targets file** of known genes to enable precision and ROC plots.
5. Click **Run**. A live log reports progress as the pipeline executes.
6. On completion, the ranked predictions and plots are displayed in the browser. Use the **Download** button to export all results as a single archive.

The pretrained model is applied automatically; no model file needs to be supplied.

---

## 3. Output directory layout

The structure of the results is identical regardless of the interface used; the two interfaces differ only in *where* the results are placed:

- **Command-line interface.** Results are written directly to the directory specified by `--outdir`. If the directory does not exist, it is created.
- **SMARTIE application.** Results are displayed in the browser and made available through the **Download** button as a single ZIP archive. When extracted, the archive has exactly the structure shown below.

A run may cover a **single dataset** or **multiple datasets**. The layout is described for both cases below.

### 3.1 Single-dataset prediction

A single-dataset run analyses one experiment/control pair:

- *Command line* — a metadata TSV containing **one row**.
- *Application* — the **Predict Targets** page, with one set of experiment and control files uploaded.

The output contains one dataset folder, named after the dataset's `label`, together with the shared `comparison/` folder:

```
results/
├── MyRBP_K562/                  # one folder, named after the dataset label
│   ├── predictions.tsv          # ranked gene list (sorted by rf_probability)
│   └── gene_features.tsv        # all features computed per gene
├── all_results.tsv              # flat summary (a single dataset, in this case)
└── comparison/
    ├── precision_heatmap.{pdf,png}
    ├── precision_lineplot.{pdf,png}
    ├── roc_curves.{pdf,png}
    ├── roc_auc_barchart.{pdf,png}
    ├── pr_curves.{pdf,png}
    ├── score_distributions.{pdf,png}
    ├── feature_value_heatmap.{pdf,png}
    └── summary.tsv
```

For a single-dataset run, the per-dataset figures — ROC, precision–recall, score distribution, and precision plots — are the relevant outputs. The cross-dataset comparison figure `gene_rank_overlap` is not produced, as it requires at least two datasets to compare.

### 3.2 Multi-dataset prediction

A multi-dataset run analyses several experiment/control pairs in one execution:

- *Command line* — a metadata TSV containing **multiple rows**, one per dataset.
- *Application* — the **Cross-Dataset Testing** page, or a metadata TSV supplied to it.

Each dataset receives its own folder; the `comparison/` folder then aggregates results across all of them:

```
results/
├── MyRBP_K562/
│   ├── predictions.tsv
│   └── gene_features.tsv
├── MyRBP_HEK/
│   ├── predictions.tsv
│   └── gene_features.tsv
├── MyRBP_HeLa/
│   ├── predictions.tsv
│   └── gene_features.tsv
├── all_results.tsv              # flat summary across all datasets
└── comparison/
    ├── precision_heatmap.{pdf,png}
    ├── precision_lineplot.{pdf,png}
    ├── roc_curves.{pdf,png}
    ├── roc_auc_barchart.{pdf,png}
    ├── pr_curves.{pdf,png}
    ├── score_distributions.{pdf,png}
    ├── feature_value_heatmap.{pdf,png}
    ├── gene_rank_overlap.{pdf,png}
    └── summary.tsv
```

In a multi-dataset run, every figure in `comparison/` becomes informative: each dataset contributes a row to the heatmaps, a curve to the ROC and precision–recall plots, and a bar to the bar charts, and `gene_rank_overlap` reports the agreement between datasets.

> Evaluating several *models* across datasets — as opposed to several datasets with one model — is a separate workflow described in [`docs/ADVANCED.md`](docs/ADVANCED.md). In that case the `comparison/` folder is nested under each model's own subfolder.

### The predictions file

`predictions.tsv` is the primary output — a ranked list of candidate targets:

```
gene      rf_probability    rank
PUM2      0.9412            1
DDX3X     0.8830            2
RBFOX2    0.7621            3
```

The `rank` column orders genes by predicted likelihood of being a target, with rank 1 representing the strongest candidate.

### The plots

Each plot is saved both as **`.pdf`** (suitable for manuscript figures) and **`.png`** (for rapid inspection), within the `comparison/` folder.

| File                    | Description                                                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `precision_heatmap`     | Rows correspond to datasets, columns to the top-K% of known targets, and cells to the precision in that range. The principal summary figure. |
| `precision_lineplot`    | The same data presented as a line plot, with one line per dataset.                                                    |
| `roc_curves`            | ROC curve per dataset; the legend reports AUC. A dashed diagonal indicates the random baseline.                       |
| `roc_auc_barchart`      | Bar chart of ROC-AUC per dataset, sorted in descending order.                                                          |
| `pr_curves`             | Precision–recall curve per dataset, with average precision shown in the legend. More informative than ROC when targets are rare. |
| `score_distributions`   | Histogram of `rf_probability` for targets versus non-targets. Concentration of targets at high probabilities indicates good model calibration. |
| `feature_value_heatmap` | Mean feature value per dataset, useful for identifying datasets with atypical feature distributions.                  |
| `gene_rank_overlap`     | Pairwise overlap of the top-ranked genes between datasets.                                                             |

A machine-readable `summary.tsv` file, reporting precision at each threshold for every dataset, accompanies the plots.

---

# Part 2 — Command-line interface

## 4. Installation and prediction from the command line

The command-line interface provides the same functionality as the application and is intended for scripting and batch analysis. Both interfaces produce identical results.

### Installation

The installation command is the same as for the application:

```bash
pip install git+https://github.com/toolsmnl/SMARTIE.git
```

This installs the `smartie-test`, `smartie-train`, and related commands. Installation can be verified with:

```bash
smartie-test --help
```

The command's help text should be displayed. If a `command not found` error occurs, refer to the troubleshooting section of [`docs/ADVANCED.md`](docs/ADVANCED.md).

### Step 1 — Prepare a metadata TSV

Create a tab-separated file describing each dataset to be analysed:

```
label       expt_files                                          ctrl_files                                          targets_file
MyRBP_K562  data/MyRBP_K562_rep1.txt;data/MyRBP_K562_rep2.txt   data/ADAR_K562_rep1.txt;data/ADAR_K562_rep2.txt   known/MyRBP_K562_known.txt
MyRBP_HEK   data/MyRBP_HEK_rep1.txt;data/MyRBP_HEK_rep2.txt     data/ADAR_HEK_rep1.txt;data/ADAR_HEK_rep2.txt     known/MyRBP_HEK_known.txt
```

| Column             | Required | Description                                                                       |
| ------------------ | -------- | --------------------------------------------------------------------------------- |
| `label`            | Yes      | Short dataset name, used in output paths and plot labels.                         |
| `expt_files`       | Yes      | Semicolon-separated paths to experiment replicate files.                          |
| `ctrl_files`       | Yes      | Semicolon-separated paths to control replicate files.                             |
| `targets_file`     | Yes      | Plain-text file of known target genes, one per line. A placeholder may be used if none are available. |
| `background_files` | No       | Semicolon-separated background (e.g. gDNA) files for site-level noise filtering.  |

Replicate files within a single cell are separated by semicolons. The legacy column names `dataset_name`, `bg_files`, and `validation_targets` are also accepted.

### Step 2 — Run the prediction

```bash
smartie-test \
    --metadata my_data.tsv \
    --outdir results/
```

The bundled `SMARTIE.pkl` model is located automatically, and the default per-site coverage filter (`--min-reads 20`) is applied.

To use an alternative model — for example, one trained by the user:

```bash
smartie-test \
    --metadata my_data.tsv \
    --model    path/to/your/rf_model.pkl \
    --outdir   results/
```

### Prediction options

| Flag                         | Default              | Description                                                                                  |
| ---------------------------- | -------------------- | -------------------------------------------------------------------------------------------- |
| `--min-reads N`              | 20                   | Discard editing sites with fewer than N total reads before feature computation.              |
| `--min-edit-pct F`           | 0.0                  | Discard sites with an edit percentage below F (range 0–100). Useful for noisy controls.      |
| `--editing-type {AtoG,CtoT}` | AtoG                 | A-to-G for ADAR-based TRIBE; C-to-T for APOBEC-based STAMP.                                   |
| `--background F [F ...]`     | none                 | Derive a site-level background filter from gDNA / no-enzyme controls and apply it before feature extraction. |
| `--no-background-filter`     | off                  | Disable background filtering even if one is bundled with the model.                          |
| `--normalization {simple,median_of_ratios}` | simple | Replicate-normalisation strategy. `simple` is appropriate for most cases.                  |
| `--outdir DIR`               | `outputs/cross_test` | Output directory; created if it does not exist.                                              |

The complete list of options is available via `smartie-test --help`. The output is structured exactly as described in [Section 3](#3-output-directory-layout).

---

## Advanced usage

Instructions for training a custom model, performing exhaustive feature selection, and resolving common issues are provided in [`docs/ADVANCED.md`](docs/ADVANCED.md). The pretrained model documented in Parts 1 and 2 is sufficient for most analyses.

## Citation

If SMARTIE contributes to your work, please cite *[paper or preprint placeholder]*.

## License

SMARTIE is released under the [MIT License](LICENSE).

## Contact

Issues and feature requests: <https://github.com/toolsmnl/SMARTIE/issues>
