# SMARTIE

SMARTIE (**S**ystematic **M**achine-learning **A**pproach for **R**BP **T**argets **I**dentified by **E**diting) is a machine-learning tool based on the XGBoost algorithm, that ranks genes as likely targets of an RNA-binding protein (RBP) from TRIBE / STAMP data. The package ships a ready-to-use pretrained model (`SMARTIE.pkl`).

SMARTIE can be used in two ways:

- **The SMARTIE application**: a graphical, point-and-click web interface. Recommended for users without a bioinformatics background.
- **The command-line interface**: `smartie-test`, `smartie-train`, and related commands, intended for scripting and batch analysis.

---

## Table of contents

### Part 1: The SMARTIE application (graphical interface)

1. [Requirements](#1-requirements)
2. [Installation and prediction with the SMARTIE application](#2-installation-and-prediction-with-the-smartie-application)
3. [Output directory layout](#3-output-directory-layout)

### Part 2: Command-line interface

4. [Installation and prediction from the command line](#4-installation-and-prediction-from-the-command-line)

### Advanced usage

Training a custom model, exhaustive feature selection, and troubleshooting are documented separately in [`docs/ADVANCED.md`](docs/ADVANCED.md).

---

# Part 1: The SMARTIE application (graphical interface)

## 1 Requirements

### 1.1 Input Files

For running SMARTIE, Input files in a **tab-separated text file (.tsv/.txt)** are required. A representative example is shown below (Expand it and scroll horizontally to view all columns.)

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

The following column names are accepted for each required quantity :

| Column      | Accepted names                                                          |
| ----------- | ----------------------------------------------------------------------- |
| Gene name   | `Name`, `name`, `Gene`, `gene`, `gene_id`, `Gene_id`                    |
| Edit count  | `Editbase_count`, `editbase_count`, `G_count`, `g_count` (A-to-G)       |
|             | `T_count`, `t_count` (C-to-T)                                           |
| Total count | `Total_count`, `total_count`, `Total_count.1`                           |

Each experimental and control replicate has to be provided as a separate file. For example:
- **Experiment replicates**: RBP-Exp-1 and RBP-Exp-2.
- **Control replicates**: Ctrl-1 and Ctrl-2.

Optionally, A **targets file** (if available) can be provided to validate the list of targets in a plain-text format listing known target gene names, one per line:

```
DDX3X
PUM1
PUM2
NUDT21
```

### 1.2 Dependencies

SMARTIE runs on **Linux, macOS, and Windows** under **Python 3.10 or newer**.

Dependencies are installed automatically by `pip`. (see [Section 2](#2-installation-and-prediction-with-the-smartie-application)). In case of an error where dependencies need to be installed manually, a `requirements.txt` file bundled with the package:

```bash
pip install -r requirements.txt
```

---

## 2. Installation and prediction with the SMARTIE application

The SMARTIE application is a web interface that runs locally.

### Step 1: Install SMARTIE

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

### Step 2: Launch the application

In the terminal, run:

```bash
SMARTIE
```

A browser tab opens automatically at **http://localhost:8501**.

> **Windows:** if a *"command not recognized"* error appears, use the following equivalent command:
> ```bash
> python -m smartie_mnl.app.launcher
> ```

### Step 3: Run a prediction

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
4. *(Optional)* Upload a **targets file** of known genes to enable the EPAR heatmap and Venn diagram.
5. Click **Run**. A live log reports progress as the pipeline executes.
6. On completion, the ranked predictions and plots are displayed in the browser. Use the **Download** button to export all results as a single archive.

The pretrained model is applied automatically; no model file needs to be supplied.

### Tutorial: try it with the example data

The repository includes a small example dataset using Ataxin-2 experiment [Singh et al, 2021](https://elifesciences.org/articles/60326) and control files [Koppaka et al, 2024](https://wellcomeopenresearch.org/articles/10-112) as (['example_data.7z'](https://github.com/toolsmnl/SMARTIE/blob/23540c2c36e2557d3ee127c177a0feb206aacf7a/Example_Data.7z)) which unzips to [`example_data/`] folder. It can be used to verify that the installation is working and to see what a complete SMARTIE run looks like before applying it to a real experiment. The files are:

| File                                  | Role                                  |
| ------------------------------------- | ------------------------------------- |
| `RBP_Exp_1.txt`, `RBP_Exp_2.txt`      | Experiment replicates                 |
| `Ctrl_1.txt`, `Ctrl_2.txt`            | Control replicates                    |
| `Targets.csv`                         | Known target genes (for evaluation)   |

If you installed SMARTIE via `pip` without cloning the repository, download the example dataset as (['example_data.7z'](https://github.com/toolsmnl/SMARTIE/blob/23540c2c36e2557d3ee127c177a0feb206aacf7a/Example_Data.7z)) of the GitHub repository.

To run the example end-to-end:

1. Launch the application with `SMARTIE` and select **Predict Targets** in the sidebar.
2. Under **Experiment replicates**, upload `RBP_Exp_1.txt` and `RBP_Exp_2.txt`.
3. Under **Control replicates**, upload `Ctrl_1.txt` and `Ctrl_2.txt`.
4. Under **Targets file**, upload `Targets.csv`.
5. Click **Run** and wait for the progress log to finish.
6. The browser displays the ranked predictions, the EPAR heatmap, and the Venn diagram. Use **Download** to export everything as `smartie_predictions.zip`.

If the run completes and the plots appear without errors, the installation is working correctly. The same dataset is used in the [command-line tutorial](#tutorial-run-the-example-data-from-the-command-line), and the two outputs should be identical.

---

## 3. Output directory layout

- **Command-line interface.** Results are written directly to the directory specified by `--outdir`. If the directory does not exist, it is created.
- **SMARTIE application.** Results are displayed in the browser and made available through the **Download** button as a single ZIP archive (`smartie_predictions.zip`). The archive contains the same files described below.

**The plots produced depend on whether a known-targets file was supplied.** When known targets are provided, SMARTIE generates the evaluation plots (EPAR heatmap and Venn diagram); when they are not, it generates a score-distribution plot instead. See [Section 3.3](#33-the-plots) for details.

A run may cover a **single dataset** or **multiple datasets**; the layout for each case is described below.

### 3.1 Single-dataset prediction

A single-dataset run analyses one experiment/control pair, for example an Ataxin-2 (`Atx2`) dataset:

- *Command line*: a metadata TSV containing **one row**.
- *Application*: the **Predict Targets** page, with one set of experiment and control files uploaded.

If **no** known-targets file is supplied, the `plots/` subfolder contains a score-distribution plot:

```
    └── plots/
        └── score_distribution.{pdf,png}
```


When targets are provided for validation, additional plots are generated which include a EPAR heatmap and Venn diagram comparing between the SMARTIE predictions and the provided targets.

```
results/
└── Atx2/
    ├── predictions.tsv          # ranked gene list (sorted by rf_probability)
    ├── gene_features.tsv        # all features computed per gene
    └── plots/
        ├── epar_heatmap.{pdf,png}    # evaluation plot, requires a known-targets file
        ├── epar_values.tsv          # raw numbers behind the EPAR heatmap
        └── venn_diagram.{pdf,png}    # evaluation plot, requires a known-targets file
```


### 3.2 Multi-dataset prediction

A multi-dataset run analyses several experiment/control pairs in one execution:

- *Command line*: a metadata TSV containing **multiple rows**, one per dataset.
- *Application*: the **Cross-Dataset Testing** page.

Each dataset receives its own folder, structured exactly as in the single-dataset case. In addition, a `comparison/` folder collects the cross-dataset figure:

```
results/
├── Atx2/
│   ├── predictions.tsv
│   ├── gene_features.tsv
│   └── plots/
│       ├── epar_heatmap.{pdf,png}
│       ├── epar_values.tsv
│       └── venn_diagram.{pdf,png}
├── Hrp-48/
│   ├── predictions.tsv
│   ├── gene_features.tsv
│   └── plots/
│       └── ...
├── Thor/
│   ├── predictions.tsv
│   ├── gene_features.tsv
│   └── plots/
│       └── ...
└── comparison/
    └── multi_rbp_epar_heatmap.{pdf,png}    # generated when ≥2 datasets have known targets
```

The `comparison/` folder is produced only when at least two datasets have an associated known-targets file; the `multi_rbp_epar_heatmap` places those datasets on a single heatmap (one row per dataset) for direct comparison.

> Evaluating several *models* across datasets, rather than several datasets with one model, is a separate workflow described in [`docs/ADVANCED.md`](docs/ADVANCED.md).

### The predictions file

`predictions.tsv` is the primary output, a ranked list of candidate targets:

```
gene      rf_probability    rank
dpr13     0.9412            1
NPF       0.8830            2
CG1665    0.7621            3
```

The `rank` column orders genes by predicted likelihood of being a target, with rank 1 representing the strongest candidate.

### 3.3 The plots

Each plot is saved both as **`.pdf`** (for manuscript figures) and **`.png`** (for rapid inspection), inside the dataset's `plots/` subfolder. Which plots appear depends on whether a known-targets file was provided.

| Plot                     | When produced                          | Description                                                                                                                                                       |
| ------------------------ | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `epar_heatmap`           | A known-targets file is provided        | Enrichment-Precision-At-Rank heatmap. The known-target list is partitioned into deciles (10%-100%); for each decile, an equivalent number of top-ranked SMARTIE predictions is intersected with the full target list, and the percentage overlap is shown. Each cell is annotated with the number of predictions (`n=`) used at that decile. |
| `venn_diagram`           | A known-targets file is provided        | Overlap between the top SMARTIE predictions and the full known-target list, with the percentage overlap reported in the title.                                    |
| `score_distribution`     | No known-targets file is provided       | Histogram of `rf_probability` across all genes, coloured by confidence band, with the number of high-confidence genes annotated.                                  |
| `multi_rbp_epar_heatmap` | Multi-dataset run, ≥2 datasets with targets | The EPAR heatmap extended across datasets, with one row per dataset (RBP) and deciles as columns. Written to the `comparison/` folder.                               |

The raw values behind the EPAR heatmap are also written as `epar_values.tsv`, allowing the figure to be reproduced or re-plotted independently.

---

# Part 2: Command-line interface

## 4. Installation and prediction from the command line

The command-line interface provides the same functionality as the application and is intended for scripting and batch analysis.

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

### Step 1: Prepare a metadata TSV

Create a tab-separated file describing each dataset to be analysed:

```
label       expt_files                                          ctrl_files                                          targets_file
MyRBP_A  data/MyRBP_A_rep1.txt;data/MyRBP_A_rep2.txt   data/Ctrl_A_rep1.txt;data/Ctrl_B_rep2.txt   known/MyRBP_A_known.txt
MyRBP_B  data/MyRBP_B_rep1.txt;data/MyRBP_B_rep2.txt     data/Ctrl_B_rep1.txt;data/Ctrl_B_rep2.txt     known/MyRBP_B_known.txt
```

| Column             | Required | Description                                                                       |
| ------------------ | -------- | --------------------------------------------------------------------------------- |
| `label`            | Yes      | Short dataset name, used in output paths and plot labels.                         |
| `expt_files`       | Yes      | Semicolon-separated paths to experiment replicate files.                          |
| `ctrl_files`       | Yes      | Semicolon-separated paths to control replicate files.                             |
| `targets_file`     | Yes      | Plain-text file of known target genes, one per line. A placeholder may be used if none are available. |

Replicate files within a single cell are separated by semicolons. 

### Step 2: Run the prediction

```bash
smartie-test \
    --metadata my_data.tsv \
    --outdir results/
```

The bundled `SMARTIE.pkl` model is located automatically.

### Tutorial: run the example data from the command line

The same example dataset used in the [GUI tutorial](#tutorial-try-it-with-the-example-data) can be run from the command line. The files (`RBP_Exp_1.txt`, `RBP_Exp_2.txt`, `Ctrl_1.txt`, `Ctrl_2.txt`, `Targets.csv`) are in the [`example_data.7z`] file of the repository.

1. From the repository root, create a metadata TSV pointing at the example files. Save it as `example_data/metadata.tsv`:

```
label        expt_files                                              ctrl_files                                          targets_file
Example_RBP  example_data/RBP_Exp_1.txt;example_data/RBP_Exp_2.txt   example_data/Ctrl_1.txt;example_data/Ctrl_2.txt     example_data/Targets.csv
```

The columns are separated by tabs and the replicate files within each cell by semicolons.

2. Run the prediction:

```bash
smartie-test \
    --metadata example_data/metadata.tsv \
    --outdir   results/example/
```

3. When the command finishes, the output is organised as follows:

```
results/example/
└── Example_RBP/
    ├── predictions.tsv
    ├── gene_features.tsv
    └── plots/
        ├── epar_heatmap.{pdf,png}
        ├── epar_values.tsv
        └── venn_diagram.{pdf,png}
```

`predictions.tsv` is the ranked candidate list, and the EPAR heatmap and Venn diagram show how strongly the top predictions overlap with the known targets in `Targets.csv`. If both plots appear and `predictions.tsv` is populated, the command-line installation is working correctly.

### Prediction options

| Flag                         | Default              | Description                                                                                  |
| ---------------------------- | -------------------- | -------------------------------------------------------------------------------------------- |
| `--min-reads N`              | 20                   | Discard editing sites with fewer than N total reads before feature computation.              |
| `--min-edit-pct F`           | 0.0                  | Discard sites with an edit percentage below F (range 0-100). Useful for noisy controls.      |
| `--editing-type {AtoG,CtoT}` | AtoG                 | A-to-G for ADAR-based TRIBE; C-to-T for APOBEC-based STAMP.                                   |
| `--normalization {simple,median_of_ratios}` | simple | Replicate-normalisation strategy. `simple` is appropriate for most cases.                  |
| `--outdir DIR`               | `outputs/cross_test` | Output directory; created if it does not exist.                                              |

The complete list of options is available via `smartie-test --help`. The output is structured exactly as described in [Section 3](#3-output-directory-layout).

---

## Advanced usage

Instructions for training a custom model, performing exhaustive feature selection, and resolving common issues are provided in [`docs/ADVANCED.md`](docs/ADVANCED.md). The pretrained model documented in Parts 1 and 2 is sufficient for most analyses.

## Citation

If SMARTIE contributes to your work, please cite the bioRxiv preprint:

> Koppaka O, Kumar U, Ahuja G, Yadav R, Bakthavachalu B (2026).
> *SMARTIE: A Machine-Learning approach for investigating RBP-RNA interactions identified by Editing.*
> bioRxiv 2026.05.18.726004. <https://doi.org/10.64898/2026.05.18.726004>

## License

SMARTIE is released under the [GNU General Public License v3.0 or later](LICENSE). This permits free use, modification, and redistribution of the code for any purpose, provided that any redistributed modified version is itself released under GPL-3.0 with source code made available. Internal use, including in commercial settings, does not trigger any obligation. Citation of the manuscript above is expected for any published work that uses SMARTIE.

## Contact

Issues and feature requests: <https://github.com/toolsmnl/SMARTIE/issues>
