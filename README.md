# SMARTIE

**S**ystematic **M**achine-learning **A**pproach for **R**BP **T**arget **I**dentification from **E**diting data.

SMARTIE is a Random Forest classifier that ranks genes as likely targets of an RNA-binding protein (RBP) from TRIBE / STAMP A-to-G editing data. The package ships a pretrained model (`SMARTIE.pkl`) so you can predict on your own data immediately, without training anything.

If you have editing-site files from an RBP-ADAR / RBP-APOBEC fusion experiment and a control, you can install SMARTIE and get a ranked list of candidate targets in one command. Everything else in this README is optional.

---

## Table of contents

1. [Installation](#installation)
2. [Quick start — predict on your data](#quick-start--predict-on-your-data-with-the-pretrained-model)
3. [Input file formats](#input-file-formats)
4. [Plots that are generated](#plots-that-are-generated-during-testing)
5. [Output directory layout](#output-directory-layout)
6. [Useful prediction options](#useful-prediction-options)
7. [Advanced: cross-dataset / LOO testing with multiple models](#advanced-cross-dataset--loo-testing-with-multiple-models)
8. [Training your own model](#training-your-own-model)
9. [Brute-force feature selection](#brute-force-feature-selection-advanced)
10. [Troubleshooting](#troubleshooting)

---

## Installation

SMARTIE is a pure-Python package. It works on Linux, macOS, and Windows (via WSL or native), with Python 3.10 or newer.

### The one-line install (recommended for everyone)

```bash
pip install git+https://github.com/<your-username>/SMARTIE.git
```

That's it. The pretrained model is bundled with the package, so you can start predicting right after the install finishes.

### If you don't already have Python set up

If you have never used Python before, the easiest way is `pipx`, which installs SMARTIE in its own isolated environment and makes the commands available everywhere:

```bash
# macOS
brew install pipx

# Ubuntu / Debian
sudo apt install pipx

# Then:
pipx install git+https://github.com/<your-username>/SMARTIE.git
```

After this, `smartie-test`, `smartie-train`, and the other commands work from any terminal, in any directory, with no `conda activate` or `source venv/bin/activate` needed.

### If you use conda

```bash
conda create -n smartie python=3.11 pip
conda activate smartie
pip install git+https://github.com/<your-username>/SMARTIE.git
```

### Development install (if you cloned the repo)

```bash
git clone https://github.com/<your-username>/SMARTIE.git
cd SMARTIE
pip install -e .
```

### Verify the install

```bash
smartie-test --help
```

You should see the help text for the prediction command. If you get `command not found`, see the [Troubleshooting](#troubleshooting) section.

---

## Quick start — predict on your data with the pretrained model

The pretrained `SMARTIE.pkl` is bundled inside the package. You don't need to know where it lives — `smartie-test` finds it automatically when you don't pass `--model`.

### Step 1 — Prepare your editing-site files

You need TRIBE or STAMP-style editing-site files for:

- **Experiment replicates**: your RBP-ADAR (or RBP-APOBEC) fusion samples.
- **Control replicates**: ADAR-only (or APOBEC-only) baseline samples.

Each file is a tab-separated text file with one row per editing site. SMARTIE will accept any of these column names for the gene, edit count, and total read count:

| Column      | Accepted names                                                          |
| ----------- | ----------------------------------------------------------------------- |
| Gene name   | `Name`, `name`, `Gene`, `gene`, `gene_id`, `Gene_id`                    |
| Edit count  | `Editbase_count`, `editbase_count`, `G_count`, `g_count` (A-to-G)       |
|             | `T_count`, `t_count` (C-to-T)                                           |
| Total count | `Total_count`, `total_count`, `Total_count.1`                           |

Any other columns in the file are ignored.

### Step 2 — Write a metadata TSV

Make a tab-separated file describing each dataset you want to predict on. Minimum schema:

```
label       expt_files                                     ctrl_files                                     targets_file
MyRBP_K562  data/MyRBP_K562_rep1.txt;data/MyRBP_K562_rep2.txt   data/ADAR_K562_rep1.txt;data/ADAR_K562_rep2.txt   known/MyRBP_K562_known.txt
MyRBP_HEK   data/MyRBP_HEK_rep1.txt;data/MyRBP_HEK_rep2.txt     data/ADAR_HEK_rep1.txt;data/ADAR_HEK_rep2.txt     known/MyRBP_HEK_known.txt
```

- Replicate files are separated by **semicolons** in a single cell.
- `targets_file` is a plain text file of known target gene names, one per line. It's used only to compute precision; if you don't have known targets, put any single placeholder name there — SMARTIE will still produce ranked predictions, you just won't get precision plots for that dataset.
- Optional column `background_files` (semicolon-separated) adds per-dataset background filtering for genomic-DNA / no-enzyme controls.

### Step 3 — Run the prediction

```bash
smartie-test \
    --metadata my_data.tsv \
    --outdir results/
```

That's the entire command. The default `--min-reads 20` per-site coverage filter is applied automatically.

If you want to point at a different model file (e.g. one you trained yourself):

```bash
smartie-test \
    --metadata my_data.tsv \
    --model    path/to/your/rf_model.pkl \
    --outdir   results/
```

---

## Input file formats

### Editing-site file (one per replicate)

A tab-separated text file. Required columns (any of the accepted names from the table above):

```
Name    Editbase_count    Total_count
ACTB    12                420
GAPDH   3                 350
DDX3X   45                530
...
```

Sites with `Total_count == 0` and rows with non-numeric counts are silently dropped. Sites below `--min-reads` (default 20) are filtered before any features are computed.

### Metadata TSV

| Column             | Required | What it is                                                                           |
| ------------------ | -------- | ------------------------------------------------------------------------------------ |
| `label`            | yes      | Short dataset name. Used in output paths and plot labels.                            |
| `expt_files`       | yes      | Semicolon-separated paths to experiment replicate files.                             |
| `ctrl_files`       | yes      | Semicolon-separated paths to control replicate files.                                |
| `targets_file`     | yes      | Text file of known target genes, one per line. Lowercase / whitespace are tolerated. |
| `background_files` | no       | Semicolon-separated background (e.g. gDNA) files for site-level noise filtering.     |

Legacy column names `dataset_name` (for `label`), `bg_files` (for `ctrl_files`), and `validation_targets` (for `targets_file`) are also accepted.

### Targets file

A plain text file, one gene name per line:

```
DDX3X
PUM1
PUM2
NUDT21
...
```

CSV with a `Name` column also works.

---

## Plots that are generated during testing

`smartie-test` writes its plots under `<outdir>/comparison/` (single-model run) or `<outdir>/<model_label>/comparison/` (multi-model run). Every plot is saved as both **`.pdf`** (for figures in a manuscript) and **`.png`** (for quick viewing).

| File                                 | What it shows                                                                                                                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `precision_heatmap.{pdf,png}`        | Rows = your test datasets, columns = "top-K% of known targets" (10 %, 20 %, …, 100 %), cell = precision in the top-K predictions. The headline plot.                           |
| `precision_lineplot.{pdf,png}`       | The same data as a line plot — one line per test dataset, x-axis = depth as percentage of known targets, y-axis = precision.                                                   |
| `roc_curves.{pdf,png}`               | ROC curve per test dataset; legend reports AUC. Random baseline shown as a dashed diagonal.                                                                                    |
| `roc_auc_barchart.{pdf,png}`         | Bar chart of ROC-AUC per dataset, sorted high to low. Quickest way to see which datasets the model generalises to.                                                             |
| `pr_curves.{pdf,png}`                | Precision–Recall curve per dataset with average-precision (AP) in the legend. More informative than ROC when targets are rare.                                                 |
| `score_distributions.{pdf,png}`      | Histogram of `rf_probability` overlaid for targets vs non-targets, one panel per dataset. If targets pile up at high probabilities, the model is well-calibrated for that set. |
| `feature_value_heatmap.{pdf,png}`    | Mean feature value per dataset, normalised across datasets. Helps spot which datasets have unusual feature distributions.                                                      |
| `gene_rank_overlap.{pdf,png}`        | Pairwise overlap (top-N intersection) of ranked predictions between datasets — does the model put similar genes at the top across datasets?                                    |
| `dataset_precision_barchart.{pdf,png}` | Precision @ each depth threshold per dataset.                                                                                                                                |

Plus per-dataset machine-readable outputs:

- `predictions.tsv` — every gene with `rf_probability`, ranked.
- `gene_features.tsv` — every feature value SMARTIE computed for that dataset.
- `summary.tsv` — precision @ each threshold for every dataset, in long form.

---

## Output directory layout

```
results/
├── DatasetA/
│   ├── predictions.tsv          # ranked gene list (sorted by rf_probability)
│   └── gene_features.tsv        # all 18 features per gene
├── DatasetB/
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

A `predictions.tsv` file looks like this:

```
gene      rf_probability    rank
PUM2      0.9412            1
DDX3X     0.8830            2
RBFOX2    0.7621            3
...
```

That `rank` column is the answer to "what are my top candidate targets?"

---

## Useful prediction options

| Flag                       | Default | What it does                                                                                                                                                          |
| -------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--min-reads N`            | 20      | Drop editing sites with fewer than N total reads before feature computation. Raise this for very high-coverage libraries; lower it for sparse data.                   |
| `--min-edit-pct F`         | 0.0     | Drop sites whose edit% is below F (range 0–100). Useful for very noisy controls.                                                                                      |
| `--editing-type {AtoG,CtoT}` | AtoG  | A-to-G for ADAR-based TRIBE; C-to-T for APOBEC-based STAMP.                                                                                                          |
| `--background F [F ...]`   | none    | Learn a site-level background filter from gDNA / no-enzyme control file(s) and apply it before feature extraction. Also accepts per-dataset `background_files` column. |
| `--no-background-filter`   | off     | Disable background filtering even if one is bundled with the model.                                                                                                   |
| `--normalization {simple,median_of_ratios}` | simple | Replicate-normalisation strategy. `simple` is fine for most cases.                                                                                  |
| `--outdir DIR`             | `outputs/cross_test` | Where to put results. Will be created if missing.                                                                                                       |

Run `smartie-test --help` for the full list.

---

## Advanced: cross-dataset / LOO testing with multiple models

You can also use SMARTIE to compare several trained models against each other across many datasets. This is useful when you have trained dataset-specific models (one per held-out RBP, for example) and want to see how each generalises.

### Leave-one-out batch mode

If you trained a set of models with `smartie-train --metadata ...` (see next section), all models are saved under one directory as `Trained_<label>/rf_model.pkl`. To test every model against every dataset *except* the one it was trained on:

```bash
smartie-test \
    --metadata datasets.tsv \
    --loo-dir  outputs/training/ \
    --outdir   outputs/cross_test/
```

SMARTIE auto-discovers every `Trained_*/rf_model.pkl` and skips each model's own training dataset during evaluation.

### Explicit multi-model mode

```bash
smartie-test \
    --metadata datasets.tsv \
    --models   outputs/training/Trained_RBP1/rf_model.pkl \
               outputs/training/Trained_RBP2/rf_model.pkl \
               outputs/training/Trained_RBP3/rf_model.pkl \
    --outdir   outputs/cross_test/
```

### Extra plots produced in multi-model mode

In addition to all the single-model plots above, multi-model runs produce:

| File                                          | What it shows                                                              |
| --------------------------------------------- | -------------------------------------------------------------------------- |
| `multi_model_precision_heatmap.{pdf,png}`     | Models on rows, test datasets on columns, cell = precision @ 20 %.         |
| `multi_model_roc_heatmap.{pdf,png}`           | Same shape, cell = ROC-AUC.                                                |
| `multi_model_auc_comparison.{pdf,png}`        | Side-by-side AUC bars per model.                                           |
| `multi_model_feature_importance_heatmap.{pdf,png}` | Feature importance rows = models, columns = features.                  |
| `multi_model_avg_precision_heatmap.{pdf,png}` | Mean precision across test datasets per model.                             |
| `model_ranking_by_precision.{pdf,png}`        | Models ranked by their average performance.                                |
| `roc_pr_<dataset>.{pdf,png}`                  | One ROC + PR panel per test dataset, with all models overlaid.             |
| `precision_full_<dataset>.{pdf,png}`          | Full precision-vs-depth curve, all models overlaid, one panel per dataset. |

---

## Training your own model

If the bundled `SMARTIE.pkl` doesn't perform well on your RBP, or you want a model tuned specifically to your conditions, you can train your own. SMARTIE supports two training modes.

### Single-dataset training

```bash
smartie-train \
    --expt    data/MyRBP_rep1.txt data/MyRBP_rep2.txt \
    --ctrl    data/ADAR_rep1.txt  data/ADAR_rep2.txt \
    --targets data/known_targets.txt \
    --outdir  outputs/MyRBP_model/
```

This produces:

```
outputs/MyRBP_model/
├── rf_model.pkl              # the trained Random Forest
├── model_config.json         # all the hyperparameters/filters used at training
├── gene_features.tsv         # 18-feature matrix used for training
├── predictions.tsv           # predictions on the training set
├── feature_importance.tsv    # ranked importances
├── topk_metrics.tsv          # precision at each top-K threshold
└── plots/                    # per-model diagnostic plots (PDF + PNG)
    ├── roc_curve
    ├── precision_recall_curve
    ├── feature_importance
    ├── cv_fold_scores         # 5-fold CV AUC per fold + mean band
    ├── score_distribution
    └── topk_precision
```

Once trained, point `smartie-test` at the resulting `rf_model.pkl`:

```bash
smartie-test --metadata new_data.tsv --model outputs/MyRBP_model/rf_model.pkl --outdir results/
```

### Batch / leave-one-out training

If you have several labelled datasets and want one model per held-out dataset:

```bash
smartie-train \
    --metadata train_datasets.tsv \
    --outdir   outputs/loo_training/
```

This trains one model per row in the metadata file (each on its own dataset), saves them as `Trained_<label>/rf_model.pkl`, and also writes cross-model comparison plots in `outputs/loo_training/comparison/`:

- `feature_importance_heatmap` — features × models heatmap of MDI importance.
- `feature_rank_stability_heatmap` — which features rank consistently high across models.
- `cv_auc_comparison` — mean ± SD CV AUC bar chart, models sorted.
- `cv_score_distributions` — fold-level AUC distributions (boxplots + jitter).
- `topk_precision_comparison` — overlay of top-K precision curves.
- `model_similarity_heatmap` — Pearson r between models' feature importance vectors.
- `training_data_summary` — gene / target counts per model.
- `feature_consolidated_rank_barplot` — features ranked by average importance rank across models.

### Training options

| Flag                  | Default      | Purpose                                                                                            |
| --------------------- | ------------ | -------------------------------------------------------------------------------------------------- |
| `--min-reads`         | **20**       | Same site-level filter as in testing. Keep it consistent between train and test.                   |
| `--min-total-reads`   | 0            | Gene-level filter: drop genes with average total reads below this in the experiment replicates.    |
| `--min-fold-change`   | 0.0          | Gene-level filter: drop genes with `mean_cum_expt% / mean_cum_ctrl%` below this.                   |
| `--models`            | all          | Which classifiers to fit. Defaults to all 10 (rf, dt, nb, knn, logreg, linreg, svm, gb, xgb, lgbm). The RF is always the deliverable; the others are baselines. |
| `--drop-models`       | none         | Remove models from the selection (e.g. `--drop-models linreg svm`).                                |
| `--feature-weights`   | none         | Bias the RF toward particular features. `--feature-weights fold_change:3 site_enrichment:0.5`.     |
| `--test-fraction`     | 0.20         | Stratified held-out test split fraction.                                                           |
| `--test-bootstrap`    | 0            | Bootstrap N times for confidence intervals on every test metric.                                   |

Run `smartie-train --help` for the full list.

---

## Brute-force feature selection (advanced)

SMARTIE comes with an exhaustive feature-combination search pipeline (`smartie-brute`), useful when you want to find the smallest or best-performing subset of features for your data. It's separate from regular training because it's computationally expensive — overnight on a workstation, minutes on a small subset.

### What it does

Given the SMARTIE feature schema (9 **core** features + 16 **variable** features), the brute-force search:

1. Enumerates every combination of *k* variable features added to the 9 core features, for *k* in `[--k-min, --k-max]`.
2. For every combination and every training dataset in your training pool, trains a Random Forest.
3. Evaluates that model against every dataset in your **test** pool, recording precision @ {10, 20, …, 100} % of known targets.
4. Writes everything to a flat Parquet index that you can later query.

For *k* = 1 through 10 this is 58,650 combinations. With 4 workers it typically finishes overnight; results are resumable.

### Run a search

```bash
smartie-brute \
    --train-metadata metadata/train_datasets.tsv \
    --test-metadata  metadata/test_datasets.tsv \
    --trees       100 200 \
    --output-dir  results/brute_force/ \
    --workers     4 \
    --resume
```

Training and test pools must not overlap by dataset name — any test row whose `name` matches a training row is automatically dropped at load time.

### Output layout

```
results/brute_force/
├── results_index.parquet            # one row per (n_trees, combo) — query this
├── progress.json                    # ETA / completion tracker
├── 200/                             # n_trees value
│   └── 12/                          # n_features (= 9 core + 3 variable)
│       └── 8a1f3c4e2b9d/            # combo hash (first 12 hex chars)
│           ├── combo_metadata.json  # which features are in this combo
│           ├── combo_summary.json   # aggregated metrics
│           └── train_<dataset>/
│               ├── heatmap_data.csv
│               ├── feature_importance.csv
│               └── metrics.json
└── cache/                           # per-dataset feature matrix cache (Parquet)
```

### Query the results

The `smartie-brute-query` CLI filters the flat index by precision thresholds, feature presence, and importance:

```bash
# Best 20 combos at the 20% threshold with 200 trees
smartie-brute-query \
    --index results/brute_force/results_index.parquet \
    --precision-at 20 --min-precision 0.75 \
    --trees 200 --top 20

# Combos that contain fold_change and rank high by AUC
smartie-brute-query \
    --index results/brute_force/results_index.parquet \
    --must-contain fold_change \
    --sort-by mean_auc --top 50 \
    --output top50.csv

# List every known feature with its description
smartie-brute-query --index results/brute_force/results_index.parquet --list-features
```

### Plotting brute-force results

Three plotting scripts under `smartie.brute_force.plots`:

| Command                            | What it makes                                                                                                                                                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `smartie-brute-plot-kmax-best`     | For every value of *k*, picks the top-N combos by average test-only precision, draws one precision-heatmap PNG per combo (rows = test datasets, columns = depth thresholds), and writes a ranked CSV.                                    |
| `smartie-brute-plot-feature-rank`  | Rank-weighted feature-importance bar chart aggregated across all *k*. Core features are highlighted; variable features get lighter colour. Saves PDF, SVG, and PNG.                                                                      |
| `smartie-brute-plot-assembled`     | Publication-ready 4-panel figure: (A) precision heatmap rows = *k* / cols = threshold, (B) precision distribution violin per *k* with Kruskal-Wallis + Dunn-Bonferroni CLD, (C) rank-weighted feature ranking, (D) precision-vs-depth trend lines. |

Typical pipeline after a brute-force run:

```bash
# 1. Pull top combos per kmax
smartie-brute-plot-kmax-best \
    --parquet     results/brute_force/results_index.parquet \
    --results-dir results/brute_force/ \
    --outdir      figures/kmax_analysis/ \
    --top-n       20

# 2. Aggregate feature ranking across all kmax
smartie-brute-plot-feature-rank \
    --kmax-dir       figures/kmax_analysis/ \
    --core-features  fold_change fisher_odds_ratio binom_neg_log10_p \
                     site_enrichment norm_edits log_true_signal \
                     site_fraction avg_cum_expt_edit_pct site_edit_sd \
    --top-n          10 \
    --outdir         figures/feature_ranking/

# 3. Assemble the publication figure
smartie-brute-plot-assembled \
    --kmax-dir       figures/kmax_analysis/ \
    --core-features  fold_change fisher_odds_ratio binom_neg_log10_p \
                     site_enrichment norm_edits log_true_signal \
                     site_fraction avg_cum_expt_edit_pct site_edit_sd \
    --top-n          10 \
    --outdir         figures/assembled/
```

---

## Troubleshooting

**`command not found: smartie-test` after a pip install.** Your shell hasn't picked up the install location. Run `python -m pip show smartie | grep Location` and add the corresponding `bin/` directory to your `PATH`, or use `pipx install …` instead.

**`ImportError: No module named brute_force`.** You're running scripts directly from a checkout that has spaces in directory names. The package directory must be named `brute_force/` (underscore, no space). Use the install (`pip install -e .`) instead of running scripts by path.

**My test dataset has no known targets — can I still get predictions?** Yes. Provide a `targets_file` with one placeholder line; you'll still get `predictions.tsv` with the ranked genes, you just won't get precision plots for that dataset.

**Predictions look identical for every gene.** Almost always the feature schema in your model doesn't match the data: check that `model_config.json` (next to the `.pkl`) was loaded. If you copied a `.pkl` without its sibling JSON, the test command falls back to defaults that may differ.

**Memory blows up during the brute-force run.** Lower `--workers` and/or `--rf-threads`. The default fits a typical 16-core workstation; halve both if you have less.

---

## Citation

If SMARTIE is useful in your work, please cite *[paper or preprint placeholder]*.

## License

[Pick one — MIT / BSD-3-Clause / Apache-2.0 are all reasonable for biological tools — and put the full license text in `LICENSE`.]

## Contact

Issues and feature requests: <https://github.com/&lt;your-username&gt;/SMARTIE/issues>
