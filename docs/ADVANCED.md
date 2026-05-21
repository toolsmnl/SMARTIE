# SMARTIE: Advanced usage

This document covers advanced functionality intended for users who wish to go beyond the bundled pretrained model: training a custom model, performing exhaustive feature selection, and resolving common issues.

For installation and standard prediction workflows, see the main [`README.md`](../README.md). The pretrained model documented there is sufficient for most analyses.

---

## Table of contents

1. [Training a custom model](#1-training-a-custom-model)
2. [Brute-force feature selection](#2-brute-force-feature-selection)
3. [Troubleshooting](#3-troubleshooting)

---

## 1. Training a custom model

If the bundled `SMARTIE.pkl` model does not perform adequately on a particular RBP, or a model tuned to specific experimental conditions is desired, a custom model can be trained. SMARTIE supports both single-dataset and batch (leave-one-out) training.

### 1.1 Single-dataset training

```bash
smartie-train \
    --expt    data/MyRBP_rep1.txt data/MyRBP_rep2.txt \
    --ctrl    data/ADAR_rep1.txt  data/ADAR_rep2.txt \
    --targets data/known_targets.txt \
    --outdir  outputs/MyRBP_model/
```

This produces the following outputs:

```
outputs/MyRBP_model/
├── rf_model.pkl              # the trained model
├── model_config.json         # all hyperparameters and filters used at training
├── gene_features.tsv         # the feature matrix used for training
├── predictions.tsv           # predictions on the training set
├── feature_importance.tsv    # ranked feature importances
├── topk_metrics.tsv          # precision at each top-K threshold
└── plots/                    # per-model diagnostic plots (PDF and PNG)
```

Once trained, the resulting model can be supplied to `smartie-test`:

```bash
smartie-test --metadata new_data.tsv --model outputs/MyRBP_model/rf_model.pkl --outdir results/
```

### 1.2 Batch / leave-one-out training

When several labelled datasets are available and one model per held-out dataset is required:

```bash
smartie-train \
    --metadata train_datasets.tsv \
    --outdir   outputs/loo_training/
```

This trains one model per row in the metadata file, saves each as `Trained_<label>/rf_model.pkl`, and writes cross-model comparison plots to `outputs/loo_training/comparison/`.

### 1.3 Training options

| Flag | Default | Description |
| --- | --- | --- |
| `--min-reads` | 20  | Site-level coverage filter. Should be kept consistent between training and testing. |
| `--min-total-reads` | 0   | Gene-level filter: discards genes with average total reads below this value in the experiment replicates. |
| `--min-fold-change` | 0.0 | Gene-level filter: discards genes with `mean_cum_expt% / mean_cum_ctrl%` below this value. |
| `--models` | all | Classifiers to fit. Defaults to all ten. The Random Forest is always the deliverable; the others serve as baselines. |
| `--drop-models` | none | Removes specified models from the selection, e.g. `--drop-models linreg svm`. |
| `--feature-weights` | none | Biases the model toward particular features, e.g. `--feature-weights fold_change:3 site_enrichment:0.5`. |
| `--test-fraction` | 0.20 | Fraction of the data held out as a stratified test split. |
| `--test-bootstrap` | 0   | Number of bootstrap iterations for confidence intervals on test metrics. |

The complete list of options is available via `smartie-train --help`.

### 1.4 Cross-dataset and leave-one-out testing

To evaluate every trained model against every dataset except its own training set:

```bash
smartie-test \
    --metadata datasets.tsv \
    --loo-dir  outputs/loo_training/ \
    --outdir   outputs/cross_test/
```

Alternatively, models may be specified explicitly:

```bash
smartie-test \
    --metadata datasets.tsv \
    --models   outputs/loo_training/Trained_RBP1/rf_model.pkl \
               outputs/loo_training/Trained_RBP2/rf_model.pkl \
    --outdir   outputs/cross_test/
```

Multi-model runs generate additional comparison figures, including `multi_model_precision_heatmap`, `multi_model_roc_heatmap`, and `model_ranking_by_precision`.

---

## 2. Brute-force feature selection

SMARTIE includes an exhaustive feature-combination search, `smartie-brute`, for identifying the smallest or best-performing subset of features for a given dataset. This procedure is computationally intensive, typically requiring an overnight run on a workstation, and is therefore provided separately from standard training.

### 2.1 Method

Given the SMARTIE feature schema (9 **core** features and 16 **variable** features), the search enumerates every combination of *k* variable features added to the 9 core features, for *k* within `[--k-min, --k-max]`. For each combination, a model is trained on every training dataset and evaluated against every test dataset. All results are written to a queryable Parquet index.

### 2.2 Running a search

```bash
smartie-brute \
    --train-metadata metadata/train_datasets.tsv \
    --test-metadata  metadata/test_datasets.tsv \
    --trees          100 200 \
    --output-dir     results/brute_force/ \
    --workers        4 \
    --resume
```

The training and test pools must not overlap by dataset name; any test row whose `name` matches a training row is removed automatically at load time.

### 2.3 Querying the results

The `smartie-brute-query` command filters the index by precision thresholds, feature presence, and importance:

```bash
# Best 20 combinations at the 20% threshold, with 200 trees
smartie-brute-query \
    --index results/brute_force/results_index.parquet \
    --precision-at 20 --min-precision 0.75 \
    --trees 200 --top 20

# Combinations containing a given feature, ranked by AUC
smartie-brute-query \
    --index results/brute_force/results_index.parquet \
    --must-contain fold_change \
    --sort-by mean_auc --top 50 --output top50.csv

# List every known feature with its description
smartie-brute-query --index results/brute_force/results_index.parquet --list-features
```

### 2.4 Plotting the results

Three commands convert the search output into figures:

| Command | Output |
| --- | --- |
| `smartie-brute-plot-kmax-best` | For each value of *k*, selects the top-N combinations by average test precision and draws a precision heatmap per combination. |
| `smartie-brute-plot-feature-rank` | A rank-weighted feature-importance bar chart aggregated across all values of *k*. |
| `smartie-brute-plot-assembled` | A publication-ready four-panel figure combining the precision heatmap, per-*k* distributions, feature ranking, and depth trends. |

A representative post-processing pipeline:

```bash
smartie-brute-plot-kmax-best \
    --parquet     results/brute_force/results_index.parquet \
    --results-dir results/brute_force/ \
    --outdir      figures/kmax_analysis/ \
    --top-n       20

smartie-brute-plot-feature-rank \
    --kmax-dir       figures/kmax_analysis/ \
    --core-features  fold_change fisher_odds_ratio binom_neg_log10_p \
                     site_enrichment norm_edits log_true_signal \
                     site_fraction avg_cum_expt_edit_pct site_edit_sd \
    --top-n          10 \
    --outdir         figures/feature_ranking/

smartie-brute-plot-assembled \
    --kmax-dir       figures/kmax_analysis/ \
    --core-features  fold_change fisher_odds_ratio binom_neg_log10_p \
                     site_enrichment norm_edits log_true_signal \
                     site_fraction avg_cum_expt_edit_pct site_edit_sd \
    --top-n          10 \
    --outdir         figures/assembled/
```

---

## 3. Troubleshooting

**`command not found: smartie-test` (or `SMARTIE`) after installation.**
The shell has not registered the installation location. Run `python -m pip show smartie-mnl | grep Location` and add the corresponding `bin/` directory to the system `PATH`, or reinstall using `pipx install …`.

**The `SMARTIE` command is not recognized on Windows.**
Use the equivalent launch command: `python -m smartie_mnl.app.launcher`.

**`ImportError: No module named brute_force`.**
This occurs when scripts are run directly from a source checkout located in a directory containing spaces. Install the package properly (`pip install -e .`) rather than executing scripts by path.

**A test dataset has no known targets.**
Predictions can still be generated. Provide a `targets_file` containing a single placeholder entry; `predictions.tsv` with the ranked genes is still produced, although precision plots for that dataset are omitted.

**Predictions are identical for every gene.**
This almost always indicates a mismatch between the feature schema of the model and that of the data. Confirm that `model_config.json` (located alongside the `.pkl` file) was loaded. If a `.pkl` file is copied without its accompanying JSON, the test command reverts to default settings, which may differ.

**Memory usage is excessive during a brute-force run.**
Reduce `--workers` and/or `--rf-threads`. The default values are appropriate for a typical 16-core workstation; both should be halved on systems with fewer resources.
