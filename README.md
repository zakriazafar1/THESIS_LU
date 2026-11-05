# Do Sleep Arousals Form Distinct Subtypes?
**An Exploratory Clustering Study in Adults with Insomnia Symptoms**

## Overview

This repository contains the analysis pipeline used to test whether expert-scored sleep arousals form distinct subtypes. For each arousal event, spectral, temporal, motion and PPG features are extracted from the raw recordings. The features are transformed, standardized and reduced, after which clustering tendency is assessed (Hopkins statistic) and the data are clustered with Gaussian mixture models and HDBSCAN.

## Repository structure

Scripts are numbered in the order in which they are run.

### `1. features/` – feature extraction and preprocessing

| Script | Description |
|---|---|
| `0_check_edf.py` | Prints the unit and range of each channel in an EDF file (sanity check) |
| `1_feature_matrix.py` | Builds the feature matrix (one row per scored arousal) from the expert scorings and the raw EDF channels |
| `1.1_clip_nights.py` | Recomputes features for a few nights whose final part is artifact, using only the clean part of the recording |
| `1.2_clean_feature_matrix.py` | Creates the final analysis matrix: keeps arousals of 3–15 s that occur during sleep |
| `1.3_feature_summary.py` | Summary table of all features |
| `1.4_visualise_features.py` | Distribution plots of all features |
| `2.1_transform_features.py` | Natural-log transformation of skewed features |
| `2.2_scale_features.py` | Standardization (z-scores) |
| `3.1_correlation.py` | Correlation analysis (Spearman, with Pearson as check) used for feature reduction |
| `3.2_reduce_features.py` | Reduces the matrix to the 8 selected features |
| `4_center_features.py` | Centers the scaled features within participants and re-standardizes them |

### `2. clustering/` – clustering analyses

| Script | Description |
|---|---|
| `1_hopkins.py` | Hopkins statistic to assess clustering tendency |
| `2_GMM.py` | Full-covariance Gaussian mixture models for K = 1 to K_max |
| `3_pca.py` | PCA on the reduced, scaled features (whitened and unwhitened scores) |
| `3.1_pca_congruence.py` | Compares the within-person centered PCA with the original PCA |
| `4_hdbscan.py` | HDBSCAN with a parameter grid, ranked by relative validity and checked for stability |
| `4.1_hdbscan_descriptives.py` | Descriptive statistics of the HDBSCAN clusters on the original features |

### `3. descriptives/` – sample description

| Script | Description |
|---|---|
| `nights_overview.py` | Description of the included nights (e.g., arousal index per hour of sleep) |
| `subjects_overview.py` | Overview of participants, nights and events |

## Usage

Requires Python 3 (developed with Python 3.14). Install the dependencies with:

```bash
pip install -r requirements.txt
```

Input and output locations are defined at the top of each script (e.g., `BASE`, `INPUT`, `OUTPUT_DIR`) and point to local folders. Adjust these to your own data location before running. Most scripts also accept `--input` and `--output-dir` arguments; run a script with `--help` to see its options.

## Notes

Some comments and docstrings are written in Dutch.
