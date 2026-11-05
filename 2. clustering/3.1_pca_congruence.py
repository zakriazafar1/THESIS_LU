"""
3.1_pca_congruence.py

Compares the within-person centered PCA with the original PCA.
  - between-participant share of variance per feature
 - Tucker's congruence coefficients (centered vs. original)
  - correlations between event scores of the two analyses
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
            r"\THESIS_OUTPUTS\PROJECT 2")

SCALED_FEATURES = BASE / r"reduced feature matrix\arousal_features_reduced_scaled.csv"
SCORES_ORIGINAL = BASE / r"4. clustering\3. pca\pca_scores_5pc_whitened.csv"
SCORES_CENTERED = BASE / r"5. clustering_centered\2. pca centered\pca_scores_5pc_whitened.csv"
LOADCORR_CENTERED = BASE / r"5. clustering_centered\2. pca centered\pca_loading_correlations.csv"

KEY = ["subject_id", "night_id", "event_idx"]
FEATURES = ["mean_delta_ratio", "mean_theta_ratio", "mean_alpha_ratio", "mean_sigma_ratio",
            "mean_beta_ratio", "duration_sec", "oxy_amp_ratio", "motion_rms"]
PCS = [f"PC{i}" for i in range(1, 6)]
pd.set_option("display.width", 200)


def read(path):
    return pd.read_csv(path, sep=None, engine="python")


feat = read(SCALED_FEATURES)
old = read(SCORES_ORIGINAL)
new = read(SCORES_CENTERED)

# 1. Share of each feature's variance that lies BETWEEN participants
subj_means = feat.groupby("subject_id")[FEATURES].transform("mean")
print("Between-participant share of variance (8% duration ... 26% beta):")
print((subj_means.var() / feat[FEATURES].var()).round(2), "\n")

# 2. Component-feature correlations of the original PCA
m = feat[KEY + FEATURES].merge(old[KEY + PCS], on=KEY, validate="one_to_one")
L_old = pd.DataFrame({p: [np.corrcoef(m[f], m[p])[0, 1] for f in FEATURES] for p in PCS},
                     index=FEATURES)
L_new = read(LOADCORR_CENTERED).set_index(read(LOADCORR_CENTERED).columns[0]).loc[FEATURES, PCS]
print("Original PCA, component-feature correlations:\n", L_old.round(2), "\n")
print("Centered PCA, component-feature correlations:\n", L_new.round(2), "\n")


# 3. Tucker's congruence
def tucker(a, b):
    return a @ b / np.sqrt((a @ a) * (b @ b))


T = pd.DataFrame({o: [tucker(L_old[o].values, L_new[n].values) for n in PCS] for o in PCS},
                 index=[f"centered {p}" for p in PCS])
print("Tucker's congruence (rows = centered, columns = original); diagonal |values|:")
print(T.round(2), "\n")

# 4. Correlation between event scores of corresponding components
s = old[KEY + PCS].merge(new[KEY + PCS], on=KEY, suffixes=("_orig", "_cent"))
print("|r| between scores of corresponding components (.83 PC4 ... .94 PC2):")
for p in PCS:
    print(f"  {p}: {abs(np.corrcoef(s[p + '_orig'], s[p + '_cent'])[0, 1]):.2f}")

# 5. Check that centering worked: participant means of the centered scores ~ 0
print("\nMax |participant mean| of centered scores:",
      new.groupby("subject_id")[PCS].mean().abs().max().max().round(4))