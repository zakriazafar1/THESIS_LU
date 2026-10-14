"""
=============================================================================
3.1_correlation.py

Correlation check on the scaled arousal feature matrix
(output of 2.2_scale_features.py).

Main analysis: Spearman on the scaled matrix
  Spearman is rank-based: robust to outliers and skewed distributions, and
  captures non-linear (but monotonic) associations. It is also invariant to
  any monotonic transformation (log, scaling), so the conclusion does not
  depend on the chosen transformation.

Check: Pearson on the same matrix
  Pearson measures linear association on exactly the values the clustering
  sees (after transformation + scaling). Pairs where Spearman and Pearson
  differ strongly point to outliers or non-linear (but monotonic)
  associations -> inspect visually.

Steps:
  1. Load the scaled feature matrix (arousal_feature_matrix_scaled.csv).
  2. Spearman correlation matrix (main) + Pearson correlation matrix (check).
  3. Heatmaps of both.
  4. Highly correlated pairs (|rho| > CORR_THRESHOLD, Spearman), with Pearson.
  5. Pairs where |rho - r| > DIFF_THRESHOLD (Spearman and Pearson diverge).
  6. CORRELATION GROUPS: features that all correlate |rho| >= GROUP_THRESHOLD
     with each other (hierarchical clustering on 1 - |rho|, complete linkage).
     Pick one feature per group. Per feature: mean |rho| with the rest of the
     group (how "central") and % missing -> helps with the choice.
     Complete linkage guarantees that EVERY pair within a group correlates
     >= the threshold (no chains A~B~C where A and C barely correlate).

Figures follow APA 7 style.

Usage:
  python 3.1_correlation.py
  python 3.1_correlation.py --input path/to/other_scaled.csv
=============================================================================
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled"
)

OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\3. feature selection\correlation"
)

DEFAULT_INPUT = INPUT_DIR / "arousal_feature_matrix_scaled.csv"

METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]

CORR_THRESHOLD = 0.70   # |rho| (Spearman) above this = highly correlated
DIFF_THRESHOLD = 0.15   # |rho - r| above this = Spearman and Pearson diverge

GROUP_METHOD = "spearman"   # correlation the groups are based on
GROUP_THRESHOLD = 0.70      # within a group: every pair |rho| >= this value

FIG_DPI = 300

# =============================================================================
# APA 7 FIGURE STYLE
# =============================================================================

TEXT_COLOR = "#000000"
AXIS_COLOR = "#000000"
CUTOFF_COLOR = "#C0392B"
ABOVE_CUTOFF_COLOR = "#8C8C8C"

GROUP_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4",
                 "#008300", "#eda100", "#e34948"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.edgecolor": AXIS_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.default": "it",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

STAT_SYMBOL = {"spearman": r"$\rho$", "pearson": r"$r$"}
STAT_NAME = {"spearman": r"Spearman's $\rho$", "pearson": r"Pearson's $r$"}

BAND_NAMES = {"delta": "Delta", "theta": "Theta", "alpha": "Alpha",
              "sigma": "Sigma", "beta": "Beta"}
SPECIAL_LABELS = {
    "duration_sec": "Duration",
    "oxy_amp_ratio": "PPG amplitude ratio",
    "motion_rms": "Head movement RMS",
}


def pretty_label(name: str) -> str:
    """
    Readable feature name for figures, e.g.
      L_delta_peak_ratio -> 'Delta peak ratio (L)'
      mean_alpha_ratio   -> 'Alpha ratio (mean)'
    Unknown names are returned with underscores replaced by spaces.
    """
    if name in SPECIAL_LABELS:
        return SPECIAL_LABELS[name]
    m = re.fullmatch(r"(L|R|mean)_(delta|theta|alpha|sigma|beta)_(peak_)?ratio", name)
    if m:
        side, band, peak = m.groups()
        kind = "peak ratio" if peak else "ratio"
        return f"{BAND_NAMES[band]} {kind} ({side})"
    return name.replace("_", " ")


def apa_corr(v: float) -> str:
    """APA format for correlations: two decimals, no leading zero (.85, −.32)."""
    if pd.isna(v):
        return ""
    s = f"{v:.2f}"
    s = s.replace("0.", ".", 1) if abs(v) < 1 else s
    return s.replace("-", "\u2212")


def save_figure(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# STEP 1: LOADING
# =============================================================================

def load_scaled_matrix(path: Path) -> pd.DataFrame:
    """
    Reads the scaled feature matrix. Same separator/decimal detection as in
    2.2_scale_features.py, in case the file was opened and saved with a
    Dutch Excel locale in between.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run 2.2_scale_features.py first, or pass the "
            "correct path with --input."
        )
    df = pd.read_csv(path, sep=None, engine="python")

    if df.shape[1] == 1:
        raise ValueError(
            f"{path} appears to have only 1 column ({df.columns[0]!r}) -- "
            "the delimiter could not be detected, or the file is corrupted."
        )

    non_numeric_id_cols = {"subject_id", "group", "night_id"}
    check_cols = [c for c in df.columns if c not in non_numeric_id_cols]
    n_non_numeric = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in check_cols)

    if n_non_numeric > 0:
        df_comma_decimal = pd.read_csv(path, sep=None, engine="python", decimal=",")
        n_non_numeric_comma = sum(
            not pd.api.types.is_numeric_dtype(df_comma_decimal[c])
            for c in check_cols if c in df_comma_decimal.columns
        )
        if n_non_numeric_comma < n_non_numeric:
            print(f"[NOTE] {n_non_numeric} column(s) were read as text -- "
                  f"decimal comma detected, re-read with decimal=','.")
            df = df_comma_decimal

    print(f"Scaled feature matrix loaded: {path}")
    print(f"Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the metadata columns -> these are the features to check."""
    return [c for c in df.columns if c not in METADATA_COLS]


# =============================================================================
# STEP 2: COMPUTING CORRELATIONS
# =============================================================================

def compute_corr(df: pd.DataFrame, feature_cols: list[str], method: str) -> pd.DataFrame:
    """Correlation matrix ('pearson' or 'spearman'), pairwise, NaN/inf treated as missing."""
    sub = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return sub.corr(method=method)


def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path, method: str) -> None:
    """
    APA-style heatmap: lower triangle only (the matrix is symmetric and the
    diagonal is always 1), readable feature names, diverging colour scale
    centred on 0, annotated (APA format) when the matrix is small enough.
    """
    n = len(corr)
    labels = [pretty_label(c) for c in corr.columns]
    data = corr.copy()
    data.index, data.columns = labels, labels
    # lower triangle without diagonal -> drop first row / last column (would be empty)
    mask = np.triu(np.ones_like(data, dtype=bool))
    data, mask = data.iloc[1:, :-1], mask[1:, :-1]

    annot = n <= 20
    annot_text = data.map(apa_corr) if annot else False

    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.45 * n + 2))
    sns.heatmap(
        data,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        annot=annot_text,
        fmt="",
        annot_kws={"size": 7},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": STAT_NAME[method], "shrink": 0.7,
                  "ticks": [-1, -0.5, 0, 0.5, 1]},
        ax=ax,
    )
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(["\u22121", "\u2212.50", "0", ".50", "1"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=90, length=0)
    ax.tick_params(axis="y", labelsize=8, rotation=0, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_path)
    print(f"Heatmap saved: {out_path} (+ .pdf)")


def upper_triangle_pairs(spearman: pd.DataFrame, pearson: pd.DataFrame) -> pd.DataFrame:
    """All unique feature pairs with Spearman rho (main), Pearson r (check) and their difference."""
    cols = spearman.columns
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = pearson.iloc[i, j]
            rho = spearman.iloc[i, j]
            rows.append({
                "feature_1": cols[i],
                "feature_2": cols[j],
                "spearman_rho": rho,
                "pearson_r": r,
                "diff_rho_minus_r": rho - r if pd.notna(r) and pd.notna(rho) else np.nan,
            })
    return pd.DataFrame(rows)


def list_high_corr_pairs(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Pairs with |Spearman rho| > threshold, sorted from high to low."""
    out = pairs[pairs["spearman_rho"].abs() > threshold].copy()
    out = out.reindex(out["spearman_rho"].abs().sort_values(ascending=False).index)
    return out.round(4).reset_index(drop=True)


def list_divergent_pairs(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Pairs where Spearman and Pearson differ by more than `threshold`."""
    out = pairs[pairs["diff_rho_minus_r"].abs() > threshold].copy()
    out = out.reindex(out["diff_rho_minus_r"].abs().sort_values(ascending=False).index)
    return out.round(4).reset_index(drop=True)


# =============================================================================
# STEP 3: IDENTIFYING CORRELATION GROUPS
# =============================================================================

def find_corr_groups(corr: pd.DataFrame, threshold: float) -> pd.Series:
    """
    Hierarchical clustering on distance 1 - |r|, complete linkage, cut at
    1 - threshold. Result: group number per feature. Within a group every
    pair has |r| >= threshold.
    """
    dist = 1 - corr.abs().fillna(0).to_numpy()
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2                      # make numerically symmetric
    Z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(Z, t=1 - threshold, criterion="distance")
    return pd.Series(labels, index=corr.columns, name="group"), Z


def summarize_groups(corr: pd.DataFrame, groups: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per feature: group, group size, mean and minimum |rho| with the
    other group members, and % missing. Groups numbered 1..k by size.
    'suggested' = the most central member (highest mean |rho|) -- only a
    starting point, choose on content.
    """
    # renumber: largest group = 1
    sizes = groups.value_counts()
    order = sorted(sizes.index, key=lambda g: (-sizes[g], groups[groups == g].index[0]))
    renum = {old: new for new, old in enumerate(order, start=1)}
    groups = groups.map(renum)

    rows = []
    for g, members in groups.groupby(groups):
        feats = list(members.index)
        for f in feats:
            others = [o for o in feats if o != f]
            a = corr.loc[f, others].abs() if others else pd.Series(dtype=float)
            rows.append({
                "group": g,
                "group_size": len(feats),
                "feature": f,
                f"mean_abs_{GROUP_METHOD}_in_group": round(a.mean(), 3) if others else np.nan,
                f"min_abs_{GROUP_METHOD}_in_group": round(a.min(), 3) if others else np.nan,
                "pct_missing": round(100 * df[f].isna().mean(), 2),
            })
    out = pd.DataFrame(rows)
    key = f"mean_abs_{GROUP_METHOD}_in_group"
    out = out.sort_values(["group", key], ascending=[True, False], na_position="last")
    out["suggested"] = False
    multi = out["group_size"] > 1
    out.loc[out[multi].groupby("group")[key].idxmax(), "suggested"] = True
    return out.reset_index(drop=True)


def print_groups(summary: pd.DataFrame, threshold: float) -> None:
    key = f"mean_abs_{GROUP_METHOD}_in_group"
    multi = summary[summary["group_size"] > 1]
    single = summary[summary["group_size"] == 1]["feature"].tolist()
    n_groups = multi["group"].nunique()
    print(f"\nStep 6: {n_groups} correlation group(s) (every pair |{GROUP_METHOD}| >= {threshold}); "
          "pick one feature per group:")
    for g, sub in multi.groupby("group"):
        print(f"\n  Group {g} ({len(sub)} features, min |rho| in group = "
              f"{sub[f'min_abs_{GROUP_METHOD}_in_group'].min():.2f}):")
        for _, r in sub.iterrows():
            flag = "  <- most central" if r["suggested"] else ""
            print(f"    {r['feature']:<22} mean |rho| = {r[key]:.2f}   "
                  f"missing = {r['pct_missing']:.1f}%{flag}")
    print(f"\n  Stand-alone features (no partner >= {threshold}), keep as is: "
          f"{single if single else '(none)'}")


def plot_dendrogram(Z, labels: list[str], threshold: float, out_path: Path) -> None:
    """
    APA-style dendrogram: groups below the cut-off in colour, links above the
    cut-off in grey, dashed cut-off line, readable feature names.
    """
    from scipy.cluster import hierarchy
    hierarchy.set_link_color_palette(GROUP_PALETTE)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(labels)), 5))
    dendrogram(Z, labels=[pretty_label(l) for l in labels],
               color_threshold=1 - threshold, above_threshold_color=ABOVE_CUTOFF_COLOR,
               leaf_rotation=90, leaf_font_size=8, ax=ax)
    hierarchy.set_link_color_palette(None)
    for line in ax.collections:
        line.set_linewidth(1.4)

    sym = STAT_SYMBOL[GROUP_METHOD]
    ax.axhline(1 - threshold, color=CUTOFF_COLOR, ls="--", lw=1,
               label=f"Cut-off: |{sym}| = {apa_corr(threshold)}")
    ax.set_ylabel(f"Distance (1 \u2212 |{sym}|)")
    ax.set_xlabel("Feature")
    ax.set_ylim(0, 1.02)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    print(f"Dendrogram saved: {out_path} (+ .pdf)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to arousal_feature_matrix_scaled.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_scaled_matrix(args.input)
    feature_cols = get_feature_columns(df)
    print(f"\n{len(feature_cols)} features (metadata columns excluded): {feature_cols}")

    # --- Steps 2+3: correlation matrices + heatmaps ---
    spearman = compute_corr(df, feature_cols, "spearman")
    pearson = compute_corr(df, feature_cols, "pearson")
    spearman.to_csv(out_dir / "spearman_corr_matrix.csv")
    pearson.to_csv(out_dir / "pearson_corr_matrix.csv")
    plot_corr_heatmap(spearman, out_dir / "spearman_corr_heatmap.png", "spearman")
    plot_corr_heatmap(pearson, out_dir / "pearson_corr_heatmap.png", "pearson")

    pairs = upper_triangle_pairs(spearman, pearson)

    # --- Step 4: highly correlated pairs (Spearman) ---
    high_corr = list_high_corr_pairs(pairs, CORR_THRESHOLD)
    high_corr.to_csv(out_dir / "high_corr_pairs.csv", index=False)
    print(f"\nStep 4: {len(high_corr)} pair(s) with |rho| > {CORR_THRESHOLD} (Spearman):")
    print(high_corr.to_string(index=False) if not high_corr.empty else "  (none)")

    # --- Step 5: Spearman vs Pearson diverge ---
    divergent = list_divergent_pairs(pairs, DIFF_THRESHOLD)
    divergent.to_csv(out_dir / "pearson_vs_spearman_divergent.csv", index=False)
    print(f"\nStep 5: {len(divergent)} pair(s) with |rho - r| > {DIFF_THRESHOLD}:")
    print(divergent.to_string(index=False) if not divergent.empty else "  (none)")

    # --- Step 6: correlation groups ---
    group_corr = spearman if GROUP_METHOD == "spearman" else pearson
    groups, Z = find_corr_groups(group_corr, GROUP_THRESHOLD)
    summary = summarize_groups(group_corr, groups, df)
    summary.to_csv(out_dir / "corr_groups.csv", index=False)
    print_groups(summary, GROUP_THRESHOLD)
    plot_dendrogram(Z, list(group_corr.columns), GROUP_THRESHOLD,
                    out_dir / "corr_groups_dendrogram.png")

    print(f"\nAll output saved in: {out_dir}")
    print("  - spearman_corr_matrix.csv / spearman_corr_heatmap.png(.pdf) <- main analysis")
    print("  - pearson_corr_matrix.csv / pearson_corr_heatmap.png(.pdf)   <- check")
    print("  - high_corr_pairs.csv                <- redundant pairs (Spearman, with r)")
    print("  - pearson_vs_spearman_divergent.csv  <- pairs to inspect visually (scatterplot)")
    print("  - corr_groups.csv / corr_groups_dendrogram.png(.pdf) <- pick 1 feature per group")


if __name__ == "__main__":
    main()