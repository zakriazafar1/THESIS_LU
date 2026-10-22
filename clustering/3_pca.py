"""
=============================================================================
3_pca.py

PCA on the reduced, scaled arousal feature matrix
(output of 3.3_reduce_features.py).

Aim: remove the correlations between the features (especially the spectral
block) before HDBSCAN.
  - With --whiten every retained PC gets variance 1, so every independent
    dimension counts equally in the distances. Only defensible once the noise 
    PCs with very small eigenvalues have been dropped (--n-components), otherwise 
    those will be inflated.

What the script does:
  1. All PCs (including those not retained): eigenvalue, % explained
     variance, cumulative % > explained_variance.csv
  2. Loadings of all PCs:
       - pca_loadings.csv              (eigenvectors, weight per feature)
       - pca_loading_correlations.csv   (correlation feature <-> PC = loading *
         sqrt(eigenvalue) + heatmap of the correlations.
     Features are always shown in FEATURE_ORDER (5 spectral bands, PPG,
     head movement, duration), in the CSVs as well as in the figures.
  3. Cumulative explained variance plot, the chosen number of PCs marked.
  4. Check: correlation matrix of the PC scores (all PCs).
       - Pearson: should be ~0 off the diagonal by construction. Deviations 
         mean something is wrong.
       - Spearman: does NOT have to be 0. A Spearman corr of not 0 shows that 
         PCs still share monotonic, non-linear association (outliers/skew)
  5. PC scores of the retained components (+ metadata), with or without
     whitening, as CSV -> input for HDBSCAN.

Usage:
  python 3_pca.py                 # DEFAULT PCs, without and with whitening
  python 3_pca.py --whiten        # only without whitening
  python 3_pca.py --whiten yes            # only with whitening
  python 3_pca.py --n-components 0.90     # as many PCs as needed for 90%
  python 3_pca.py --n-components 6        # fixed number of PCs 
=============================================================================
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.decomposition import PCA

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2"
)
DEFAULT_INPUT = BASE / r"reduced feature matrix\arousal_features_reduced_scaled.csv"
OUTPUT_DIR = BASE / r"4. clustering\3. pca"

METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]

# Order of the features in all tables and figures
FEATURE_ORDER = [
    "mean_delta_ratio", "mean_theta_ratio", "mean_alpha_ratio",
    "mean_sigma_ratio", "mean_beta_ratio",
    "oxy_amp_ratio",
    "motion_rms",
    "duration_sec",
]

N_COMPONENTS_DEFAULT = 0.90  # int = number of PCs, float < 1 = target cumulative variance
VARIANCE_REF = 90            # % reference line in the scree plot
RANDOM_STATE = 2554542

FIG_DPI = 300

# =============================================================================
# APA 7 FIGURE STYLE
# =============================================================================

TEXT_COLOR = "#000000"
BAR_COLOR = "#2a78d6"          # retained components
BAR_COLOR_DROPPED = "#C9D9EF"  # components not retained
LINE_COLOR = "#1F3F66"
REF_COLOR = "#7F7F7F"
KEEP_COLOR = "#C0392B"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.edgecolor": TEXT_COLOR,
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

SIDE_NAMES = {"mean": "Mean", "L": "Left", "R": "Right"}
SPECIAL_LABELS = {
    "duration_sec": "Duration",
    "oxy_amp_ratio": "PPG ratio",
    "motion_rms": "Head movement RMS",
}


def pretty_label(name: str) -> str:
    """
    Readable feature name for figures
    """
    if name in SPECIAL_LABELS:
        return SPECIAL_LABELS[name]
    m = re.fullmatch(r"(L|R|mean)_(delta|theta|alpha|sigma|beta)_(peak_)?ratio", name)
    if m:
        side, band, peak = m.groups()
        kind = "peak ratio" if peak else "ratio"
        return f"{SIDE_NAMES[side]} {band} {kind}"
    return name.replace("_", " ")


def order_features(features: list[str]) -> list[str]:
    """Features in FEATURE_ORDER first; unknown features appended at the end."""
    known = [f for f in FEATURE_ORDER if f in features]
    rest = [f for f in features if f not in FEATURE_ORDER]
    if rest:
        print(f"[NOTE] features not in FEATURE_ORDER, placed at the end: {rest}")
    return known + rest


def apa_corr(v: float) -> str:
    """APA format for correlations."""
    if pd.isna(v):
        return ""
    v = 0.0 if abs(v) < 0.005 else v          
    s = f"{v:.2f}"
    s = s.replace("0.", ".", 1) if abs(v) < 1 else s
    return s.replace("-", "\u2212")


def save_figure(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def panel_letter(ax, letter: str, x: float = -0.12) -> None:
    ax.text(x, 1.04, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", color=TEXT_COLOR)


# =============================================================================
# STEP 1: LOADING
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist -- run 3.3_reduce_features.py first.")
    df = pd.read_csv(path, sep=None, engine="python")
    text_cols = {"subject_id", "group", "night_id"}
    check = [c for c in df.columns if c not in text_cols]
    n_bad = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in check)
    if n_bad > 0:
        df2 = pd.read_csv(path, sep=None, engine="python", decimal=",")
        n_bad2 = sum(not pd.api.types.is_numeric_dtype(df2[c]) for c in check if c in df2.columns)
        if n_bad2 < n_bad:
            print("[NOTE] decimal comma detected, re-read with decimal=','.")
            df = df2
    print(f"Loaded: {path}  shape={df.shape}")
    return df


def parse_n_components(value: str, n_features: int) -> int | float:
    """'5' -> 5 PCs; '0.90' -> target cumulative variance."""
    v = float(value)
    if 0 < v < 1:
        return v
    k = int(v)
    if not 1 <= k <= n_features:
        raise ValueError(f"--n-components must be between 1 and {n_features} (or 0-1 as a fraction).")
    return k


# =============================================================================
# STEP 2: PCA
# =============================================================================

def fit_full_pca(X: pd.DataFrame) -> tuple[PCA, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    PCA with all components. Returns the model + tables for explained variance,
    loadings (eigenvectors) and loading correlations. The signs of PCs are arbitrary, 
    they are set so that the largest absolute loading per PC is positive (for readability).
    """
    pca = PCA(random_state=RANDOM_STATE).fit(X)

    # fix signs
    for i in range(pca.n_components_):
        if pca.components_[i, np.argmax(np.abs(pca.components_[i]))] < 0:
            pca.components_[i] *= -1

    pcs = [f"PC{i + 1}" for i in range(pca.n_components_)]
    ev = pd.DataFrame({
        "component": pcs,
        "eigenvalue": pca.explained_variance_,
        "pct_variance": 100 * pca.explained_variance_ratio_,
        "cum_pct_variance": 100 * np.cumsum(pca.explained_variance_ratio_),
    }).round(4)

    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=pcs).round(4)
    # correlation feature <-> PC = eigenvector * sqrt(eigenvalue) / SD(feature)
    sd = X.std(ddof=1).to_numpy()[:, None]
    corr = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_)[None, :] / sd,
                        index=X.columns, columns=pcs).round(4)
    return pca, ev, loadings, corr


def resolve_k(ev: pd.DataFrame, n_components: int | float) -> int:
    """Number of PCs to retain; for a fraction: the smallest k that reaches that variance."""
    if isinstance(n_components, float):
        return int(np.searchsorted(ev["cum_pct_variance"].to_numpy(), 100 * n_components - 1e-9) + 1)
    return n_components


# =============================================================================
# STEP 3: PLOTS
# =============================================================================

def plot_scree(ev: pd.DataFrame, k: int, out_path: Path) -> None:
    """
    Cumulative explained variance per number of PCs.
    """
    x = np.arange(1, len(ev) + 1)
    xlabels = [str(i) for i in x]
    fig, ax2 = plt.subplots(figsize=(6, 4.2))

    cum = ev["cum_pct_variance"].to_numpy()
    ax2.plot(x, cum, color=LINE_COLOR, lw=1.8, marker="o", ms=5, zorder=3)
    for lvl in (VARIANCE_REF,):
        ax2.axhline(lvl, color=REF_COLOR, ls=":", lw=1, zorder=1)
        ax2.text(len(ev) + 0.45, lvl, f"{lvl}%", va="center", ha="left", fontsize=8,
                 color=REF_COLOR)
    ax2.scatter([k], [cum[k - 1]], s=110, facecolor="none", edgecolor=KEEP_COLOR, lw=1.8, zorder=4)
    ax2.annotate(f"{k} components: {cum[k - 1]:.1f}%", (k, cum[k - 1]),
                 textcoords="offset points", xytext=(12, -42), fontsize=9, color=TEXT_COLOR,
                 bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"),
                 arrowprops=dict(arrowstyle="-", color=KEEP_COLOR, lw=0.8, shrinkA=0, shrinkB=6))
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels)
    ax2.set_xlim(0.5, len(ev) + 0.5)
    ax2.set_ylim(0, 102)
    ax2.set_xlabel("Number of principal components")
    ax2.set_ylabel("Cumulative explained variance (%)")

    fig.tight_layout()
    save_figure(fig, out_path)
    print(f"Scree plot saved: {out_path} (+ .pdf)")


def plot_loadings(corr: pd.DataFrame, ev: pd.DataFrame, k: int, out_path: Path) -> None:
    """
    Heatmap of the feature-PC correlations for all PCs. The retained PCs are
    separated from the others by a red line and labelled above the heatmap.
    Row order follows the column order of corr (= FEATURE_ORDER).
    """
    data = corr.copy()
    data.index = [pretty_label(c) for c in data.index]
    xlabels = [f"{c}\n{p:.1f}%" for c, p in zip(ev["component"], ev["pct_variance"])]
    annot = data.map(apa_corr)

    fig, ax = plt.subplots(figsize=(0.95 * corr.shape[1] + 3, 0.5 * corr.shape[0] + 2))
    sns.heatmap(data, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=annot, fmt="",
                annot_kws={"size": 8}, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Correlation with component ($r$)", "shrink": 0.8,
                          "ticks": [-1, -0.5, 0, 0.5, 1]}, ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(["\u22121", "\u2212.50", "0", ".50", "1"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticklabels(xlabels, rotation=0, fontsize=8)
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    ax.set_xlabel("Principal components (with % explained variance)")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)

    n_pc = corr.shape[1]
    if k < n_pc:
        ax.axvline(k, color=KEEP_COLOR, lw=2)
        ax.text(k / 2, -0.25, "Retained", ha="center", va="bottom", fontsize=9,
                color=TEXT_COLOR, transform=ax.transData)
        ax.text(k + (n_pc - k) / 2, -0.25, "Not retained", ha="center", va="bottom",
                fontsize=9, color=REF_COLOR, transform=ax.transData)
    fig.tight_layout()
    save_figure(fig, out_path)
    print(f"Loadings heatmap saved: {out_path} (+ .pdf)")


# =============================================================================
# STEP 4: CORRELATION BETWEEN THE PCs
# =============================================================================

def pc_correlations(X: pd.DataFrame, full: PCA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pearson and Spearman correlation matrix of the scores of ALL PCs.
    Unwhitened scores are used.
    """
    scores = compute_scores(X, full, full.n_components_, whiten=False)
    return scores.corr(method="pearson"), scores.corr(method="spearman")


def max_offdiag(c: pd.DataFrame) -> tuple[float, str]:
    """Largest |r| off the diagonal + which pair."""
    a = c.abs().to_numpy().copy()
    np.fill_diagonal(a, 0)
    i, j = np.unravel_index(np.argmax(a), a.shape)
    return float(a[i, j]), f"{c.index[i]} - {c.columns[j]}"


def plot_pc_correlations(pear: pd.DataFrame, spear: pd.DataFrame, k: int, out_path: Path) -> None:
    """
    Two heatmaps side by side: A = Pearson (should be ~0), B = Spearman.
    """
    n = len(pear)
    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.75 * n + 2.5), 0.65 * n + 2))
    cbar_labels = ("Pearson's $r$", "Spearman's $\\rho$")
    for ax, c, lab, letter in zip(axes, (pear, spear), cbar_labels, "AB"):
        m = np.triu(np.ones_like(c, dtype=bool), k=1)   # lower triangle + diagonal
        data = c
        sns.heatmap(data, mask=m, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                    annot=data.map(apa_corr), fmt="", annot_kws={"size": 8}, square=True,
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.75, "label": lab, "ticks": [-1, -0.5, 0, 0.5, 1]},
                    ax=ax)
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(["\u22121", "\u2212.50", "0", ".50", "1"])
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=8)
        ax.tick_params(axis="both", length=0, labelsize=9)
        ax.tick_params(axis="y", rotation=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for s in ax.spines.values():
            s.set_visible(False)
        # boundary after the last retained PC, on both axes, across the whole plot
        if k < n:
            ax.axhline(k, color=KEEP_COLOR, lw=1.5)
            ax.axvline(k, color=KEEP_COLOR, lw=1.5)
        panel_letter(ax, letter, x=-0.08)
    handles = [Line2D([], [], color=KEEP_COLOR, lw=1.5,
                      label=f"Boundary of retained components ($k$ = {k})")]
    fig.legend(handles=handles, loc="lower center", frameon=False, ncol=1,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, out_path)
    print(f"PC correlation heatmap saved: {out_path} (+ .pdf)")


# =============================================================================
# STEP 5: SCORES (with / without whitening)
# =============================================================================

def compute_scores(X: pd.DataFrame, full: PCA, k: int, whiten: bool) -> pd.DataFrame:
    """
    PC scores of the first k components. Uses the same (sign-corrected)
    eigenvectors as the full PCA, so scores and loadings belong together.
    whiten=True: scores divided by sqrt(eigenvalue) -> variance 1 per PC.
    """
    centered = X.to_numpy() - full.mean_
    scores = centered @ full.components_[:k].T
    if whiten:
        scores = scores / np.sqrt(full.explained_variance_[:k])
    return pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(k)], index=X.index)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-components", default=N_COMPONENTS_DEFAULT,
                        help="Number of PCs (e.g. 5) or target fraction of variance (e.g. 0.90)")
    parser.add_argument("--whiten", choices=["no", "yes", "both"], default="both",
                        help="Scores without whitening, with, or both (default)")
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_robust(args.input)
    meta = [c for c in METADATA_COLS if c in df.columns]
    features = order_features([c for c in df.columns if c not in METADATA_COLS])
    X = df[features].replace([np.inf, -np.inf], np.nan)
    n_nan = int(X.isna().any(axis=1).sum())
    if n_nan:
        print(f"[NOTE] {n_nan} row(s) with NaN/inf excluded from the PCA.")
        keep = ~X.isna().any(axis=1)
        X, df = X[keep], df[keep]
    print(f"{len(features)} features: {features}")

    # Full PCA 
    full, ev, loadings, corr = fit_full_pca(X)
    k = resolve_k(ev, parse_n_components(args.n_components, len(features)))

    ev.to_csv(out_dir / "explained_variance.csv", index=False)
    loadings.to_csv(out_dir / "pca_loadings.csv")
    corr.to_csv(out_dir / "pca_loading_correlations.csv")

    print("\nAll components:")
    print(ev.to_string(index=False))
    print(f"\nChosen: {k} PCs = {ev['cum_pct_variance'].iloc[k - 1]:.1f}% of variance.")
    print("\nLoadings as correlation feature <-> PC (all PCs):")
    print(corr.round(2).to_string())

    # Communalities of the retained PCs 
    h2 = (corr.iloc[:, :k] ** 2).sum(axis=1)
    print(f"\nCommunalities h2 over the {k} retained PCs:")
    print(h2.rename(index=pretty_label).round(2).to_string())

    # Plots 
    plot_scree(ev, k, out_dir / "pca_screeplot.png")
    plot_loadings(corr, ev, k, out_dir / "pca_loadings_heatmap.png")

    # Check correlation between PCs 
    pear, spear = pc_correlations(X, full)
    pear.to_csv(out_dir / "pc_correlation_pearson.csv")
    spear.to_csv(out_dir / "pc_correlation_spearman.csv")
    plot_pc_correlations(pear, spear, k, out_dir / "pc_correlation_heatmap.png")

    p_max, p_pair = max_offdiag(pear)
    s_max, s_pair = max_offdiag(spear)
    print("\nCheck: correlation between PC scores (off the diagonal)")
    print(f"  Pearson  max |r|   = {p_max:.2e}  ({p_pair})  "
          f"-> {'OK, uncorrelated' if p_max < 1e-8 else '[NOTE] not ~0, check the input'}")
    print(f"  Spearman max |rho| = {s_max:.3f}  ({s_pair})  "
          "-> non-linear/monotonic association that PCA does not remove")
    print("\nSpearman rho between PCs:")
    print(spear.round(2).to_string())

    # PC scores
    modes = {"no": [False], "yes": [True], "both": [False, True]}[args.whiten]
    print()
    for w in modes:
        scores = compute_scores(X, full, k, w)
        out = pd.concat([df[meta].reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        name = f"pca_scores_{k}pc_{'whitened' if w else 'unwhitened'}.csv"
        out.to_csv(out_dir / name, index=False)
        var = scores.var(ddof=1).round(2).to_dict()
        print(f"Scores {'WITH' if w else 'WITHOUT'} whitening -> {name}")
        print(f"  variance per PC: {var}")

    print(f"\nAll output saved in: {out_dir}")
    print("  - explained_variance.csv               <- eigenvalue / % / cumulative % per PC")
    print("  - pca_loadings.csv                     <- eigenvectors")
    print("  - pca_loading_correlations.csv         <- correlation feature <-> PC (interpretation)")
    print("  - pc_correlation_pearson.csv / _spearman.csv / pc_correlation_heatmap.png(.pdf)")
    print("                                         <- check: are the PCs uncorrelated?")
    print("  - pca_screeplot.png(.pdf) / pca_loadings_heatmap.png(.pdf)")
    print("  - pca_scores_*pc_(un)whitened.csv      <- input for HDBSCAN")


if __name__ == "__main__":
    main()