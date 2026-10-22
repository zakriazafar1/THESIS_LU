"""
=============================================================================
4_hdbscan.py

HDBSCAN on three inputs, run automatically one after the other:
  1. whitened         -- pca_scores_<k>pc_whitened.csv   
  2. unwhitened       -- pca_scores_<k>pc_unwhitened.csv 
  3. reduced_features -- arousal_features_reduced_scaled.csv 

Each run writes its output to its own subfolder OUTPUT_DIR/<tag>.
With --allow-single-cluster all three runs are done with
allow_single_cluster=True and saved under <tag>_single.

Features are detected automatically: if there are columns starting with "PC",
those are the features, otherwise all numeric non-metadata columns.

Parameters:
  min_cluster_size  -- the smallest group you would still call a "subtype".
  min_samples       -- how conservative: the larger, the more points are labelled
                       noise (-1) and the more only truly dense regions become
                       clusters. 
  cluster_selection_method
                    -- 'eom' (default): selects the most stable, often larger clusters.
                       'leaf': selects the smallest, finest clusters in the tree.
  allow_single_cluster
                    -- by default HDBSCAN excludes the root of the cluster tree
                       (the whole dataset), so it can return 0 or >= 2 clusters
                       but never exactly 1. With --allow-single-cluster a
                       one-cluster solution becomes possible.

Scoring (label-free):
  HDBSCAN's relative_validity_ (fast DBCV approximation), higher = better.
  Plus sanity checks (number of clusters, noise fraction): configurations with
  one giant cluster or 90% noise can still score well on relative_validity.
  The script also reports how many configurations did NOT pass -- if almost
  nothing passes, that itself indicates there is no clear cluster structure.

Stability (top-5 configurations only):
  Repeated re-clustering on a random 80% of the SUBJECTS (not of the events,
  because events from the same night/person resemble each other). Per
  repetition: Adjusted Rand Index (ARI) between the labels on the full data
  and on the subset, over the shared events (noise counts as its own label).
  ARI ~1 = same partition, ~0 = no better than chance.

Final result:
  For the best configuration (or one chosen with --final), labels, membership
  probabilities and outlier scores per event are saved, plus per cluster its
  size and how many different subjects it contains.

Usage:
  python 4_hdbscan.py                          # all 3 runs, default grid
  python 4_hdbscan.py --allow-single-cluster   # all 3 runs, saved as <tag>_single
  python 4_hdbscan.py --quick                  # small grid, quick check
  python 4_hdbscan.py --runs whitened          # only selected run(s)
  python 4_hdbscan.py --final 130 25 eom       # fixed final configuration (all runs)
  python 4_hdbscan.py --output-dir             # other output folder (for every mode)

  # Only redraw figures from saved hdbscan_final_labels.csv (no fitting):
  python 4_hdbscan.py --plot-only              # EVERY subfolder of the output dir
                                               # that contains hdbscan_final_labels.csv
  python 4_hdbscan.py --plot-only --runs whitened unwhitened
  python 4_hdbscan.py --plot-only --runs whitened --allow-single-cluster   # whitened_single
  python 4_hdbscan.py --plot-only --output-dir "D:\\andere\\map"

Output (in OUTPUT_DIR/<tag>[_single]):
  hdbscan_grid.csv                  -- every configuration
  hdbscan_ranked.csv                -- ranked (valid, then relative_validity)
  hdbscan_stability.csv             -- ARI stability of the top N
  hdbscan_final_labels.csv          -- metadata + features + label/probability/outlier_score
  hdbscan_final_cluster_summary.csv -- per cluster: n, %, n_subjects, largest subject share
  fig_clusters_main.png/.pdf        -- 2 scatterplots, for the results section
  fig_clusters_pairs.png/.pdf       -- all feature/PC pairs (corner plot), for the appendix

And in OUTPUT_DIR:
  hdbscan_runs_overview[_single].csv -- one row per run: best configuration + stability
=============================================================================
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score

try:
    import hdbscan
except ImportError:
    sys.exit("hdbscan is not installed:  pip install hdbscan")

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2")
# PCA_DIR = BASE / r"4. clustering\3. pca"
# REDUCED_INPUT = BASE / r"reduced feature matrix\arousal_features_reduced_scaled.csv"
# OUTPUT_DIR = BASE / r"4. clustering\4. hdbscan"

# PATHS TO CENTERED DATA 
PCA_DIR = BASE / r"5. clustering_centered/2. pca centered"
REDUCED_INPUT = BASE / r"centered feature matrix\arousal_features_reduced_scaled_centered_participant.csv"
OUTPUT_DIR = BASE / r"5. clustering_centered/3. hdbscan centered"

# Number of PCs in the PCA score files (pca_scores_<k>pc_*.csv).
# None = detect automatically (if there is exactly one file per variant).
PCA_N_COMPONENTS = 5 

# The three runs
RUN_TAGS = ["whitened_centered", "unwhitened_centered", "features_8_centered"]  

# Full grid
DEFAULT_MIN_CLUSTER_SIZE = [65, 130, 260, 650]
DEFAULT_MIN_SAMPLES = [5, 10, 25, 50, 100]
DEFAULT_SELECTION = ["eom", "leaf"]

# --quick grid
QUICK_MIN_CLUSTER_SIZE = [130, 260]
QUICK_MIN_SAMPLES = [10, 50]
QUICK_SELECTION = ["eom"]

# Sanity checks for "valid" configurations
MIN_CLUSTERS = 1
MAX_CLUSTERS = 5
MAX_NOISE_FRACTION = 0.5

# Stability
TOP_N_STABILITY = 5
N_RESAMPLES = 10
SUBJECT_FRACTION = 0.80
RANDOM_STATE = 2554542

# Metadata
METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event", "stage_rk",
    "cluster", "membership_probability", "outlier_score",
]

# Axis labels: interpretation per PC (based on the loadings from 3_pca.py)
PC_LABELS = {
    "PC1": "PC1: overall spectral activation",
    "PC2": "PC2: event duration",
    "PC3": "PC3: contrast PPG vs. duration",
    "PC4": "PC4: head movement",
    "PC5": "PC5: contrast slow vs. fast frequencies",
    "mean_delta_ratio": "Mean delta ratio ($z$)",
    "mean_theta_ratio": "Mean theta ratio ($z$)",
    "mean_alpha_ratio": "Mean alpha ratio ($z$)",
    "mean_sigma_ratio": "Mean sigma ratio ($z$)",
    "mean_beta_ratio": "Mean beta ratio ($z$)",
    "duration_sec": "Duration ($z$)",
    "oxy_amp_ratio": "PPG amplitude ratio ($z$)",
    "motion_rms": "Head movement RMS ($z$)",
}
# Fixed plot pairs for features (instead of PCs) when there are < 2 clusters
FEATURE_PLOT_PAIRS = [("mean_theta_ratio", "duration_sec"),
                      ("mean_delta_ratio", "mean_beta_ratio")]

FIG_DPI = 300

# =============================================================================
# APA 7 FIGURE STYLE
# =============================================================================

# Fixed colour + marker per CLUSTER NUMBER 
CLUSTER_COLORS = ["#2a78d6",  # 0 blue
                  "#1baf7a",  # 1 green
                  "#eb6834",  # 2 orange
                  "#4a3aa7",  # 3 purple
                  "#e87ba4",  # 4 pink
                  "#eda100",  # 5 amber
                  "#e34948",  # 6 red
                  "#008300"]  # 7 dark green (spare)
CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
NOISE_COLOR = "#C4C4C4"
TEXT_COLOR = "#000000"
MUTED_COLOR = "#555555"
POINT_SIZE = 5
POINT_ALPHA = 0.45

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
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


def save_figure(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# STEP 1: CONFIGURATION
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist -- run 3_pca.py / 3.3_reduce_features.py first.")
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
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """PC columns if present, otherwise all numeric non-metadata columns."""
    pc_cols = [c for c in df.columns if c.upper().startswith("PC")]
    if pc_cols:
        return pc_cols
    return [c for c in df.columns
            if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])]


def load_scores(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = read_csv_robust(path)
    pc_cols = get_feature_cols(df)
    if not pc_cols:
        raise ValueError(f"No feature columns found in {path}.")
    if df[pc_cols].isna().any().any():
        raise ValueError("NaN in the features -- resolve this upstream first.")
    if "subject_id" not in df.columns:
        raise ValueError("Column 'subject_id' is missing (needed for the stability check).")
    print(f"Loaded: {path}")
    kind = "PCs" if pc_cols[0].upper().startswith("PC") else "features"
    print(f"{len(df)} events x {len(pc_cols)} {kind}: {pc_cols}")
    return df, pc_cols


def find_pca_file(pca_dir: Path, variant: str, n_pc: int | None) -> Path:
    """Locate pca_scores_<k>pc_<variant>.csv; detect k if n_pc is None."""
    if n_pc is not None:
        return pca_dir / f"pca_scores_{n_pc}pc_{variant}.csv"
    matches = sorted(pca_dir.glob(f"pca_scores_*pc_{variant}.csv"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No pca_scores_*pc_{variant}.csv in {pca_dir} -- run 3_pca.py first.")
    if len(matches) > 1:
        print(f"[NOTE] several {variant} PCA files found; using the most recent: {matches[0].name} "
              "(set PCA_N_COMPONENTS to choose explicitly).")
    return matches[0]


def build_runs(args) -> list[tuple[str, Path]]:
    """(tag, input path) per run; tags get '_single' with --allow-single-cluster."""
    suffix = "_single" if args.allow_single_cluster else ""
    if args.input is not None:                       # single custom run
        return [((args.tag or args.input.stem) + suffix, args.input)]
    inputs = {
        "whitened": lambda: find_pca_file(args.pca_dir, "whitened", args.n_pcs),
        "unwhitened": lambda: find_pca_file(args.pca_dir, "unwhitened", args.n_pcs),
        "reduced_features": lambda: args.reduced_input,
    }
    return [(tag + suffix, inputs[tag]()) for tag in args.runs]


def build_plot_only_runs(args, runs_explicit: bool) -> list[tuple[str, None]]:
    """
    Which subfolders of the output dir to redraw with --plot-only
    """
    if not args.output_dir.exists():
        sys.exit(f"Output dir does not exist: {args.output_dir}")
    suffix = "_single" if args.allow_single_cluster else ""
    if args.tag:
        tags = [args.tag + suffix]
    elif runs_explicit:
        tags = [t + suffix for t in args.runs]
    else:
        tags = sorted(p.name for p in args.output_dir.iterdir()
                      if p.is_dir() and (p / "hdbscan_final_labels.csv").exists())
    if not tags:
        sys.exit(f"No subfolders with hdbscan_final_labels.csv found in {args.output_dir}")
    return [(t, None) for t in tags]


# =============================================================================
# STEP 2: HDBSCAN
# =============================================================================

def fit_hdbscan(X: np.ndarray, mcs: int, ms: int, method: str,
                allow_single: bool, validity: bool = True):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        cluster_selection_method=method,
        allow_single_cluster=allow_single,
        gen_min_span_tree=validity,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X)
    return clusterer, labels


def describe_labels(labels: np.ndarray) -> dict:
    clusters = [l for l in np.unique(labels) if l != -1]
    sizes = sorted([int((labels == l).sum()) for l in clusters], reverse=True)
    n = len(labels)
    return {
        "n_clusters": len(clusters),
        "noise_fraction": float(np.mean(labels == -1)),
        "largest_cluster_fraction": sizes[0] / n if sizes else 0.0,
        "cluster_sizes": ",".join(map(str, sizes)),
    }


def run_grid(X, mcs_list, ms_list, methods, allow_single) -> pd.DataFrame:
    configs = list(itertools.product(mcs_list, ms_list, methods))
    rows = []
    t_start = time.time()
    for i, (mcs, ms, method) in enumerate(configs, start=1):
        t0 = time.time()
        clusterer, labels = fit_hdbscan(X, mcs, ms, method, allow_single)
        try:
            rel_val = float(clusterer.relative_validity_)
        except Exception:
            rel_val = np.nan
        info = describe_labels(labels)
        rows.append({"min_cluster_size": mcs, "min_samples": ms, "selection": method,
                     "relative_validity": rel_val, **info})
        print(f"[{i}/{len(configs)}] mcs={mcs:<4} ms={ms:<4} {method:<4} -> "
              f"{info['n_clusters']} cluster(s), noise {info['noise_fraction']:.1%}, "
              f"largest {info['largest_cluster_fraction']:.1%}, "
              f"rel_val {rel_val:.3f}  ({time.time() - t0:.1f}s)")
    print(f"\nGrid finished in {time.time() - t_start:.1f}s ({len(rows)} configurations)")
    return pd.DataFrame(rows)


def rank_grid(grid: pd.DataFrame) -> pd.DataFrame:
    grid = grid.copy()
    grid["valid"] = (
        (grid["n_clusters"] >= MIN_CLUSTERS)
        & (grid["n_clusters"] <= MAX_CLUSTERS)
        & (grid["noise_fraction"] <= MAX_NOISE_FRACTION)
    )
    return grid.sort_values(["valid", "relative_validity"], ascending=[False, False]
                            ).reset_index(drop=True)


# =============================================================================
# STEP 3: STABILITY (subject-level split)
# =============================================================================

def stability(X: np.ndarray, subjects: np.ndarray, mcs: int, ms: int, method: str,
              allow_single: bool, n_resamples: int, frac: float, seed: int) -> dict:
    """
    Fit on all data, then repeatedly on a random fraction of the subjects.
    ARI between both labellings on the shared events (noise = own label).
    """
    _, full_labels = fit_hdbscan(X, mcs, ms, method, allow_single, validity=False)
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    n_pick = max(2, int(round(frac * len(uniq))))

    aris, n_clusters = [], []
    for _ in range(n_resamples):
        picked = rng.choice(uniq, size=n_pick, replace=False)
        mask = np.isin(subjects, picked)
        _, sub_labels = fit_hdbscan(X[mask], mcs, ms, method, allow_single, validity=False)
        aris.append(adjusted_rand_score(full_labels[mask], sub_labels))
        n_clusters.append(len(set(sub_labels)) - (1 if -1 in sub_labels else 0))

    return {
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris, ddof=1)) if len(aris) > 1 else np.nan,
        "ari_min": float(np.min(aris)),
        "n_clusters_subsets_mean": float(np.mean(n_clusters)),
        "n_clusters_subsets_range": f"{min(n_clusters)}-{max(n_clusters)}",
    }


# =============================================================================
# STEP 4: FINAL CONFIGURATION
# =============================================================================

def final_fit(df: pd.DataFrame, pc_cols: list[str], X: np.ndarray,
              mcs: int, ms: int, method: str, allow_single: bool,
              out_dir: Path) -> dict:
    clusterer, labels = fit_hdbscan(X, mcs, ms, method, allow_single)
    out = df.copy()
    out["cluster"] = labels
    out["membership_probability"] = clusterer.probabilities_
    out["outlier_score"] = clusterer.outlier_scores_
    out.to_csv(out_dir / "hdbscan_final_labels.csv", index=False)

    n = len(out)
    rows = []
    for c, sub in out.groupby("cluster"):
        subj_counts = sub["subject_id"].value_counts()
        rows.append({
            "cluster": c,
            "n_events": len(sub),
            "pct_events": round(100 * len(sub) / n, 2),
            "n_subjects": int(sub["subject_id"].nunique()),
            "n_nights": int(sub.groupby(["subject_id", "night_id"]).ngroups)
                        if "night_id" in sub.columns else np.nan,
            "largest_subject_share_pct": round(100 * subj_counts.iloc[0] / len(sub), 1),
            "largest_subject": subj_counts.index[0],
            "mean_membership_probability": round(sub["membership_probability"].mean(), 3),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "hdbscan_final_cluster_summary.csv", index=False)

    try:
        rel_val = float(clusterer.relative_validity_)
    except Exception:
        rel_val = np.nan
    print(f"\nFinal configuration: min_cluster_size={mcs}, min_samples={ms}, {method} "
          f"(relative_validity {rel_val:.3f})")
    print(f"(cluster -1 = noise; {out['subject_id'].nunique()} subjects in total)")
    print(summary.to_string(index=False))

    make_plots(out, pc_cols, out_dir)
    info = describe_labels(labels)
    return {"final_min_cluster_size": mcs, "final_min_samples": ms, "final_selection": method,
            "final_relative_validity": rel_val, "final_n_clusters": info["n_clusters"],
            "final_noise_fraction": info["noise_fraction"],
            "final_cluster_sizes": info["cluster_sizes"]}


# =============================================================================
# STEP 5: PLOTS
# =============================================================================

def _style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)


def _cluster_ids(labels: pd.Series) -> list[int]:
    """Clusters sorted by size (largest first), without noise."""
    counts = labels[labels != -1].value_counts()
    return list(counts.index)


def _style_map(labels: pd.Series) -> dict:
    """
    Colour and marker follow the cluster NUMBER, so they are consistent across runs.
    The dict order stays large -> small, so small clusters are drawn on top
    in _scatter_pair.
    """
    ids = _cluster_ids(labels)
    if ids and max(ids) >= len(CLUSTER_COLORS):
        print(f"[NOTE] cluster {max(ids)} but only {len(CLUSTER_COLORS)} colours -- "
              "higher cluster numbers reuse colours.")
    return {c: (CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)],
                CLUSTER_MARKERS[int(c) % len(CLUSTER_MARKERS)])
            for c in ids}


def _legend_handles(labels: pd.Series, styles: dict) -> list:
    n = len(labels)
    handles = []
    for c in sorted(styles):                       # legend in cluster-number order
        col, mk = styles[c]
        k = int((labels == c).sum())
        handles.append(Line2D([], [], ls="", marker=mk, markersize=7, color=col,
                              label=f"Cluster {c} ($n$ = {k:,}, {100 * k / n:.1f}%)"))
    k_noise = int((labels == -1).sum())
    if k_noise:
        handles.append(Line2D([], [], ls="", marker="o", markersize=6, color=NOISE_COLOR,
                              label=f"Noise ($n$ = {k_noise:,}, {100 * k_noise / n:.1f}%)"))
    return handles


def _scatter_pair(ax, df: pd.DataFrame, x: str, y: str, styles: dict,
                  annotate: bool = True, size: float = POINT_SIZE) -> None:
    """Noise underneath, then clusters from large to small (small clusters on top)."""
    noise = df[df["cluster"] == -1]
    if len(noise):
        ax.scatter(noise[x], noise[y], s=size * 0.7, c=NOISE_COLOR, alpha=0.35,
                   linewidths=0, rasterized=True, zorder=1)
    for c, (col, mk) in styles.items():
        sub = df[df["cluster"] == c]
        ax.scatter(sub[x], sub[y], s=size, c=col, marker=mk, alpha=POINT_ALPHA,
                   linewidths=0, rasterized=True, zorder=2)
        if annotate and len(sub):
            ax.annotate(f"{c}", (sub[x].median(), sub[y].median()), fontsize=9,
                        fontweight="bold", color=TEXT_COLOR, ha="center", va="center", zorder=4,
                        bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=col, lw=1.5))


def _discriminating_pair(df: pd.DataFrame, pc_cols: list[str]) -> tuple[str, str]:
    """
    The PC pair on which the clusters differ most
    """
    sub = df[df["cluster"] != -1]
    if sub["cluster"].nunique() < 2:
        if not pc_cols[0].upper().startswith("PC") and all(c in pc_cols for c in FEATURE_PLOT_PAIRS[1]):
            return FEATURE_PLOT_PAIRS[1]
        return (pc_cols[0], pc_cols[4]) if len(pc_cols) >= 5 else (pc_cols[0], pc_cols[-1])
    grand = sub[pc_cols].mean()
    between = sub.groupby("cluster")[pc_cols].apply(
        lambda g: len(g) * (g.mean() - grand) ** 2).sum()
    within = sub.groupby("cluster")[pc_cols].apply(lambda g: ((g - g.mean()) ** 2).sum()).sum()
    f = (between / within).sort_values(ascending=False)
    pair = sorted(f.index[:2], key=pc_cols.index)
    print("Separation per dimension (between/within-cluster variance): "
          + ", ".join(f"{k} {v:.2f}" for k, v in f.items()))
    return pair[0], pair[1]


def _axis_label(col: str) -> str:
    return PC_LABELS.get(col, col.replace("_", " "))


def _short_label(col: str) -> str:
    """Short label for the corner plot: 'PC1' for PCs, readable name for features."""
    if col.upper().startswith("PC"):
        return col
    return _axis_label(col)


def plot_main(df: pd.DataFrame, pc_cols: list[str], out_dir: Path) -> None:
    """Figure for the results section: two scatterplots (A, B)."""
    styles = _style_map(df["cluster"])
    first = FEATURE_PLOT_PAIRS[0]
    if not pc_cols[0].upper().startswith("PC") and all(c in pc_cols for c in first):
        pairs = [first]
    else:
        pairs = [(pc_cols[0], pc_cols[1])]
    disc = _discriminating_pair(df, pc_cols)
    if set(disc) != set(pairs[0]):
        pairs.append(disc)
    elif len(pc_cols) >= 4:
        pairs.append((pc_cols[2], pc_cols[3]))

    fig, axes = plt.subplots(1, len(pairs), figsize=(6.2 * len(pairs), 5.6))
    axes = np.atleast_1d(axes)
    for i, (ax, (x, y)) in enumerate(zip(axes, pairs)):
        _scatter_pair(ax, df, x, y, styles)
        ax.set_xlabel(_axis_label(x))
        ax.set_ylabel(_axis_label(y))
        if len(pairs) > 1:
            ax.text(-0.12, 1.03, "AB"[i], transform=ax.transAxes, fontsize=12,
                    fontweight="bold", color=TEXT_COLOR)
        _style_axes(ax)

    fig.legend(handles=_legend_handles(df["cluster"], styles), loc="lower center",
               ncol=min(4, len(styles) + 1), frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, out_dir / "fig_clusters_main.png")
    print(f"Figure saved: {out_dir / 'fig_clusters_main.png'} (+ .pdf)")


def plot_pairs(df: pd.DataFrame, pc_cols: list[str], out_dir: Path) -> None:
    """
    Corner plot: every pair as a scatterplot below the diagonal, the
    distribution per cluster on the diagonal (density, so small clusters
    remain visible).
    """
    styles = _style_map(df["cluster"])
    k = len(pc_cols)
    fig, axes = plt.subplots(k, k, figsize=(2.6 * k, 2.6 * k))
    for i, yi in enumerate(pc_cols):
        for j, xj in enumerate(pc_cols):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                lo, hi = np.percentile(df[xj], [0.5, 99.5])
                bins = np.linspace(lo, hi, 40)
                noise = df.loc[df["cluster"] == -1, xj]
                if len(noise):
                    ax.hist(noise, bins=bins, density=True, histtype="stepfilled",
                            color=NOISE_COLOR, alpha=0.6)
                for c, (col, _) in styles.items():
                    ax.hist(df.loc[df["cluster"] == c, xj], bins=bins, density=True,
                            histtype="step", color=col, lw=1.6)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
            else:
                _scatter_pair(ax, df, xj, yi, styles, annotate=False, size=2.5)
            _style_axes(ax)
            if i == k - 1:
                ax.set_xlabel(_short_label(xj), fontsize=10)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(_short_label(yi), fontsize=10)
            elif j > 0:
                ax.set_yticklabels([])

    # legend
    fig.legend(handles=_legend_handles(df["cluster"], styles), loc="upper right",
               bbox_to_anchor=(0.98, 0.98), frameon=False, fontsize=13)
    if pc_cols[0].upper().startswith("PC"):
        pc_txt = "\n".join(_axis_label(pc) for pc in pc_cols)
        fig.text(0.98, 0.72, pc_txt, ha="right", va="top", fontsize=13, color=MUTED_COLOR)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig_clusters_pairs.png")
    print(f"Figure saved: {out_dir / 'fig_clusters_pairs.png'} (+ .pdf)")


def make_plots(df: pd.DataFrame, pc_cols: list[str], out_dir: Path) -> None:
    n_clusters = df.loc[df["cluster"] != -1, "cluster"].nunique()
    if n_clusters == 0:
        print("[NOTE] No clusters found (everything noise) -- figures show noise only.")
    plot_main(df, pc_cols, out_dir)
    plot_pairs(df, pc_cols, out_dir)


# =============================================================================
# ONE RUN
# =============================================================================

def run_one(args, tag: str, input_path: Path) -> dict:
    print("\n" + "=" * 78)
    print(f"RUN: {tag}" + ("   (plot only)" if args.plot_only
                           else f"   (allow_single_cluster={args.allow_single_cluster})"))
    print("=" * 78)
    out_dir = args.output_dir / tag

    if args.plot_only:
        labels_path = out_dir / "hdbscan_final_labels.csv"
        if not labels_path.exists():
            print(f"[SKIP] {labels_path} not found -- run without --plot-only first.")
            return {"tag": tag, "redrawn": False}
        df_lab = read_csv_robust(labels_path)
        make_plots(df_lab, get_feature_cols(df_lab), out_dir)
        return {"tag": tag, "redrawn": True}

    out_dir.mkdir(parents=True, exist_ok=True)
    df, pc_cols = load_scores(input_path)
    X = df[pc_cols].to_numpy()
    subjects = df["subject_id"].astype(str).to_numpy()

    n_cfg = len(args.min_cluster_size) * len(args.min_samples) * len(args.selection)
    print(f"Grid: {len(args.min_cluster_size)} x {len(args.min_samples)} x "
          f"{len(args.selection)} = {n_cfg} configurations\n")

    # Grid
    grid = run_grid(X, args.min_cluster_size, args.min_samples, args.selection,
                    args.allow_single_cluster)
    grid.to_csv(out_dir / "hdbscan_grid.csv", index=False)
    ranked = rank_grid(grid)
    ranked.to_csv(out_dir / "hdbscan_ranked.csv", index=False)

    n_valid = int(ranked["valid"].sum())
    print(f"\n{n_valid} of {len(ranked)} configurations valid "
          f"({MIN_CLUSTERS}-{MAX_CLUSTERS} clusters, <= {MAX_NOISE_FRACTION:.0%} noise).")
    print(f"  1 cluster:                  {int((ranked['n_clusters'] == 1).sum())}")
    print(f"  0 clusters (all noise):     {int((ranked['n_clusters'] == 0).sum())}")
    print(f"  > {MAX_NOISE_FRACTION:.0%} noise:                "
          f"{int((ranked['noise_fraction'] > MAX_NOISE_FRACTION).sum())}")
    if n_valid == 0:
        print("[NOTE] No configuration is valid -- an indication that there is no clear "
              "cluster structure (or that the grid needs adjusting).")

    show = ["min_cluster_size", "min_samples", "selection", "valid", "relative_validity",
            "n_clusters", "noise_fraction", "largest_cluster_fraction", "cluster_sizes"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\nTop 10:")
        print(ranked[show].head(10).to_string(index=False))

    # Stability of the top configs
    top = ranked.head(args.top_n)
    print(f"\nStability top {len(top)} ({args.n_resamples}x {SUBJECT_FRACTION:.0%} of the subjects):")
    stab_rows = []
    for _, r in top.iterrows():
        s = stability(X, subjects, int(r["min_cluster_size"]), int(r["min_samples"]),
                      r["selection"], args.allow_single_cluster, args.n_resamples,
                      SUBJECT_FRACTION, RANDOM_STATE)
        stab_rows.append({"min_cluster_size": int(r["min_cluster_size"]),
                          "min_samples": int(r["min_samples"]), "selection": r["selection"],
                          "valid": bool(r["valid"]), "n_clusters_full": int(r["n_clusters"]), **s})
        print(f"  mcs={int(r['min_cluster_size']):<4} ms={int(r['min_samples']):<4} {r['selection']:<4} "
              f"-> ARI {s['ari_mean']:.2f} ± {s['ari_std']:.2f} (min {s['ari_min']:.2f}), "
              f"clusters in subsets {s['n_clusters_subsets_range']}")
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(out_dir / "hdbscan_stability.csv", index=False)

    # Final config
    if args.final:
        mcs, ms, method = int(args.final[0]), int(args.final[1]), args.final[2]
        if method not in ("eom", "leaf"):
            raise ValueError("--final METHOD must be 'eom' or 'leaf'.")
    else:
        best = ranked.iloc[0]
        mcs, ms, method = int(best["min_cluster_size"]), int(best["min_samples"]), best["selection"]
    final = final_fit(df, pc_cols, X, mcs, ms, method, args.allow_single_cluster, out_dir)

    match = stab[(stab["min_cluster_size"] == mcs) & (stab["min_samples"] == ms)
                 & (stab["selection"] == method)]
    stab_final = match.iloc[0].to_dict() if len(match) else {}

    print(f"\nRun '{tag}' saved in: {out_dir}")
    return {
        "tag": tag, "input": str(input_path), "allow_single_cluster": args.allow_single_cluster,
        "n_events": len(df), "n_dimensions": len(pc_cols),
        "n_configurations": len(ranked), "n_valid": n_valid,
        "n_one_cluster": int((ranked["n_clusters"] == 1).sum()),
        "n_all_noise": int((ranked["n_clusters"] == 0).sum()),
        **final,
        "final_ari_mean": stab_final.get("ari_mean", np.nan),
        "final_ari_min": stab_final.get("ari_min", np.nan),
        "final_n_clusters_subsets_range": stab_final.get("n_clusters_subsets_range", ""),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", choices=RUN_TAGS, default=None,
                        help="Which runs to do (default: all three; with --plot-only: "
                             "every subfolder that has saved labels)")
    parser.add_argument("--pca-dir", type=Path, default=PCA_DIR)
    parser.add_argument("--n-pcs", type=int, default=PCA_N_COMPONENTS,
                        help="Number of PCs in the PCA score file names (default: auto-detect)")
    parser.add_argument("--reduced-input", type=Path, default=REDUCED_INPUT)
    parser.add_argument("--input", type=Path, default=None,
                        help="Optional: one custom input file instead of the three runs")
    parser.add_argument("--tag", default=None,
                        help="Tag for --input (default: file name); with --plot-only: "
                             "redraw only this subfolder")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Output folder (default: {OUTPUT_DIR}); also used by --plot-only")
    parser.add_argument("--min-cluster-size", type=int, nargs="+", default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, nargs="+", default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--selection", nargs="+", choices=["eom", "leaf"], default=DEFAULT_SELECTION)
    parser.add_argument("--allow-single-cluster", action="store_true",
                        help="Run all runs with allow_single_cluster=True; output in <tag>_single")
    parser.add_argument("--quick", action="store_true", help="Small grid for a quick check")
    parser.add_argument("--top-n", type=int, default=TOP_N_STABILITY,
                        help="Number of top configurations for the stability check")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--plot-only", action="store_true",
                        help="Only redraw the figures from hdbscan_final_labels.csv "
                             "(no fitting). Without --runs/--tag: every subfolder of the "
                             "output dir that contains that file.")
    parser.add_argument("--final", nargs=3, metavar=("MCS", "MS", "METHOD"),
                        help="Fixed final configuration for every run, e.g. --final 130 25 eom "
                             "(default: the best-ranked one per run)")
    args = parser.parse_args()

    runs_explicit = args.runs is not None
    if args.runs is None:
        args.runs = RUN_TAGS

    # Only plots: redraw figures from saved labels
    if args.plot_only:
        print(f"Plot only -- output dir: {args.output_dir}")
        runs = build_plot_only_runs(args, runs_explicit)
        results = [run_one(args, tag, None) for tag, _ in runs]
        done = [r["tag"] for r in results if r["redrawn"]]
        skipped = [r["tag"] for r in results if not r["redrawn"]]
        print("\nFigures redrawn for: " + (", ".join(done) if done else "(none)"))
        if skipped:
            print("Skipped (no labels file): " + ", ".join(skipped))
        return

    if args.quick:
        args.min_cluster_size = QUICK_MIN_CLUSTER_SIZE
        args.min_samples = QUICK_MIN_SAMPLES
        args.selection = QUICK_SELECTION
        args.n_resamples = min(args.n_resamples, 3)
        print("--quick: small grid.\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = build_runs(args)

    overview = [run_one(args, tag, path) for tag, path in runs]

    ov = pd.DataFrame(overview)
    ov_name = f"hdbscan_runs_overview{'_single' if args.allow_single_cluster else ''}.csv"
    ov.to_csv(args.output_dir / ov_name, index=False)

    print("\n" + "=" * 78)
    print("OVERVIEW OF ALL RUNS")
    print("=" * 78)
    cols = ["tag", "n_valid", "n_configurations", "n_one_cluster", "n_all_noise",
            "final_n_clusters", "final_noise_fraction", "final_relative_validity",
            "final_ari_mean", "final_n_clusters_subsets_range"]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(ov[cols].round(3).to_string(index=False))
    print(f"\nOverview saved: {args.output_dir / ov_name}")
    print("Per run (in OUTPUT_DIR/<tag>): hdbscan_grid.csv, hdbscan_ranked.csv, "
          "hdbscan_stability.csv, hdbscan_final_labels.csv, hdbscan_final_cluster_summary.csv, "
          "fig_clusters_main.png/.pdf, fig_clusters_pairs.png/.pdf")
    print("\nNext step: describe the clusters with the ORIGINAL features "
          "(arousal_features_reduced_original.csv) -- relative_validity and ARI are a "
          "ranking, not proof of a meaningful subtype.")


if __name__ == "__main__":
    main()