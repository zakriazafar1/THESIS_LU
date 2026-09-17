"""
3.4.1_cluster_inspect.py

Step 4 of the arousal-subtyping pipeline: take one specific, already-chosen
UMAP + HDBSCAN parameter combination (e.g. the winner from
3.3.2_umap_hdbscan_bayes.py) and actually produce the clustering + inspection
visuals, so you can judge whether the clusters mean anything - instead of
just looking at validity numbers.

This does three things:
    1. Fits UMAP once with your chosen parameters (defaults below match the
       n_components=2 winner: n_neighbors=17, min_dist=0.04146).
    2. Runs HDBSCAN on that embedding with your chosen min_cluster_size /
       min_samples.
    3. Produces:
       - the embedding + cluster labels as a CSV
       - a scatter plot colored by cluster label (noise in gray)
       - a grid of boxplots, one per diagnostic feature, showing that
         feature's distribution PER CLUSTER (not colored scatter this time -
         boxplots make it much easier to see whether a cluster's feature
         values are actually shifted vs. just visually separated). By
         default each panel uses a FIXED y-axis, computed once from the
         full feature matrix (not from this run's clusters), so boxplots
         from different parameter combinations line up on the same scale
         and are directly comparable. Use --boxplot-scale auto for the old
         per-run auto-scaling behaviour.
       - a summary table: size, % of data, mean noise/duration/features per
         cluster, and the sleep-stage (stage_rk) composition per cluster

Usage:
    python 3.4.1_cluster_inspect.py
    python 3.4.1_cluster_inspect.py --n-neighbors 17 --min-dist 0.04146 \\
        --min-cluster-size 11 --min-samples 5
    python 3.4.1_cluster_inspect.py --boxplot-scale auto

Output:
    embedding_with_clusters.csv - id columns + UMAP_1/2 + cluster label
    cluster_scatter.png         - 2D scatter colored by cluster
    cluster_boxplots.png        - one boxplot panel per diagnostic feature,
                                   grouped by cluster
    cluster_summary.csv         - per-cluster size, %, mean of each
                                   diagnostic feature, stage_rk composition
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import umap
except ImportError:
    sys.exit(
        "umap-learn is not installed. Install it with:\n"
        "    pip install umap-learn --break-system-packages"
    )

try:
    import hdbscan
except ImportError:
    sys.exit(
        "hdbscan is not installed. Install it with:\n"
        "    pip install hdbscan --break-system-packages"
    )

DEFAULT_INPUT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
    r"\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled\arousal_feature_matrix_scaled_clean.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
    r"\THESIS_OUTPUTS\PROJECT 2\4. clustering"
)

DEFAULT_ID_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec", "duration_sec", "sec_prev_event",
]

DIAGNOSTIC_FEATURES = [
    "duration_sec", "mean_alpha_ratio", "mean_theta_ratio", "mean_beta_ratio",
    "mean_sigma_ratio", "mean_delta_ratio", "oxy_amp_ratio", "motion_rms",
]


def detect_delimiter(path: Path, sample_lines: int = 5) -> str:
    import csv as csv_module

    with open(path, "r", encoding="utf-8-sig") as f:
        sample = "".join(f.readline() for _ in range(sample_lines))
    try:
        dialect = csv_module.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except csv_module.Error:
        return ","


def detect_decimal(path: Path, delimiter: str, sample_lines: int = 20) -> str:
    import re

    comma_decimal = re.compile(r"^-?\d+,\d+$")
    dot_decimal = re.compile(r"^-?\d+\.\d+$")
    comma_hits, dot_hits = 0, 0

    with open(path, "r", encoding="utf-8-sig") as f:
        next(f, None)
        for _ in range(sample_lines):
            line = f.readline()
            if not line:
                break
            for field in line.strip().split(delimiter):
                field = field.strip()
                if comma_decimal.match(field):
                    comma_hits += 1
                elif dot_decimal.match(field):
                    dot_hits += 1

    return "," if comma_hits > dot_hits else "."


def load_features(path: Path, id_columns):
    delimiter = detect_delimiter(path)
    decimal = detect_decimal(path, delimiter)
    df = pd.read_csv(path, sep=delimiter, decimal=decimal)
    print(f"Detected delimiter='{delimiter}', decimal='{decimal}'")

    present_id_cols = [c for c in id_columns if c in df.columns]
    feature_cols = [c for c in df.columns if c not in present_id_cols]

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns found in what should be the feature set: "
            f"{non_numeric}. Add them to --id-columns if they're identifiers."
        )
    if df[feature_cols].isna().any().any():
        raise ValueError("Found NaNs in the feature matrix - impute/drop them upstream.")

    print(f"Loaded {len(df)} events x {len(feature_cols)} features")
    return df, feature_cols


def compute_fixed_ylims(df, features, whisker_iqr=1.5, pad_frac=0.05):
    """Per-feature y-axis bounds computed once from the FULL feature matrix
    (i.e. independent of any particular clustering run), so that boxplots
    from different UMAP/HDBSCAN parameter combinations end up on the same
    scale and can be compared side by side.

    Uses the same Q1-1.5*IQR / Q3+1.5*IQR rule matplotlib's boxplot uses for
    its whiskers, clipped to the actual data range, plus a small padding
    fraction so boxes don't touch the panel edges.
    """
    ylims = {}
    for feat in features:
        if feat not in df.columns:
            continue
        col = df[feat].dropna()
        if col.empty:
            continue
        q1, q3 = col.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo = max(q1 - whisker_iqr * iqr, col.min())
        hi = min(q3 + whisker_iqr * iqr, col.max())
        if hi <= lo:  # degenerate (near-constant) feature - fall back to actual range
            lo, hi = col.min(), col.max()
        pad = (hi - lo) * pad_frac or 1e-6
        ylims[feat] = (lo - pad, hi + pad)
    return ylims


def make_cluster_scatter(embedding, labels, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(8, 7))
    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.35,
                       color="lightgray", label=f"noise (n={mask.sum()})")
        else:
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=10, alpha=0.7,
                       color=cmap(lab % 20), label=f"cluster {lab} (n={mask.sum()})")
    ax.set_xlabel("UMAP_1")
    ax.set_ylabel("UMAP_2")
    ax.set_title("UMAP embedding colored by HDBSCAN cluster")
    ax.legend(fontsize=8, loc="best", markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_cluster_boxplots(df, labels, features, out_path, fixed_ylims=None):
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  NOTE: diagnostic features not found, skipping: {missing}")
    if not present:
        print("  NOTE: no diagnostic features found - skipping boxplots.")
        return

    plot_df = df[present].copy()
    plot_df["cluster"] = labels
    # order: noise (-1) first, then clusters in ascending order
    cluster_order = sorted(plot_df["cluster"].unique(), key=lambda x: (x != -1, x))
    tick_labels = ["noise" if c == -1 else str(c) for c in cluster_order]

    n = len(present)
    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, present):
        data = [plot_df.loc[plot_df["cluster"] == c, feature].values for c in cluster_order]
        ax.boxplot(data, tick_labels=tick_labels, showfliers=False)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("cluster")
        ax.tick_params(axis="x", rotation=45)
        if fixed_ylims and feature in fixed_ylims:
            ax.set_ylim(*fixed_ylims[feature])

    for ax in axes[len(present):]:
        ax.axis("off")

    fig.suptitle("Feature distributions per HDBSCAN cluster", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_cluster_summary(df, labels, features):
    present = [f for f in features if f in df.columns]
    summary_df = df[present].copy()
    summary_df["cluster"] = labels

    rows = []
    n_total = len(df)
    cluster_order = sorted(summary_df["cluster"].unique(), key=lambda x: (x != -1, x))
    for c in cluster_order:
        sub = summary_df[summary_df["cluster"] == c]
        row = {
            "cluster": "noise" if c == -1 else c,
            "n_events": len(sub),
            "pct_of_total": 100 * len(sub) / n_total,
        }
        for feat in present:
            row[f"mean_{feat}"] = sub[feat].mean()
        if "stage_rk" in df.columns:
            stage_counts = df.loc[sub.index, "stage_rk"].value_counts(normalize=True).sort_index()
            for stage, frac in stage_counts.items():
                row[f"pct_stage_{stage}"] = 100 * frac
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--id-columns", type=str, nargs="*", default=DEFAULT_ID_COLUMNS)
    parser.add_argument("--diagnostic-features", type=str, nargs="*", default=DIAGNOSTIC_FEATURES)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument(
        "--boxplot-scale", choices=["fixed", "auto"], default="fixed",
        help="'fixed' (default): every combo's boxplots use the same y-axis per "
             "feature, computed once from the full feature matrix, so combos are "
             "directly comparable. 'auto': old behaviour, each panel auto-scales "
             "to that run's own clusters.",
    )

    # UMAP params - defaults match the n_components=2 Bayes-opt winner
    parser.add_argument("--n-neighbors", type=int, default=17)
    parser.add_argument("--min-dist", type=float, default=0.04146)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)

    # HDBSCAN params - defaults match the same winner
    parser.add_argument("--min-cluster-size", type=int, default=11)
    parser.add_argument("--min-samples", type=int, default=5)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_features(args.input, args.id_columns)
    X = df[feature_cols].to_numpy()

    fixed_ylims = None
    if args.boxplot_scale == "fixed":
        fixed_ylims = compute_fixed_ylims(df, args.diagnostic_features)

    print(f"\nFitting UMAP: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, "
          f"n_components={args.n_components}, random_state={args.random_state}")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors, min_dist=args.min_dist, n_components=args.n_components,
        metric=args.metric, random_state=args.random_state, n_jobs=1,
    )
    embedding = reducer.fit_transform(X)

    print(f"Running HDBSCAN: min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_fraction = float(np.mean(labels == -1))
    try:
        rel_val = float(clusterer.relative_validity_)
    except Exception:
        rel_val = float("nan")
    print(f"Found {n_clusters} clusters, {noise_fraction:.1%} noise, relative_validity={rel_val:.3f}")

    # --- Save embedding + labels ---
    present_id_cols = [c for c in args.id_columns if c in df.columns]
    emb_df = df[present_id_cols].copy()
    for i in range(args.n_components):
        emb_df[f"UMAP_{i + 1}"] = embedding[:, i]
    emb_df["cluster"] = labels
    emb_path = args.output_dir / "embedding_with_clusters.csv"
    emb_df.to_csv(emb_path, index=False)
    print(f"Saved {emb_path}")

    # --- Scatter colored by cluster (only meaningful/plottable for 2D) ---
    if args.n_components == 2:
        scatter_path = args.output_dir / "cluster_scatter.png"
        make_cluster_scatter(embedding, labels, scatter_path)
        print(f"Saved {scatter_path}")
    else:
        print(f"Skipping cluster scatter plot: n_components={args.n_components} != 2")

    # --- Boxplots per feature, grouped by cluster ---
    boxplot_path = args.output_dir / "cluster_boxplots.png"
    make_cluster_boxplots(df, labels, args.diagnostic_features, boxplot_path, fixed_ylims=fixed_ylims)
    print(f"Saved {boxplot_path} (boxplot-scale={args.boxplot_scale})")

    # --- Summary table ---
    summary = make_cluster_summary(df, labels, args.diagnostic_features)
    summary_path = args.output_dir / "cluster_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    print("\nCluster summary:\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.to_string(index=False))

    print("\nNext step: look for clusters whose mean feature values are clearly "
          "shifted (not just visually separated in the scatter) and whose "
          "stage_rk composition isn't just mirroring the overall dataset - "
          "that combination is what makes a cluster a plausible real subtype "
          "rather than an artifact of the embedding.")


if __name__ == "__main__":
    main()