"""
3.3_umap_hdbscan_search.py

Step 3 of the arousal-subtyping pipeline: data-driven search over UMAP +
HDBSCAN parameters, since there are no ground-truth subtype labels to
validate against. This uses an internal (label-free) cluster validity
metric instead of accuracy/F1/etc.

Why UMAP and HDBSCAN must be searched TOGETHER, not one after the other:
    n_neighbors/min_dist/n_components shape the space HDBSCAN has to find
    clusters in, and min_cluster_size/min_samples decide what HDBSCAN then
    calls a cluster in that space. A UMAP setting that "looks clean" on its
    own can still produce a bad clustering, and vice versa - so this script
    fits every UMAP configuration once, then tries every HDBSCAN
    configuration on top of that SAME embedding.

Scoring:
    HDBSCAN's own `relative_validity_` (a fast DBCV-style density-based
    validity score, computed with gen_min_span_tree=True) is used, since it
    needs no labels. Higher is better. This is the standard label-free
    metric for exactly this UMAP -> HDBSCAN validation problem.

Because UMAP is itself stochastic (even with a fixed random_state, results
can vary in how "lucky" a given seed is), each UMAP configuration is run
across several seeds (--seeds) and results are aggregated (mean/std) rather
than trusting a single run.

Sanity constraints (to avoid rewarding degenerate "solutions" like one giant
cluster or 90% noise, which can still score deceptively well on relative
validity alone):
    --min-clusters      minimum number of clusters to count as valid (default 2)
    --max-clusters      maximum number of clusters to count as valid (default 10)
    --max-noise-fraction  maximum fraction of points HDBSCAN may label noise (default 0.3)

Usage:
    python 3.3_umap_hdbscan_search.py            # full default grid
    python 3.3_umap_hdbscan_search.py --quick     # small grid, for a fast sanity check

Output:
    hdbscan_search_raw.csv     - every individual run (one row per UMAP config
                                  x seed x HDBSCAN config)
    hdbscan_search_ranked.csv  - aggregated over seeds, ranked best-first by
                                  (fraction of seeds that passed the sanity
                                  constraints, then mean relative_validity)
"""

import argparse
import itertools
import sys
import time
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

# Hardcoded default paths - same input as 3.2_umap.py, edit here if they move.
DEFAULT_INPUT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
    r"\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled\arousal_feature_matrix_scaled.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
    r"\THESIS_OUTPUTS\PROJECT 2\3. feature selection\umap_hdbscan_search\3"
)

DEFAULT_ID_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec", "duration_sec", "sec_prev_event",
]

# Full default grid. Use --quick for a much smaller sanity-check grid.
DEFAULT_N_NEIGHBORS = [15,20,25]
DEFAULT_MIN_DIST = [0.0, 0.05, 0.1]
DEFAULT_N_COMPONENTS = [2, 5, 10]
DEFAULT_MIN_CLUSTER_SIZE = [15, ]
DEFAULT_MIN_SAMPLES = [5, 10, 12, 15]
DEFAULT_SEEDS = [42, 7, 123, 567, 684]

QUICK_N_NEIGHBORS = [30, 50]
QUICK_MIN_DIST = [0.0]
QUICK_N_COMPONENTS = [2, 3]
QUICK_MIN_CLUSTER_SIZE = [15, 25]
QUICK_MIN_SAMPLES = [10]
QUICK_SEEDS = [42]


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


def run_search(X, n_neighbors_list, min_dist_list, n_components_list,
                min_cluster_size_list, min_samples_list, seeds, metric):
    umap_configs = list(itertools.product(n_neighbors_list, min_dist_list, n_components_list))
    total = len(umap_configs) * len(seeds)
    results = []
    done = 0
    t_start = time.time()

    for n_neighbors, min_dist, n_components in umap_configs:
        for seed in seeds:
            done += 1
            t0 = time.time()
            reducer = umap.UMAP(
                n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
                metric=metric, random_state=seed, n_jobs=1,
            )
            embedding = reducer.fit_transform(X)
            umap_time = time.time() - t0
            print(f"[{done}/{total}] UMAP nn={n_neighbors} md={min_dist} nc={n_components} "
                  f"seed={seed} ({umap_time:.1f}s) -> testing {len(min_cluster_size_list) * len(min_samples_list)} HDBSCAN configs")

            for min_cluster_size, min_samples in itertools.product(min_cluster_size_list, min_samples_list):
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    gen_min_span_tree=True,
                )
                labels = clusterer.fit_predict(embedding)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_fraction = float(np.mean(labels == -1))
                try:
                    rel_val = float(clusterer.relative_validity_)
                except Exception:
                    rel_val = float("nan")

                results.append({
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "n_components": n_components,
                    "seed": seed,
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "noise_fraction": noise_fraction,
                    "relative_validity": rel_val,
                })

    print(f"\nSearch finished in {time.time() - t_start:.1f}s ({len(results)} HDBSCAN runs total)")
    return pd.DataFrame(results)


def aggregate_results(results_df, min_clusters, max_clusters, max_noise_fraction):
    results_df = results_df.copy()
    results_df["valid"] = (
        (results_df["n_clusters"] >= min_clusters)
        & (results_df["n_clusters"] <= max_clusters)
        & (results_df["noise_fraction"] <= max_noise_fraction)
    )

    group_cols = ["n_neighbors", "min_dist", "n_components", "min_cluster_size", "min_samples"]
    agg = results_df.groupby(group_cols).agg(
        mean_relative_validity=("relative_validity", "mean"),
        std_relative_validity=("relative_validity", "std"),
        mean_n_clusters=("n_clusters", "mean"),
        mean_noise_fraction=("noise_fraction", "mean"),
        frac_seeds_valid=("valid", "mean"),
        n_seeds=("valid", "count"),
    ).reset_index()

    agg = agg.sort_values(
        ["frac_seeds_valid", "mean_relative_validity"], ascending=[False, False]
    ).reset_index(drop=True)
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--id-columns", type=str, nargs="*", default=DEFAULT_ID_COLUMNS)
    parser.add_argument("--metric", type=str, default="euclidean")

    parser.add_argument("--n-neighbors", type=int, nargs="+", default=DEFAULT_N_NEIGHBORS)
    parser.add_argument("--min-dist", type=float, nargs="+", default=DEFAULT_MIN_DIST)
    parser.add_argument("--n-components", type=int, nargs="+", default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--min-cluster-size", type=int, nargs="+", default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, nargs="+", default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)

    parser.add_argument("--quick", action="store_true",
                         help="Use a small grid (1 seed) for a fast sanity check before the full search")

    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--max-noise-fraction", type=float, default=0.3)

    args = parser.parse_args()

    if args.quick:
        args.n_neighbors = QUICK_N_NEIGHBORS
        args.min_dist = QUICK_MIN_DIST
        args.n_components = QUICK_N_COMPONENTS
        args.min_cluster_size = QUICK_MIN_CLUSTER_SIZE
        args.min_samples = QUICK_MIN_SAMPLES
        args.seeds = QUICK_SEEDS
        print("Running in --quick mode with a reduced grid.\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_features(args.input, args.id_columns)
    X = df[feature_cols].to_numpy()

    n_umap_configs = len(args.n_neighbors) * len(args.min_dist) * len(args.n_components)
    n_hdbscan_configs = len(args.min_cluster_size) * len(args.min_samples)
    print(f"Grid: {n_umap_configs} UMAP configs x {len(args.seeds)} seeds x "
          f"{n_hdbscan_configs} HDBSCAN configs = "
          f"{n_umap_configs * len(args.seeds) * n_hdbscan_configs} total runs\n")

    results_df = run_search(
        X, args.n_neighbors, args.min_dist, args.n_components,
        args.min_cluster_size, args.min_samples, args.seeds, args.metric,
    )

    raw_path = args.output_dir / "hdbscan_search_raw.csv"
    results_df.to_csv(raw_path, index=False)
    print(f"\nSaved {raw_path}")

    ranked = aggregate_results(results_df, args.min_clusters, args.max_clusters, args.max_noise_fraction)
    ranked_path = args.output_dir / "hdbscan_search_ranked.csv"
    ranked.to_csv(ranked_path, index=False)
    print(f"Saved {ranked_path}")

    print(f"\nTop 10 configurations (ranked by fraction of seeds passing "
          f"{args.min_clusters}-{args.max_clusters} clusters & "
          f"<={args.max_noise_fraction:.0%} noise, then mean relative_validity):\n")
    cols_to_show = [
        "n_neighbors", "min_dist", "n_components", "min_cluster_size", "min_samples",
        "frac_seeds_valid", "mean_relative_validity", "std_relative_validity",
        "mean_n_clusters", "mean_noise_fraction",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(ranked[cols_to_show].head(10).to_string(index=False))

    print("\nNext step: take the top-ranked config(s), regenerate that specific "
          "UMAP embedding + HDBSCAN clustering, and inspect the resulting clusters "
          "directly (small multiples / boxplots per feature) before committing to it - "
          "relative_validity is a useful ranking signal, not a proof of a meaningful subtype.")


if __name__ == "__main__":
    main()