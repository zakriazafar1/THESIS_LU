"""
3.4_umap_hdbscan_bayesopt.py

Adaptive version of 3.3_umap_hdbscan_search.py: instead of trying every
combination in a fixed grid, this uses Bayesian optimization (Optuna's TPE
sampler) to pick the next UMAP+HDBSCAN configuration to try based on how
well previous configurations scored - spending more trials in promising
regions of parameter space and fewer in clearly bad ones.

Two phases:

    1. SEARCH   - Optuna proposes n_neighbors / min_dist / n_components /
                  min_cluster_size / min_samples each trial, fits UMAP +
                  HDBSCAN with ONE fixed seed (--search-seed) for speed and
                  a fair comparison between trials, and scores it with
                  HDBSCAN's relative_validity_ MINUS a soft penalty if the
                  result falls outside your constraints (too few/many
                  clusters, too much noise). The penalty is gradual, not a
                  hard reject, so the optimizer can still learn which
                  direction to move in even from an out-of-bounds trial.

    2. REFINE   - the top --top-k-refine distinct configurations found by
                  the search are then re-evaluated across several seeds
                  (--refine-seeds), exactly like the grid-search script, so
                  you can see which of the search's favorites are actually
                  stable rather than a lucky roll of one seed.

This does NOT replace 3.3_umap_hdbscan_search.py - a fixed grid is still
useful for exhaustively checking a small, deliberately chosen set of
values. Use this instead when you want to search wider ranges without
manually choosing which values in that range to test.

Usage:
    python 3.4_umap_hdbscan_bayes.py                  # default search + refine
    python 3.4_umap_hdbscan_bayes.py --n-trials 30     # quicker search

Output:
    bayesopt_trials.csv         - every trial from the search phase (params,
                                   score, n_clusters, noise_fraction,
                                   relative_validity)
    bayesopt_refined_raw.csv    - every individual refine-phase run (one row
                                   per top config x seed)
    bayesopt_refined_ranked.csv - refine-phase results aggregated over
                                   seeds, ranked best-first (same format as
                                   3.3's hdbscan_search_ranked.csv)
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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    sys.exit(
        "optuna is not installed. Install it with:\n"
        "    pip install optuna --break-system-packages"
    )

# Same input as 3.2_umap.py / 3.3_umap_hdbscan_search.py - edit if it moves.
DEFAULT_INPUT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
    r"\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled\arousal_feature_matrix_scaled.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
    r"\THESIS_OUTPUTS\PROJECT 2\3. feature selection\umap_hdbscan_bayes"
)

DEFAULT_ID_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec", "duration_sec", "sec_prev_event",
]

DEFAULT_REFINE_SEEDS = [42, 7, 123, 567, 684]


# ---- shared loading helpers (same as 3.2/3.3, duplicated to keep this a
# standalone script) -------------------------------------------------------

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


def fit_and_score(X, n_neighbors, min_dist, n_components, min_cluster_size, min_samples, seed, metric):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
        metric=metric, random_state=seed, n_jobs=1,
    )
    embedding = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_fraction = float(np.mean(labels == -1))
    try:
        rel_val = float(clusterer.relative_validity_)
    except Exception:
        rel_val = float("nan")

    return n_clusters, noise_fraction, rel_val


# ---- phase 1: Bayesian search ---------------------------------------------

def make_objective(X, args):
    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", args.n_neighbors_min, args.n_neighbors_max)
        min_dist = trial.suggest_float("min_dist", args.min_dist_min, args.min_dist_max)
        n_components = trial.suggest_int("n_components", args.n_components_min, args.n_components_max)
        min_cluster_size = trial.suggest_int("min_cluster_size", args.min_cluster_size_min, args.min_cluster_size_max)
        ms_high = min_cluster_size if args.min_samples_max is None else min(min_cluster_size, args.min_samples_max)
        ms_low = min(args.min_samples_min, ms_high)
        min_samples = trial.suggest_int("min_samples", ms_low, ms_high)

        n_clusters, noise_fraction, rel_val = fit_and_score(
            X, n_neighbors, min_dist, n_components, min_cluster_size, min_samples,
            args.search_seed, args.metric,
        )
        if np.isnan(rel_val):
            rel_val = -1.0

        score = rel_val
        if n_clusters < args.min_clusters:
            score -= 2.0 * (args.min_clusters - n_clusters)
        if args.max_clusters is not None and n_clusters > args.max_clusters:
            score -= 0.5 * (n_clusters - args.max_clusters)
        if noise_fraction > args.max_noise_fraction:
            score -= 5.0 * (noise_fraction - args.max_noise_fraction)

        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr("noise_fraction", noise_fraction)
        trial.set_user_attr("relative_validity", rel_val)

        print(f"[trial {trial.number}] nn={n_neighbors} md={min_dist:.3f} nc={n_components} "
              f"mcs={min_cluster_size} ms={min_samples} -> "
              f"n_clusters={n_clusters} noise={noise_fraction:.1%} "
              f"rel_val={rel_val:.3f} score={score:.3f}")

        return score

    return objective


# ---- phase 2: multi-seed refinement of the top candidates ----------------

def refine_top_configs(X, top_configs, seeds, metric):
    results = []
    for i, cfg in enumerate(top_configs, start=1):
        for seed in seeds:
            n_clusters, noise_fraction, rel_val = fit_and_score(
                X, cfg["n_neighbors"], cfg["min_dist"], cfg["n_components"],
                cfg["min_cluster_size"], cfg["min_samples"], seed, metric,
            )
            results.append({**cfg, "seed": seed, "n_clusters": n_clusters,
                             "noise_fraction": noise_fraction, "relative_validity": rel_val})
        print(f"Refined config {i}/{len(top_configs)}: {cfg}")
    return pd.DataFrame(results)


def aggregate_results(results_df, min_clusters, max_clusters, max_noise_fraction):
    results_df = results_df.copy()
    valid = results_df["n_clusters"] >= min_clusters
    if max_clusters is not None:
        valid &= results_df["n_clusters"] <= max_clusters
    valid &= results_df["noise_fraction"] <= max_noise_fraction
    results_df["valid"] = valid

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

    # search space bounds (ranges, not fixed lists - Optuna samples within them)
    parser.add_argument("--n-neighbors-min", type=int, default=15)
    parser.add_argument("--n-neighbors-max", type=int, default=100)
    parser.add_argument("--min-dist-min", type=float, default=0.0)
    parser.add_argument("--min-dist-max", type=float, default=0.2)
    parser.add_argument("--n-components-min", type=int, default=2)
    parser.add_argument("--n-components-max", type=int, default=10)
    parser.add_argument("--min-cluster-size-min", type=int, default=10)
    parser.add_argument("--min-cluster-size-max", type=int, default=60)
    parser.add_argument("--min-samples-min", type=int, default=3)
    parser.add_argument("--min-samples-max", type=int, default=None,
                         help="Defaults to min_cluster_size (per-trial) if not set")

    # search phase
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--search-seed", type=int, default=42,
                         help="Fixed UMAP seed used during the search phase, for fair comparison between trials")
    parser.add_argument("--study-seed", type=int, default=42,
                         help="Seed for Optuna's own sampler (reproducibility of which trials get proposed)")

    # refine phase
    parser.add_argument("--top-k-refine", type=int, default=10)
    parser.add_argument("--refine-seeds", type=int, nargs="+", default=DEFAULT_REFINE_SEEDS)

    # validity constraints (soft penalty during search, hard filter during refine ranking)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=str, default="none",
                         help="Max clusters to count as valid, or 'none' for no upper limit")
    parser.add_argument("--max-noise-fraction", type=float, default=0.3)

    args = parser.parse_args()
    args.max_clusters = None if args.max_clusters.lower() == "none" else int(args.max_clusters)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_features(args.input, args.id_columns)
    X = df[feature_cols].to_numpy()

    # --- Phase 1: search ---
    print(f"\nStarting Bayesian search: {args.n_trials} trials, search_seed={args.search_seed}\n")
    t0 = time.time()
    sampler = optuna.samplers.TPESampler(seed=args.study_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(X, args), n_trials=args.n_trials, show_progress_bar=False)
    print(f"\nSearch finished in {time.time() - t0:.1f}s")

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials_df.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in trials_df.columns]
    trials_path = args.output_dir / "bayesopt_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    print(f"Saved {trials_path}")

    print(f"\nBest single trial: {study.best_params} "
          f"(score={study.best_value:.3f}, n_clusters={study.best_trial.user_attrs['n_clusters']}, "
          f"noise={study.best_trial.user_attrs['noise_fraction']:.1%})")

    # --- Phase 2: refine top-K distinct configs across multiple seeds ---
    param_cols = ["n_neighbors", "min_dist", "n_components", "min_cluster_size", "min_samples"]
    ranked_trials = trials_df.sort_values("value", ascending=False)
    dedup = ranked_trials.drop_duplicates(subset=param_cols, keep="first")
    top_configs = dedup.head(args.top_k_refine)[param_cols].to_dict("records")

    print(f"\nRefining top {len(top_configs)} distinct configs across seeds {args.refine_seeds}...\n")
    refined_raw = refine_top_configs(X, top_configs, args.refine_seeds, args.metric)
    refined_raw_path = args.output_dir / "bayesopt_refined_raw.csv"
    refined_raw.to_csv(refined_raw_path, index=False)
    print(f"\nSaved {refined_raw_path}")

    refined_ranked = aggregate_results(refined_raw, args.min_clusters, args.max_clusters, args.max_noise_fraction)
    refined_ranked_path = args.output_dir / "bayesopt_refined_ranked.csv"
    refined_ranked.to_csv(refined_ranked_path, index=False)
    print(f"Saved {refined_ranked_path}")

    cols_to_show = param_cols + [
        "frac_seeds_valid", "mean_relative_validity", "std_relative_validity",
        "mean_n_clusters", "mean_noise_fraction",
    ]
    print("\nRefined ranking (top configs found by search, validated across seeds):\n")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(refined_ranked[cols_to_show].to_string(index=False))

    print("\nNext step: as before, inspect the winning config's actual clusters "
          "(small multiples / boxplots per feature) before committing to it.")


if __name__ == "__main__":
    main()