"""
3.3.2_umap_hdbscan_bayes.py

Adaptive version of 3.3.1_umap_hdbscan_manual.py: instead of trying every
combination in a fixed grid, this uses Bayesian optimization (Optuna's TPE
sampler) to pick the next UMAP+HDBSCAN configuration to try based on how
well previous configurations scored - spending more trials in promising
regions of parameter space and fewer in clearly bad ones.

Two phases:

    1. SEARCH   - Optuna proposes n_neighbors / min_dist / n_components /
                  min_cluster_size / min_samples each trial, fits UMAP +
                  HDBSCAN averaged over one or more fixed seeds
                  (--search-seeds) for a fair, stable comparison between
                  trials, and scores it with HDBSCAN's relative_validity_
                  MINUS a soft penalty if the result falls outside your
                  constraints (too few/many clusters, too much noise). The
                  penalty is gradual, not a hard reject, so the optimizer
                  can still learn which direction to move in even from an
                  out-of-bounds trial.

    2. REFINE   - the top --top-k-refine distinct configurations found by
                  the search are then re-evaluated across several seeds
                  (--refine-seeds), exactly like the grid-search script, so
                  you can see which of the search's favorites are actually
                  stable rather than a lucky roll of one seed.

TRUSTWORTHINESS (new): relative_validity is a density-based metric that
structurally tends to score HIGHER in lower embedding dimensions - so it is
NOT a fair way to decide between e.g. n_components=2 vs 3. Alongside it,
this script now also computes sklearn's trustworthiness score, which
compares the embedding's neighborhood structure directly against the
neighborhood structure of your original 27 features. Trustworthiness tends
to score BETTER in higher dimensions (more of the original structure is
preserved), so the two metrics pull in opposite directions on
dimensionality - looking at both together, rather than either alone, is
what lets you make a defensible n_components choice. Trustworthiness is
reported everywhere relative_validity is, but it does NOT change what
Optuna optimizes for (still relative_validity) - it's there for you to
weigh manually per n_components, exactly as advised.

CONVERGENCE TRACKING: after the search phase, the best score seen so far is
plotted against trial number. Once that curve flattens out, more trials are
mostly wasted compute - this replaces guessing a --n-trials number up front
with actually looking at whether the search has plateaued.

This does NOT replace 3.3.1_umap_hdbscan_manual.py - a fixed grid is still
useful for exhaustively checking a small, deliberately chosen set of
values. Use this instead when you want to search wider ranges without
manually choosing which values in that range to test.

Usage:
    python 3.3.2_umap_hdbscan_bayes.py                        # default search + refine
    python 3.3.2_umap_hdbscan_bayes.py --n-trials 30           # quicker search
    python 3.3.2_umap_hdbscan_bayes.py --search-seeds 42 7 123 # average search over 3 seeds
    python 3.3.2_umap_hdbscan_bayes.py --n-neighbors-max none  # auto ceiling based on n_events

Output:
    bayesopt_trials.csv         - every trial from the search phase (params,
                                   score, n_clusters, noise_fraction,
                                   relative_validity, trustworthiness)
    bayesopt_convergence.csv    - trial number, its score, and the best score
                                   seen up to and including that trial
    bayesopt_convergence.png    - plot of the above, to visually check for a
                                   plateau
    bayesopt_refined_raw.csv    - every individual refine-phase run (one row
                                   per top config x seed), including
                                   trustworthiness per run
    bayesopt_refined_ranked.csv - refine-phase results aggregated over
                                   seeds, ranked best-first (same format as
                                   3.3.1's hdbscan_search_ranked.csv), with
                                   added mean/std trustworthiness columns
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

try:
    from sklearn.manifold import trustworthiness as sk_trustworthiness
except ImportError:
    sys.exit(
        "scikit-learn is not installed. Install it with:\n"
        "    pip install scikit-learn --break-system-packages"
    )

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

DEFAULT_REFINE_SEEDS = [42, 7, 123, 567, 684, 950, 328, 0, 988]


# ---- shared loading helpers -------------------------------------------------------

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

    nan_mask = df[feature_cols].isna().any(axis=1)
    n_nan_rows = nan_mask.sum()
    if n_nan_rows > 0:
        print(f"  Dropping {n_nan_rows} of {len(df)} events with NaN in feature columns")
        df = df.loc[~nan_mask].reset_index(drop=True)

    print(f"Loaded {len(df)} events x {len(feature_cols)} features")
    return df, feature_cols

def fit_and_score(X, n_neighbors, min_dist, n_components, min_cluster_size, min_samples, seed, metric,
                   trustworthiness_n_neighbors):
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

    # Trustworthiness: compares the embedding's neighborhood structure
    # directly against the original 27-feature space. Unlike
    # relative_validity, this does NOT structurally favor low dimensions -
    # if anything it favors higher ones (more original structure kept) -
    # so the two metrics together let you sanity-check an n_components
    # choice instead of trusting relative_validity alone.
    try:
        trust = float(sk_trustworthiness(X, embedding, n_neighbors=trustworthiness_n_neighbors))
    except Exception:
        trust = float("nan")

    return n_clusters, noise_fraction, rel_val, trust


# ---- Step 1: Bayesian search ---------------------------------------------

def make_objective(X, args):
    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", args.n_neighbors_min, args.n_neighbors_max)
        min_dist = trial.suggest_float("min_dist", args.min_dist_min, args.min_dist_max)
        n_components = trial.suggest_int("n_components", args.n_components_min, args.n_components_max)
        min_cluster_size = trial.suggest_int("min_cluster_size", args.min_cluster_size_min, args.min_cluster_size_max)
        ms_high = min_cluster_size if args.min_samples_max is None else min(min_cluster_size, args.min_samples_max)
        ms_low = min(args.min_samples_min, ms_high)
        min_samples = trial.suggest_int("min_samples", ms_low, ms_high)

        n_clusters_runs, noise_runs, relval_runs, trust_runs = [], [], [], []
        for seed in args.search_seeds:
            n_clusters, noise_fraction, rel_val, trust = fit_and_score(
                X, n_neighbors, min_dist, n_components, min_cluster_size, min_samples,
                seed, args.metric, args.trustworthiness_n_neighbors,
            )
            if np.isnan(rel_val):
                rel_val = -1.0
            n_clusters_runs.append(n_clusters)
            noise_runs.append(noise_fraction)
            relval_runs.append(rel_val)
            trust_runs.append(trust)

        n_clusters = float(np.mean(n_clusters_runs))
        noise_fraction = float(np.mean(noise_runs))
        rel_val = float(np.mean(relval_runs))
        trust = float(np.nanmean(trust_runs))

        # Optimization target is unchanged (relative_validity + penalties) -
        # trustworthiness is tracked for reporting/comparison only, not
        # optimized against, so it doesn't distort the search.
        score = rel_val
        if n_clusters < args.min_clusters:
            score -= 2.0 * (args.min_clusters - n_clusters)
        if args.max_clusters is not None and n_clusters > args.max_clusters:
            score -= 0.5 * (n_clusters - args.max_clusters)
        if args.max_noise_fraction is not None and noise_fraction > args.max_noise_fraction:
            score -= 5.0 * (noise_fraction - args.max_noise_fraction)

        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr("noise_fraction", noise_fraction)
        trial.set_user_attr("relative_validity", rel_val)
        trial.set_user_attr("trustworthiness", trust)

        print(f"[trial {trial.number}] nn={n_neighbors} md={min_dist:.3f} nc={n_components} "
              f"mcs={min_cluster_size} ms={min_samples} (avg over {len(args.search_seeds)} seed(s)) -> "
              f"n_clusters={n_clusters:.1f} noise={noise_fraction:.1%} "
              f"rel_val={rel_val:.3f} trust={trust:.3f} score={score:.3f}")

        return score

    return objective


# ---- Step 1.1: convergence tracking ---------------------------------------------

def save_convergence(trials_df, output_dir, plateau_window=None):
    """Track the best score seen so far as trials progress, save as CSV +
    plot, and print a plain-language plateau check so you don't have to
    guess --n-trials up front.
    """
    conv = trials_df[["number", "value"]].sort_values("number").reset_index(drop=True)
    conv["best_so_far"] = conv["value"].cummax()

    conv_path = output_dir / "bayesopt_convergence.csv"
    conv.to_csv(conv_path, index=False)
    print(f"Saved {conv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(conv["number"], conv["value"], s=12, alpha=0.35, color="tab:blue", label="trial score")
    ax.step(conv["number"], conv["best_so_far"], where="post", color="tab:orange", linewidth=2, label="best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Score (relative_validity - penalties)")
    ax.set_title("Bayesian search convergence")
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / "bayesopt_convergence.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}")

    # Plain-language plateau check: did the best score improve at all in the
    # last window of trials?
    n_trials = len(conv)
    if plateau_window is None:
        plateau_window = max(10, n_trials // 5)
    if n_trials > plateau_window:
        best_before = conv["best_so_far"].iloc[-plateau_window - 1]
        best_now = conv["best_so_far"].iloc[-1]
        improvement = best_now - best_before
        if improvement <= 1e-6:
            print(f"\nConvergence check: no improvement in the best score over the last "
                  f"{plateau_window} trials (still {best_now:.3f}) - the search looks to have "
                  f"plateaued. More trials probably won't help much; consider narrowing the "
                  f"search space instead if you want a better score.")
        else:
            print(f"\nConvergence check: best score improved by {improvement:.3f} over the last "
                  f"{plateau_window} trials (now {best_now:.3f}) - still improving, more trials "
                  f"may help.")

    return conv


# ---- Step 2: multi-seed refinement of the top candidates ----------------

def refine_top_configs(X, top_configs, seeds, metric, trustworthiness_n_neighbors):
    results = []
    for i, cfg in enumerate(top_configs, start=1):
        for seed in seeds:
            n_clusters, noise_fraction, rel_val, trust = fit_and_score(
                X, cfg["n_neighbors"], cfg["min_dist"], cfg["n_components"],
                cfg["min_cluster_size"], cfg["min_samples"], seed, metric, trustworthiness_n_neighbors,
            )
            results.append({**cfg, "seed": seed, "n_clusters": n_clusters,
                             "noise_fraction": noise_fraction, "relative_validity": rel_val,
                             "trustworthiness": trust})
        print(f"Refined config {i}/{len(top_configs)}: {cfg}")
    return pd.DataFrame(results)


def aggregate_results(results_df, min_clusters, max_clusters, max_noise_fraction):
    results_df = results_df.copy()
    valid = results_df["n_clusters"] >= min_clusters
    if max_clusters is not None:
        valid &= results_df["n_clusters"] <= max_clusters
    if max_noise_fraction is not None:
        valid &= results_df["noise_fraction"] <= max_noise_fraction
    results_df["valid"] = valid

    group_cols = ["n_neighbors", "min_dist", "n_components", "min_cluster_size", "min_samples"]
    agg = results_df.groupby(group_cols).agg(
        mean_relative_validity=("relative_validity", "mean"),
        std_relative_validity=("relative_validity", "std"),
        mean_trustworthiness=("trustworthiness", "mean"),
        std_trustworthiness=("trustworthiness", "std"),
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
    parser.add_argument("--trustworthiness-n-neighbors", type=int, default=10,
                         help="Neighborhood size used by the trustworthiness metric (independent of UMAP's own n_neighbors)")

    # search space bounds (ranges, not fixed lists - Optuna samples within them).
    parser.add_argument("--n-neighbors-min", type=int, default=20)
    parser.add_argument("--n-neighbors-max", type=str, default="400")
    parser.add_argument("--min-dist-min", type=float, default=0.0)
    parser.add_argument("--min-dist-max", type=str, default="0.2")
    parser.add_argument("--n-components-min", type=int, default=2)
    parser.add_argument("--n-components-max", type=str, default="3")
    parser.add_argument("--min-cluster-size-min", type=int, default=10)
    parser.add_argument("--min-cluster-size-max", type=str, default="250")
    parser.add_argument("--min-samples-min", type=int, default=1)
    parser.add_argument("--min-samples-max", type=str, default="100",
                         help="'none' (default) caps at min_cluster_size per-trial; or a fixed number")

    # search phase
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--search-seeds", type=int, nargs="+", default=[42],
                         help="One or more UMAP seeds, averaged over each trial for a more stable search signal")
    parser.add_argument("--study-seed", type=int, default=42,
                         help="Seed for Optuna's own sampler (reproducibility of which trials get proposed)")
    parser.add_argument("--plateau-window", type=int, default=None,
                         help="How many trailing trials to check for improvement in the convergence "
                              "check (default: max(10, n_trials // 5))")

    # refine phase
    parser.add_argument("--top-k-refine", type=int, default=10)
    parser.add_argument("--refine-seeds", type=int, nargs="+", default=DEFAULT_REFINE_SEEDS)

    # validity constraints (soft penalty during search, hard filter during refine ranking)
    parser.add_argument("--min-clusters", type=int, default=2) 
    parser.add_argument("--max-clusters", type=str, default="none",
                         help="Max clusters to count as valid, or 'none' for no upper limit")
    parser.add_argument("--max-noise-fraction", type=str, default="0.3",
                         help="Max noise fraction to count as valid, or 'none' for no upper limit")

    args = parser.parse_args()

    def parse_none_or(value_str, cast):
        return None if str(value_str).lower() == "none" else cast(value_str)

    args.max_clusters = parse_none_or(args.max_clusters, int)
    args.max_noise_fraction = parse_none_or(args.max_noise_fraction, float)
    args.min_samples_max = parse_none_or(args.min_samples_max, int)
    # the remaining *-max values need data to resolve "none" into a real
    # ceiling - kept as raw strings for now, resolved just below once the
    # input is loaded.

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_features(args.input, args.id_columns)
    X = df[feature_cols].to_numpy()
    n_events, n_features = X.shape

    def resolve_bound(value_str, ceiling, label, cast=int):
        if str(value_str).lower() != "none":
            return cast(value_str)
        print(f"  {label}: 'none' -> using auto ceiling {ceiling}")
        return ceiling

    print("Resolving search-space upper bounds:")
    args.n_neighbors_max = resolve_bound(args.n_neighbors_max, max(args.n_neighbors_min + 1, n_events - 1),
                                          "--n-neighbors-max (can't exceed n_events-1)")
    args.min_dist_max = resolve_bound(args.min_dist_max, 0.99, "--min-dist-max (UMAP requires < 1.0)", cast=float)
    args.n_components_max = resolve_bound(args.n_components_max, max(args.n_components_min, n_features),
                                           "--n-components-max (capped at n_features_in, more rarely helps)")
    args.min_cluster_size_max = resolve_bound(
        args.min_cluster_size_max, max(args.min_cluster_size_min + 1, n_events // 2),
        "--min-cluster-size-max (capped at n_events // 2 - larger clusters than half the data are degenerate)",
    )

    # --- Phase 1: search ---
    print(f"\nStarting Bayesian search: {args.n_trials} trials, search_seeds={args.search_seeds}\n")
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

    save_convergence(trials_df, args.output_dir, args.plateau_window)

    print(f"\nBest single trial: {study.best_params} "
          f"(score={study.best_value:.3f}, n_clusters={study.best_trial.user_attrs['n_clusters']}, "
          f"noise={study.best_trial.user_attrs['noise_fraction']:.1%}, "
          f"trustworthiness={study.best_trial.user_attrs['trustworthiness']:.3f})")

    # --- Phase 2: refine top-K distinct configs across multiple seeds ---
    param_cols = ["n_neighbors", "min_dist", "n_components", "min_cluster_size", "min_samples"]
    ranked_trials = trials_df.sort_values("value", ascending=False)
    dedup = ranked_trials.drop_duplicates(subset=param_cols, keep="first")
    top_configs = dedup.head(args.top_k_refine)[param_cols].to_dict("records")

    print(f"\nRefining top {len(top_configs)} distinct configs across seeds {args.refine_seeds}...\n")
    refined_raw = refine_top_configs(X, top_configs, args.refine_seeds, args.metric, args.trustworthiness_n_neighbors)
    refined_raw_path = args.output_dir / "bayesopt_refined_raw.csv"
    refined_raw.to_csv(refined_raw_path, index=False)
    print(f"\nSaved {refined_raw_path}")

    refined_ranked = aggregate_results(refined_raw, args.min_clusters, args.max_clusters, args.max_noise_fraction)
    refined_ranked_path = args.output_dir / "bayesopt_refined_ranked.csv"
    refined_ranked.to_csv(refined_ranked_path, index=False)
    print(f"Saved {refined_ranked_path}")

    cols_to_show = param_cols + [
        "frac_seeds_valid", "mean_relative_validity", "std_relative_validity",
        "mean_trustworthiness", "std_trustworthiness",
        "mean_n_clusters", "mean_noise_fraction",
    ]
    print("\nRefined ranking (top configs found by search, validated across seeds):\n")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(refined_ranked[cols_to_show].to_string(index=False))

    print("\nNote: relative_validity structurally favors LOWER n_components, "
          "trustworthiness structurally favors HIGHER n_components - weigh both "
          "per n_components rather than picking on relative_validity alone.")
    print("\nNext step: as before, inspect the winning config's actual clusters "
          "(small multiples / boxplots per feature) before committing to it.")


if __name__ == "__main__":
    main()