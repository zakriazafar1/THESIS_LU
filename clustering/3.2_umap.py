#!/usr/bin/env python3
"""
3.2_umap.py

Step 2 of the arousal-subtyping pipeline: reduce the 36 scaled EEG features
(output of preprocess_features.py) down to a low-dimensional embedding with
UMAP, tuned to preserve broad/global brain-state structure rather than tiny
noisy micro-clusters, and to pack dense regions tightly so HDBSCAN can find
clean cluster boundaries downstream.

Parameter choices:
    n_neighbors : set HIGHER than UMAP's default (15). Larger n_neighbors
                  forces UMAP to balance local vs. global structure using a
                  bigger neighborhood, which smooths out small idiosyncratic
                  groupings and emphasizes the broader manifold. Default here
                  is 50; treat it as a knob to sweep (try 30-100 depending on
                  n_events).
    min_dist    : set NEAR 0.0 (default 0.0). This lets UMAP place points in
                  dense regions very close together in the embedding, which
                  is what HDBSCAN needs to detect density-based boundaries -
                  a large min_dist artificially spreads points out and can
                  blur genuine density gaps.
    n_components: swept over 2-5 (config below) since the "right" number of
                  dimensions for HDBSCAN is itself an empirical question -
                  too few can collapse real structure, too many re-introduces
                  the curse of dimensionality that UMAP is meant to fix.

Usage:
    python 3.2_umap.py

    Input and output paths are hardcoded as defaults below (DEFAULT_INPUT /
    DEFAULT_OUTPUT_DIR) so it runs with no arguments. Both can still be
    overridden on the command line if needed, e.g.:
        python 3.2_umap.py --input other_features.csv --output-dir other_out

Input:
    A CSV where each row is one arousal event and columns are the 36 scaled
    features from preprocess_features.py. Non-feature identifier columns
    (e.g. participant_id, night_id, event_id) are auto-detected by dtype/name
    and carried through untouched rather than fed into UMAP.

Output (per n_components value):
    umap_output/embedding_{n}d.csv        - identifier columns + UMAP_1..UMAP_n
    umap_output/embedding_{n}d.npy        - raw embedding array
    umap_output/correlation_{n}d.csv      - Pearson r of each diagnostic feature
                                             vs. each UMAP axis, for that run
    umap_output/umap_2d_scatter.png       - diagnostic scatter (only for n=2)
    umap_output/umap_2d_small_multiples.png - 2D scatter colored by each
                                             diagnostic feature, one panel per
                                             feature (only for n=2)
    umap_output/run_summary.json          - parameters + basic diagnostics per run
"""

import argparse
import json
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

# Hardcoded default paths - edit here if they ever move, or override via CLI.
DEFAULT_INPUT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
    r"\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled\arousal_feature_matrix_scaled.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
    r"\THESIS_OUTPUTS\PROJECT 2\3. feature selection\umap"
)

# Columns that are identifiers/metadata, not features - carried through as-is
# and excluded from the UMAP input. Adjust to match your actual ID columns.
DEFAULT_ID_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec", "duration_sec", "sec_prev_event",
]

# Features to inspect the embedding against: small-multiples panels + a
# correlation table. These are read straight from the input CSV (df), so
# they work whether or not they're also part of the UMAP input itself -
# e.g. duration_sec is an ID column here but still worth checking against
# the embedding it didn't influence.
DIAGNOSTIC_FEATURES = [
    "duration_sec", "mean_alpha_ratio", "mean_theta_ratio", "mean_beta_ratio",
    "mean_sigma_ratio", "mean_delta_ratio", "oxy_amp_ratio", "motion_rms",
]


def detect_delimiter(path: Path, sample_lines: int = 5) -> str:
    """Sniff whether the CSV uses ',' or ';' as the field delimiter."""
    import csv as csv_module

    with open(path, "r", encoding="utf-8-sig") as f:
        sample = "".join(f.readline() for _ in range(sample_lines))
    try:
        dialect = csv_module.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except csv_module.Error:
        return ","


def detect_decimal(path: Path, delimiter: str, sample_lines: int = 20) -> str:
    """Inspect a few data rows to see whether floats use ',' or '.' as the
    decimal separator. Delimiter and decimal separator are independent - a
    ';'-delimited file can still use '.' decimals (e.g. if it was written by
    pandas/numpy already, as opposed to exported from Excel in a Dutch
    locale), so this must not be inferred from the delimiter alone.
    """
    import re

    comma_decimal = re.compile(r"^-?\d+,\d+$")
    dot_decimal = re.compile(r"^-?\d+\.\d+$")
    comma_hits, dot_hits = 0, 0

    with open(path, "r", encoding="utf-8-sig") as f:
        next(f, None)  # skip header
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
            f"{non_numeric}. Add them to --id-columns if they're identifiers, "
            f"or fix upstream encoding otherwise."
        )

    if df[feature_cols].isna().any().any():
        n_nan = df[feature_cols].isna().sum().sum()
        raise ValueError(
            f"Found {n_nan} NaNs in the feature matrix. UMAP can't handle "
            f"missing values - impute or drop them upstream (preprocess_features.py)."
        )

    print(f"Loaded {len(df)} events x {len(feature_cols)} features")
    print(f"  ID columns carried through: {present_id_cols}")
    if len(feature_cols) != 36:
        print(f"  NOTE: expected 36 features, found {len(feature_cols)} - "
              f"double check --id-columns if this is unexpected.")

    return df, present_id_cols, feature_cols


def run_umap(X, n_components, n_neighbors, min_dist, metric, random_state):
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=1,  # required for reproducibility when random_state is set
    )
    embedding = reducer.fit_transform(X)
    return embedding, reducer


def make_2d_plot(embedding, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=6, alpha=0.5, linewidths=0)
    ax.set_xlabel("UMAP_1")
    ax.set_ylabel("UMAP_2")
    ax.set_title("UMAP embedding (2D) of scaled arousal features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_small_multiples(df, embedding, features, out_path):
    """One panel per diagnostic feature: the same 2D UMAP scatter, colored
    by that feature's (already-scaled) value. Only meaningful for a 2D
    embedding - uses the first two axes as x/y.
    """
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  NOTE: diagnostic features not found in input, skipping: {missing}")
    if not present:
        print("  NOTE: no diagnostic features found - skipping small multiples.")
        return

    n = len(present)
    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, present):
        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=df[feature], cmap="viridis", s=6, alpha=0.6, linewidths=0,
        )
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("UMAP_1")
        ax.set_ylabel("UMAP_2")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(present):]:
        ax.axis("off")

    fig.suptitle("UMAP embedding (2D) colored by diagnostic features", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_correlation_table(df, embedding, features, n_components):
    """Pearson correlation of each diagnostic feature against each UMAP axis
    for one embedding run. Rows = features, columns = UMAP_1..UMAP_n.
    """
    present = [f for f in features if f in df.columns]
    umap_cols = [f"UMAP_{i + 1}" for i in range(n_components)]
    emb_df = pd.DataFrame(embedding, columns=umap_cols, index=df.index)
    combined = pd.concat([df[present], emb_df], axis=1)
    corr = combined.corr().loc[present, umap_cols]
    return corr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help="CSV of scaled features (output of preprocess_features.py)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-neighbors", type=int, default=50,
                         help="Higher = more global structure, less sensitivity to micro-clusters (default: 50)")
    parser.add_argument("--min-dist", type=float, default=0.0,
                         help="Keep near 0.0 so dense regions pack tightly for HDBSCAN (default: 0.0)")
    parser.add_argument("--n-components", type=int, nargs="+", default=[2, 3, 4, 5],
                         help="One or more target dimensionalities to produce (default: 2 3 4 5)")
    parser.add_argument("--metric", type=str, default="euclidean",
                         help="Distance metric for UMAP (default: euclidean, since input is already RobustScaler-standardized)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--id-columns", type=str, nargs="*", default=DEFAULT_ID_COLUMNS)
    parser.add_argument("--diagnostic-features", type=str, nargs="*", default=DIAGNOSTIC_FEATURES,
                         help="Features to check the embedding against: small multiples + correlation table")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, id_cols, feature_cols = load_features(args.input, args.id_columns)
    X = df[feature_cols].to_numpy()

    summary = {
        "input": str(args.input),
        "n_events": len(df),
        "n_features_in": len(feature_cols),
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "random_state": args.random_state,
        "runs": [],
    }

    for n_comp in args.n_components:
        print(f"\nRunning UMAP: n_components={n_comp}, n_neighbors={args.n_neighbors}, min_dist={args.min_dist}")
        embedding, reducer = run_umap(
            X, n_comp, args.n_neighbors, args.min_dist, args.metric, args.random_state
        )

        emb_df = df[id_cols].copy()
        for i in range(n_comp):
            emb_df[f"UMAP_{i + 1}"] = embedding[:, i]

        csv_path = args.output_dir / f"embedding_{n_comp}d.csv"
        npy_path = args.output_dir / f"embedding_{n_comp}d.npy"
        emb_df.to_csv(csv_path, index=False)
        np.save(npy_path, embedding)
        print(f"  Saved {csv_path}")
        print(f"  Saved {npy_path}")

        corr = compute_correlation_table(df, embedding, args.diagnostic_features, n_comp)
        corr_path = args.output_dir / f"correlation_{n_comp}d.csv"
        corr.to_csv(corr_path)
        print(f"  Saved {corr_path}")

        if n_comp == 2:
            plot_path = args.output_dir / "umap_2d_scatter.png"
            make_2d_plot(embedding, plot_path)
            print(f"  Saved {plot_path}")

            multiples_path = args.output_dir / "umap_2d_small_multiples.png"
            make_small_multiples(df, embedding, args.diagnostic_features, multiples_path)
            print(f"  Saved {multiples_path}")

        summary["runs"].append({
            "n_components": n_comp,
            "embedding_min": embedding.min(axis=0).tolist(),
            "embedding_max": embedding.max(axis=0).tolist(),
        })

    summary_path = args.output_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {summary_path}")
    print("\nNext step: feed one of these embeddings into HDBSCAN. Start with "
          "the lowest n_components that still looks structured in the 2D plot - "
          "more dimensions isn't automatically better once you're already past "
          "UMAP's noise reduction.")


if __name__ == "__main__":
    main()