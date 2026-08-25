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
    umap_output/embedding_{n}d.csv   - identifier columns + UMAP_1..UMAP_n
    umap_output/embedding_{n}d.npy   - raw embedding array
    umap_output/umap_2d_scatter.png  - diagnostic scatter (only for n=2)
    umap_output/run_summary.json     - parameters + basic diagnostics per run
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
    "participant_id", "night_id", "event_id", "arousal_id",
    "onset_time", "onset_sample", "label", "subtype",
]


def load_features(path: Path, id_columns):
    df = pd.read_csv(path)
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

        if n_comp == 2:
            plot_path = args.output_dir / "umap_2d_scatter.png"
            make_2d_plot(embedding, plot_path)
            print(f"  Saved {plot_path}")

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