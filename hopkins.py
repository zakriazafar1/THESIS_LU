#!/usr/bin/env python3
"""
3.4.4_hopkins_statistic.py

Tests whether your feature matrix has ANY inherent clustering tendency at
all - a formal, quantitative check to run alongside (or instead of trusting
blindly) what HDBSCAN found. If HDBSCAN + UMAP keep returning one giant
cluster plus a sliver of outliers no matter the parameters, the Hopkins
statistic tells you whether that is because the data itself has no cluster
structure (H close to 0.5), rather than because you haven't found the right
parameters yet.

METHOD (Lawson & Jurs, 1990; Banerjee & Dave, 2004):
For each of --n-repeats repeats:
    1. Draw a random SAMPLE of m real data points from your feature matrix.
    2. Generate m SYNTHETIC points, drawn uniformly at random from the same
       per-feature min/max range as your real data.
    3. For each real sampled point, find its nearest-neighbor distance to
       another REAL point (w_i).
    4. For each synthetic point, find its nearest-neighbor distance to the
       nearest REAL point (u_i).
    5. H = sum(u_i) / (sum(u_i) + sum(w_i))

INTERPRETATION:
    H ~ 0.5   -> no clustering tendency; your data looks statistically like
                 uniform random noise in this feature space (this is the
                 result consistent with HDBSCAN finding one big blob)
    H > ~0.75 -> evidence of real clustering tendency
    H ~ 0     -> rare in practice; suggests regularly/evenly spaced data

A single draw of random samples is noisy, so this script repeats the whole
calculation --n-repeats times with fresh random samples each time and
reports the mean +/- std across repeats, rather than a single number.

CAVEAT worth keeping in your methods section: nearest-neighbor distances
become less discriminative as dimensionality grows (curse of
dimensionality). With ~27 features this is a real but not disqualifying
concern - it's part of why this script is one piece of evidence, meant to
sit alongside the gap statistic / silhouette curve and a VAT plot, not to
stand alone.

This uses the SAME feature-matrix loading conventions (delimiter/decimal
auto-detection, --id-columns) as 3.3.2_umap_hdbscan_bayes.py and
3.4.1_cluster_inspect.py, so point --input at the exact same scaled feature
matrix file you used for those, to test clustering tendency on the identical
data HDBSCAN saw.

Usage:
    python 3.4.4_hopkins_statistic.py
    python 3.4.4_hopkins_statistic.py --input "...\arousal_feature_matrix_scaled_clean.csv"
    python 3.4.4_hopkins_statistic.py --sample-frac 0.05 --n-repeats 50

Output:
    hopkins_statistic_repeats.csv  - one row per repeat: H value, sample size used
    hopkins_statistic_summary.csv  - one row: mean/std/min/max H across repeats,
                                      n_events, n_features, sample size, plain-
                                      language interpretation
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except ImportError:
    sys.exit(
        "scipy is not installed. Install it with:\n"
        "    pip install scipy --break-system-packages"
    )

DEFAULT_INPUT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2" 
    r"\2. preprocessing\scaled\arousal_feature_matrix_scaled.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
    r"\THESIS_OUTPUTS\PROJECT 2\4. clustering"
)

DEFAULT_ID_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec", "duration_sec", "sec_prev_event",
]


# ---- shared loading helpers (same as 3.3.2 / 3.4.1, kept identical on purpose) ---

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


# ---- Hopkins statistic -----------------------------------------------------------

def hopkins_statistic(X: np.ndarray, sample_size: int, rng: np.random.Generator) -> float:
    """One draw of the Hopkins statistic for data matrix X (n_samples x n_features).

    H = sum(u_i) / (sum(u_i) + sum(w_i))
      u_i: nearest-real-neighbor distance for each of `sample_size` synthetic
           points drawn uniformly at random within X's per-feature min/max range
      w_i: nearest-OTHER-real-neighbor distance for each of `sample_size`
           real points randomly sampled (without replacement) from X
    """
    n, d = X.shape
    tree = cKDTree(X)

    # w_i: sample real points, find distance to nearest OTHER real point.
    real_idx = rng.choice(n, size=sample_size, replace=False)
    real_sample = X[real_idx]
    # k=2 because the nearest neighbor of a real point, in a tree built on
    # the full data, is itself (distance 0) at index 0 - we want index 1.
    dists, _ = tree.query(real_sample, k=2)
    w = dists[:, 1]

    # u_i: synthetic points drawn uniformly within the real data's bounding box.
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    synthetic = rng.uniform(low=mins, high=maxs, size=(sample_size, d))
    dists, _ = tree.query(synthetic, k=1)
    u = dists

    return float(np.sum(u) / (np.sum(u) + np.sum(w)))


def interpret_hopkins(h_mean: float) -> str:
    if h_mean >= 0.90:
        return "very strong clustering tendency"
    elif h_mean >= 0.75:
        return "clustering tendency present"
    elif h_mean >= 0.60:
        return "weak/ambiguous clustering tendency"
    elif h_mean >= 0.40:
        return "no clustering tendency (data resembles uniform random noise in this feature space)"
    else:
        return "regularly/evenly spaced data (unusual; rare in practice)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--id-columns", type=str, nargs="*", default=DEFAULT_ID_COLUMNS)
    parser.add_argument("--sample-frac", type=float, default=0.05,
                         help="Fraction of events to use as the sample size m per repeat (default 0.05 = 5%%)")
    parser.add_argument("--min-sample-size", type=int, default=30,
                         help="Floor on sample size m, regardless of --sample-frac (default 30)")
    parser.add_argument("--max-sample-size", type=int, default=500,
                         help="Ceiling on sample size m, to keep repeats fast on large datasets (default 500)")
    parser.add_argument("--n-repeats", type=int, default=30,
                         help="Number of independent random draws to average over (default 30)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_features(args.input, args.id_columns)
    X = df[feature_cols].to_numpy()
    n_events, n_features = X.shape

    sample_size = int(round(n_events * args.sample_frac))
    sample_size = max(args.min_sample_size, min(sample_size, args.max_sample_size, n_events - 1))
    print(f"\nUsing sample size m={sample_size} per repeat "
          f"({sample_size}/{n_events} = {100*sample_size/n_events:.1f}% of events), "
          f"{args.n_repeats} repeats")

    rng = np.random.default_rng(args.random_state)
    h_values = []
    for i in range(args.n_repeats):
        h = hopkins_statistic(X, sample_size, rng)
        h_values.append(h)
        print(f"  repeat {i+1}/{args.n_repeats}: H = {h:.4f}")

    h_values = np.array(h_values)
    h_mean, h_std = float(h_values.mean()), float(h_values.std())
    h_min, h_max = float(h_values.min()), float(h_values.max())
    interpretation = interpret_hopkins(h_mean)

    repeats_df = pd.DataFrame({"repeat": np.arange(1, args.n_repeats + 1), "hopkins_statistic": h_values})
    repeats_path = args.output_dir / "hopkins_statistic_repeats.csv"
    repeats_df.to_csv(repeats_path, index=False)
    print(f"\nSaved {repeats_path}")

    summary_df = pd.DataFrame([{
        "n_events": n_events,
        "n_features": n_features,
        "sample_size_m": sample_size,
        "n_repeats": args.n_repeats,
        "mean_hopkins": h_mean,
        "std_hopkins": h_std,
        "min_hopkins": h_min,
        "max_hopkins": h_max,
        "interpretation": interpretation,
    }])
    summary_path = args.output_dir / "hopkins_statistic_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    print(f"\nHopkins statistic: {h_mean:.4f} +/- {h_std:.4f} "
          f"(range {h_min:.4f}-{h_max:.4f} across {args.n_repeats} repeats)")
    print(f"Interpretation: {interpretation}")
    print("\nReminder: H ~ 0.5 means the data has no more clustering tendency than "
          "uniform random noise in this feature space - consistent with HDBSCAN "
          "repeatedly returning one giant cluster regardless of parameters. "
          "Report this alongside the gap statistic/silhouette curve and a VAT plot "
          "rather than on its own.")


if __name__ == "__main__":
    main()