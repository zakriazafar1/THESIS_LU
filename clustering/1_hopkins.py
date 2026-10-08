"""
1_hopkins.py

Hopkins statistic on the REDUCED scaled feature matrix (8 features, one per
correlation group).

Tests whether the feature matrix has ANY inherent clustering tendency at all.
If GMM/HDBSCAN keep returning one component/one blob, the Hopkins statistic
tells you whether that is because the data itself has no cluster structure
(H close to 0.5), rather than because the right parameters haven't been found.

METHOD (Lawson & Jurs, 1990; Banerjee & Dave, 2004):
For each of --n-repeats repeats:
    1. Draw a random SAMPLE of m real data points from the feature matrix.
    2. Generate m SYNTHETIC points, drawn uniformly at random from the same
       per-feature min/max range as the real data.
    3. For each real sampled point, find its nearest-neighbor distance to
       another REAL point (w_i).
    4. For each synthetic point, find its nearest-neighbor distance to the
       nearest REAL point (u_i).
    5. H = sum(u_i) / (sum(u_i) + sum(w_i))

INTERPRETATION:
    H ~ 0.5   -> no clustering tendency (resembles uniform random noise)
    H > ~0.75 -> evidence of real clustering tendency
    H ~ 0     -> rare in practice; regularly/evenly spaced data

CAVEAT: 
  - Uniform sampling within the bounding box is sensitive to outliers
    (a few extreme z-scores stretch the box and inflate u_i, pushing H up).
    Heavy-tailed or skewed data can therefore give H > 0.5 without real
    clusters. Use --trim-quantile to check robustness.

Usage:
    python 1_hopkins.py
    python 1_hopkins.py --input "...\\arousal_feature_matrix_reduced_scaled.csv"
    python 1_hopkins.py --trim-quantile 0.01

Output (in --output-dir):
    hopkins_reduced_repeats.csv  - one row per repeat
    hopkins_reduced_summary.csv  - mean/std/min/max H, n_events, n_features,
                                   feature names, interpretation
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
        "    pip install scipy"
    )

DEFAULT_INPUT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\3. feature selection\reduced feature matrix\arousal_features_reduced_scaled.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\4. clustering\1. hopkins_reduced"
)

OUTPUT_PREFIX = "hopkins_reduced"

DEFAULT_ID_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec", "duration_sec", "sec_prev_event",
]

EXPECTED_N_FEATURES = 8

# ---- loading helpers ----------

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

    comma_decimal = re.compile(r"^-?\d+,\d+([eE][-+]?\d+)?$")
    dot_decimal = re.compile(r"^-?\d+\.\d+([eE][-+]?\d+)?$")
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
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n  {path}\n"
            f"Pass the correct reduced scaled matrix with --input."
        )
    delimiter = detect_delimiter(path)
    decimal = detect_decimal(path, delimiter)
    df = pd.read_csv(path, sep=delimiter, decimal=decimal)
    print(f"Detected delimiter='{delimiter}', decimal='{decimal}'")

    # Drop an unnamed index column if one was written by pandas upstream.
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

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
    print("Features: " + ", ".join(feature_cols))
    if len(feature_cols) != EXPECTED_N_FEATURES:
        print(f"WARNING: expected {EXPECTED_N_FEATURES} features for the reduced set, "
              f"got {len(feature_cols)}. Check --input / --id-columns.")
    return df, feature_cols


# ---- Hopkins statistic -----------------------------------------------------------

def hopkins_statistic(X: np.ndarray, sample_size: int, rng: np.random.Generator,
                      lo: np.ndarray, hi: np.ndarray) -> float:
    """One draw of H. Synthetic points are uniform within [lo, hi] per feature."""
    n, d = X.shape
    tree = cKDTree(X)

    # w_i: nearest OTHER real point (k=2; index 0 is the point itself).
    real_idx = rng.choice(n, size=sample_size, replace=False)
    dists, _ = tree.query(X[real_idx], k=2)
    w = dists[:, 1]

    # u_i: nearest real point to each synthetic uniform point.
    synthetic = rng.uniform(low=lo, high=hi, size=(sample_size, d))
    u, _ = tree.query(synthetic, k=1)

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
                        help="Fraction of events used as sample size m per repeat (default 0.05)")
    parser.add_argument("--min-sample-size", type=int, default=30)
    parser.add_argument("--max-sample-size", type=int, default=500)
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--trim-quantile", type=float, default=0.0,
                        help="If >0, draw synthetic points within the [q, 1-q] quantile range "
                             "per feature instead of min/max (robustness check against outliers)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_features(args.input, args.id_columns)
    X = df[feature_cols].to_numpy(dtype=float)
    n_events, n_features = X.shape

    if args.trim_quantile > 0:
        lo = np.quantile(X, args.trim_quantile, axis=0)
        hi = np.quantile(X, 1 - args.trim_quantile, axis=0)
        box = f"[{args.trim_quantile:g}, {1 - args.trim_quantile:g}] quantiles"
    else:
        lo, hi = X.min(axis=0), X.max(axis=0)
        box = "min/max"

    sample_size = int(round(n_events * args.sample_frac))
    sample_size = max(args.min_sample_size, min(sample_size, args.max_sample_size, n_events - 1))
    print(f"\nUsing sample size m={sample_size} per repeat "
          f"({100 * sample_size / n_events:.1f}% of events), {args.n_repeats} repeats, "
          f"synthetic box = {box}")

    rng = np.random.default_rng(args.random_state)
    h_values = np.array([hopkins_statistic(X, sample_size, rng, lo, hi)
                         for _ in range(args.n_repeats)])
    for i, h in enumerate(h_values, 1):
        print(f"  repeat {i}/{args.n_repeats}: H = {h:.4f}")

    h_mean, h_std = float(h_values.mean()), float(h_values.std())
    h_min, h_max = float(h_values.min()), float(h_values.max())
    interpretation = interpret_hopkins(h_mean)

    suffix = f"_trim{args.trim_quantile:g}" if args.trim_quantile > 0 else ""

    repeats_path = args.output_dir / f"{OUTPUT_PREFIX}{suffix}_repeats.csv"
    pd.DataFrame({"repeat": np.arange(1, args.n_repeats + 1),
                  "hopkins_statistic": h_values}).to_csv(repeats_path, index=False)
    print(f"\nSaved {repeats_path}")

    summary_path = args.output_dir / f"{OUTPUT_PREFIX}{suffix}_summary.csv"
    pd.DataFrame([{
        "input_file": args.input.name,
        "n_events": n_events,
        "n_features": n_features,
        "features": ";".join(feature_cols),
        "synthetic_box": box,
        "sample_size_m": sample_size,
        "n_repeats": args.n_repeats,
        "mean_hopkins": h_mean,
        "std_hopkins": h_std,
        "min_hopkins": h_min,
        "max_hopkins": h_max,
        "interpretation": interpretation,
    }]).to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    print(f"\nHopkins statistic: {h_mean:.4f} +/- {h_std:.4f} "
          f"(range {h_min:.4f}-{h_max:.4f} across {args.n_repeats} repeats)")
    print(f"Interpretation: {interpretation}")
    print("\nNote: skewed/heavy-tailed data can push H above 0.5 without real clusters. "
          "Compare against the Gaussian-copula null (3_nullmodel.py) and/or rerun "
          "with --trim-quantile 0.01 before concluding.")


if __name__ == "__main__":
    main()