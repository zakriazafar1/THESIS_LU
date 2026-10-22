"""
=============================================================================
4.1_hdbscan_descriptives.py

Descriptive statistics of the HDBSCAN clusters on the ORIGINAL (untransformed,
unscaled) features.
Because the features are skewed, each feature is summarised as median 
(interquartile range, Q1–Q3). Per group the number of events (and %) 
and the number of participants are reported. 
Groups: each cluster, noise (-1, if present) and all events.

Input per run (tag):
  <HDBSCAN_DIR>/<tag>/hdbscan_final_labels.csv   (output of 4_hdbscan.py)
  ORIGINAL_INPUT                                (arousal_features_reduced_original.csv)
Linked on subject_id + night_id + event_idx.

Default runs: whitened, whitened_single, reduced_features, reduced_features_single.
Runs without a labels file are skipped with a message.

Output per run in OUTPUT_DIR/<tag>:
  cluster_descriptives_table.xlsx / .csv  -- compact table: rows = features,
                                             columns = groups; "Mdn (Q1–Q3)"
  cluster_descriptives_long.csv           -- the same numbers unrounded, one row
                                             per group x feature (for checking)
  table_note.txt                          -- note for which table 

Usage:
  python 4.1_hdbscan_descriptives.py            # all four runs
  python 4.1_hdbscan_descriptives.py --tags whitened_single  # one or more runs
  python 4.1_hdbscan_descriptives.py --tags my_run --labels path\\to\\hdbscan_final_labels.csv
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2"
)
HDBSCAN_DIR = BASE / r"4. clustering\4. hdbscan"
ORIGINAL_INPUT = BASE / r"reduced feature matrix\arousal_features_reduced_original.csv"
OUTPUT_DIR = BASE / r"6. cluster descriptives\hdbscan"

DEFAULT_TAGS = ["whitened", "whitened_single", "reduced_features", "reduced_features_single"]

EVENT_KEY = ["subject_id", "night_id", "event_idx"]

# Features in table order
FEATURES = {
    "duration_sec": "Duration (s)",
    "mean_delta_ratio": "Mean delta ratio",
    "mean_theta_ratio": "Mean theta ratio",
    "mean_alpha_ratio": "Mean alpha ratio",
    "mean_sigma_ratio": "Mean sigma ratio",
    "mean_beta_ratio": "Mean beta ratio",
    "oxy_amp_ratio": "PPG ratio",
    "motion_rms": "Head movement RMS",
}
DECIMALS = 2

# =============================================================================
# LOADING
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    """CSV with automatic delimiter detection and fallback to decimal=','."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    df = pd.read_csv(path, sep=None, engine="python")
    text_cols = {"subject_id", "group", "night_id"}
    check = [c for c in df.columns if c not in text_cols]
    n_bad = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in check)
    if n_bad > 0:
        df2 = pd.read_csv(path, sep=None, engine="python", decimal=",")
        n_bad2 = sum(not pd.api.types.is_numeric_dtype(df2[c]) for c in check if c in df2.columns)
        if n_bad2 < n_bad:
            print(f"[NOTE] decimal comma detected in {path.name}, re-read with decimal=','.")
            df = df2
    return df


def event_key(df: pd.DataFrame) -> pd.Series:
    return (df["subject_id"].astype(str).str.strip() + "|"
            + df["night_id"].astype(str).str.strip() + "|"
            + df["event_idx"].astype(int).astype(str))


def link_labels(labels: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Attach each event's cluster label to the original features; stop if events don't match."""
    for name, d in (("labels", labels), ("original", original)):
        missing = [c for c in EVENT_KEY if c not in d.columns]
        if missing:
            raise KeyError(f"{name} file is missing key column(s) {missing}.")
    lab = pd.DataFrame({"key": event_key(labels), "cluster": labels["cluster"].astype(int)})
    orig = original.copy()
    orig["key"] = event_key(orig)

    if lab["key"].duplicated().any() or orig["key"].duplicated().any():
        raise ValueError("Duplicate events (subject_id + night_id + event_idx) -- cannot link uniquely.")

    merged = orig.merge(lab, on="key", how="inner")
    n_lab, n_orig, n_m = len(lab), len(orig), len(merged)
    if n_m != n_lab:
        raise ValueError(f"Only {n_m} of {n_lab} labelled events were found in the original "
                         "feature matrix. Were both made from the same (filtered) data?")
    if n_m != n_orig:
        print(f"[NOTE] the original matrix has {n_orig} events; {n_orig - n_m} have no cluster "
              "label and are left out.")
    print(f"Linked {n_m} events to the original features.")
    return merged.drop(columns="key")


# =============================================================================
# DESCRIPTIVES
# =============================================================================

def groups_in_order(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Clusters (by number), then noise (if any), then all events."""
    out = [(f"Cluster {c}", df[df["cluster"] == c])
           for c in sorted(df["cluster"].unique()) if c != -1]
    if (df["cluster"] == -1).any():
        out.append(("Noise", df[df["cluster"] == -1]))
    out.append(("All events", df))
    return out


def describe(df: pd.DataFrame, features: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long table (unrounded) + wide table."""
    n_total = len(df)
    long_rows, wide = [], {}
    for gname, g in groups_in_order(df):
        col = {
            "n events (%)": f"{len(g):,} ({100 * len(g) / n_total:.1f}%)",
            "n participants": f"{g['subject_id'].nunique()}",
        }
        for feat, label in features.items():
            v = g[feat].replace([np.inf, -np.inf], np.nan).dropna()
            q1, med, q3 = (np.percentile(v, [25, 50, 75]) if len(v) else (np.nan,) * 3)
            long_rows.append({"group": gname, "feature": feat, "label": label, "n": len(v),
                              "median": med, "q1": q1, "q3": q3, "iqr": q3 - q1})
            col[label] = (f"{med:.{DECIMALS}f} ({q1:.{DECIMALS}f}\u2013{q3:.{DECIMALS}f})"
                          if len(v) else "")
        wide[gname] = col
    wide_df = pd.DataFrame(wide)
    wide_df.index.name = "Variable"
    return pd.DataFrame(long_rows), wide_df


def table_note(df: pd.DataFrame, tag: str) -> str:
    n_clusters = int((df["cluster"].unique() != -1).sum())
    return (
        f"Note. Descriptive statistics of the HDBSCAN solution ({tag}) on the original, "
        f"untransformed features (N = {len(df):,} events from {df['subject_id'].nunique()} "
        f"participants; {n_clusters} cluster(s)). Values are median (interquartile range, Q1\u2013Q3). "
        "Power ratios are event power relative to the baseline (1 = baseline level). "
        "Noise = events not assigned to any cluster. No inferential tests were performed, "
        "because the clusters were derived from the same data."
    )


# =============================================================================
# ONE RUN
# =============================================================================

def run_one(tag: str, labels_path: Path, original: pd.DataFrame, out_root: Path) -> None:
    print("\n" + "=" * 70)
    print(f"RUN: {tag}")
    print("=" * 70)
    if not labels_path.exists():
        print(f"[SKIPPED] no labels file: {labels_path}")
        return
    labels = read_csv_robust(labels_path)
    df = link_labels(labels, original)

    features = {f: l for f, l in FEATURES.items() if f in df.columns}
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"[NOTE] features not found in the original matrix, left out: {missing}")

    long_df, wide_df = describe(df, features)

    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_dir / "cluster_descriptives_long.csv", index=False)
    wide_df.to_csv(out_dir / "cluster_descriptives_table.csv", encoding="utf-8-sig")
    try:
        wide_df.to_excel(out_dir / "cluster_descriptives_table.xlsx", sheet_name=tag[:31])
        xlsx = "cluster_descriptives_table.xlsx / "
    except ImportError:
        xlsx = ""
        print("[NOTE] openpyxl not installed -- no .xlsx written (pip install openpyxl).")
    note = table_note(df, tag)
    (out_dir / "table_note.txt").write_text(note, encoding="utf-8")

    with pd.option_context("display.width", 220, "display.max_columns", None,
                           "display.max_colwidth", 30):
        print(wide_df.to_string())
    print(f"\n{note}")
    print(f"\nSaved in {out_dir}: {xlsx}cluster_descriptives_table.csv, "
          "cluster_descriptives_long.csv, table_note.txt")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS,
                        help="Run(s) to describe; each reads <hdbscan-dir>/<tag>/hdbscan_final_labels.csv")
    parser.add_argument("--labels", type=Path, default=None,
                        help="Optional: explicit labels file (only with a single --tags value)")
    parser.add_argument("--hdbscan-dir", type=Path, default=HDBSCAN_DIR)
    parser.add_argument("--original", type=Path, default=ORIGINAL_INPUT,
                        help="Original (untransformed) reduced feature matrix")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.labels is not None and len(args.tags) != 1:
        parser.error("--labels can only be used with exactly one --tags value.")

    original = read_csv_robust(args.original)
    print(f"Original feature matrix loaded: {args.original}  shape={original.shape}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for tag in args.tags:
        labels_path = args.labels or (args.hdbscan_dir / tag / "hdbscan_final_labels.csv")
        run_one(tag, labels_path, original, args.output_dir)

    print(f"\nAll output saved under: {args.output_dir}")


if __name__ == "__main__":
    main()