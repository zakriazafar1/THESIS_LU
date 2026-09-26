#!/usr/bin/env python3
"""
3.4.3_filter_main_cluster.py

Removes events belonging to noise / small outlier cluster(s) - as labeled in
one reference run's embedding_with_clusters.csv - from the scaled feature
matrix, and writes a new feature matrix CSV in the exact same format as the
input. Point 3.3.2_umap_hdbscan_bayes.py / 3.4.1_cluster_inspect.py /
3.4.2_all_cluster_inspect.py at this new file with --input to refit UMAP +
HDBSCAN from scratch on just the remaining "main cluster" population.

Why refit from scratch rather than just deleting rows from an existing
embedding: HDBSCAN re-run on the SAME UMAP coordinates minus the removed
points will almost certainly reproduce the same result for the remaining
points (their nearest neighbors don't change if you remove far-away
outliers). A fresh UMAP fit on the filtered raw feature matrix is the
version that can actually behave differently, since UMAP optimizes the
embedding using the whole point set at once.

Usage:
    # keep only cluster 0 (drop noise + all other clusters) - the common case
    python 3.4.3_filter_main_cluster.py \\
        --embedding "...\\rank01_.../embedding_with_clusters.csv" \\
        --input     "...\\arousal_feature_matrix_scaled_clean.csv" \\
        --output    "...\\arousal_feature_matrix_scaled_clean_mainonly.csv" \\
        --keep-clusters 0

    # explicitly drop noise (-1) and clusters 1 and 2, keep everything else
    python 3.4.3_filter_main_cluster.py \\
        --embedding "...\\embedding_with_clusters.csv" \\
        --input "...\\arousal_feature_matrix_scaled_clean.csv" \\
        --output "...\\arousal_feature_matrix_scaled_clean_mainonly.csv" \\
        --drop-clusters -1 1 2

Pass exactly one of --keep-clusters / --drop-clusters.
"""

import argparse
import csv as csv_module
import sys
from pathlib import Path

import pandas as pd

DEFAULT_KEY_COLUMNS = ["subject_id", "night_id", "event_idx"]


def detect_delimiter(path: Path, sample_lines: int = 5) -> str:
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


def read_flexible_csv(path: Path) -> pd.DataFrame:
    delimiter = detect_delimiter(path)
    decimal = detect_decimal(path, delimiter)
    df = pd.read_csv(path, sep=delimiter, decimal=decimal)
    return df, delimiter, decimal


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding", type=Path, required=True,
                         help="embedding_with_clusters.csv from the reference run whose cluster labels decide what to drop")
    parser.add_argument("--input", type=Path, required=True,
                         help="The full scaled feature matrix CSV to filter (e.g. arousal_feature_matrix_scaled_clean.csv)")
    parser.add_argument("--output", type=Path, required=True,
                         help="Where to write the filtered feature matrix CSV (same format as --input)")
    parser.add_argument("--key-columns", type=str, nargs="*", default=DEFAULT_KEY_COLUMNS,
                         help=f"Columns that uniquely identify an event, present in both files (default: {DEFAULT_KEY_COLUMNS})")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keep-clusters", type=int, nargs="+", help="Cluster label(s) to KEEP (e.g. 0)")
    group.add_argument("--drop-clusters", type=int, nargs="+", help="Cluster label(s) to DROP (e.g. -1 1 2). Use -1 for noise.")
    args = parser.parse_args()

    emb_df, _, _ = read_flexible_csv(args.embedding)
    if "cluster" not in emb_df.columns:
        sys.exit(f"'cluster' column not found in {args.embedding}. Columns present: {list(emb_df.columns)}")

    missing_keys = [c for c in args.key_columns if c not in emb_df.columns]
    if missing_keys:
        sys.exit(f"Key column(s) {missing_keys} not found in {args.embedding}. "
                  f"Columns present: {list(emb_df.columns)}. Pass --key-columns to override.")

    print("Cluster sizes in reference embedding:")
    print(emb_df["cluster"].value_counts().sort_index().to_string())

    if args.keep_clusters is not None:
        keep_ids = set(emb_df.loc[emb_df["cluster"].isin(args.keep_clusters), tuple(args.key_columns)]
                        .apply(tuple, axis=1))
        print(f"\nKeeping cluster(s) {args.keep_clusters}: {len(keep_ids)} events")
    else:
        keep_mask = ~emb_df["cluster"].isin(args.drop_clusters)
        keep_ids = set(emb_df.loc[keep_mask, tuple(args.key_columns)].apply(tuple, axis=1))
        dropped_n = len(emb_df) - len(keep_ids)
        print(f"\nDropping cluster(s) {args.drop_clusters}: removing {dropped_n} events, keeping {len(keep_ids)}")

    feat_df, delimiter, decimal = read_flexible_csv(args.input)
    missing_keys_input = [c for c in args.key_columns if c not in feat_df.columns]
    if missing_keys_input:
        sys.exit(f"Key column(s) {missing_keys_input} not found in {args.input}. "
                  f"Columns present: {list(feat_df.columns)}. Pass --key-columns to override.")

    row_keys = feat_df[args.key_columns].apply(tuple, axis=1)
    mask = row_keys.isin(keep_ids)
    n_before = len(feat_df)
    filtered = feat_df.loc[mask].reset_index(drop=True)
    n_after = len(filtered)

    matched_from_keep = mask.sum()
    print(f"\n{args.input.name}: {n_before} events total")
    print(f"Matched against reference embedding and kept: {n_after} events ({100 * n_after / n_before:.1f}%)")
    if matched_from_keep < len(keep_ids):
        print(f"NOTE: {len(keep_ids) - matched_from_keep} event(s) from the reference embedding's keep-set "
              f"were not found in --input by key columns {args.key_columns} - check they really uniquely "
              f"identify events in both files.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, sep=delimiter, decimal=decimal, index=False)
    print(f"\nSaved {args.output}")
    print("Point 3.3.2_umap_hdbscan_bayes.py (and 3.4.x) at this file with --input "
          "to refit UMAP + HDBSCAN from scratch on just this subset.")


if __name__ == "__main__":
    main()