#!/usr/bin/env python3
"""
2.3_clean_feature_matrix_nans.py

Standalone diagnostic for arousal_feature_matrix_scaled.csv - reports which
rows/columns actually contain NaN feature values. 

This does NOT modify the original file. It only reports, and optionally
writes a cleaned COPY (with NaN rows dropped) to a new file, so you keep
the original untouched.

Usage:
    # just report, don't write anything
    python 2.3_clean_feature_matrix_nans.py --input "C:\\path\\to\\arousal_feature_matrix_scaled.csv"

    # also write a cleaned copy (rows with NaN feature values dropped)
    python 2.3_clean_feature_matrix_nans.py --input "C:\\path\\to\\arousal_feature_matrix_scaled.csv" --write-clean
"""

import argparse
import csv as csv_module
import re
from pathlib import Path

import pandas as pd

DEFAULT_ID_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec", "duration_sec", "sec_prev_event",
]


def detect_delimiter(path: Path, sample_lines: int = 5) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        sample = "".join(f.readline() for _ in range(sample_lines))
    try:
        dialect = csv_module.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except csv_module.Error:
        return ","


def detect_decimal(path: Path, delimiter: str, sample_lines: int = 20) -> str:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--id-columns", type=str, nargs="*", default=DEFAULT_ID_COLUMNS)
    parser.add_argument("--write-clean", action="store_true",
                         help="Write a cleaned copy (NaN rows dropped) next to the input, suffixed _clean.csv")
    args = parser.parse_args()

    delimiter = detect_delimiter(args.input)
    decimal = detect_decimal(args.input, delimiter)
    print(f"Detected delimiter={delimiter!r}, decimal={decimal!r}")

    df = pd.read_csv(args.input, sep=delimiter, decimal=decimal)
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}\n")

    present_id_cols = [c for c in args.id_columns if c in df.columns]
    missing_id_cols = [c for c in args.id_columns if c not in df.columns]
    feature_cols = [c for c in df.columns if c not in present_id_cols]

    if missing_id_cols:
        print(f"NOTE: expected id columns not found in this file: {missing_id_cols}")
        print("      (they're being treated as feature columns unless you pass --id-columns)\n")

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"NOTE: non-numeric columns among feature_cols (probably should be --id-columns): {non_numeric}\n")

    nan_counts = df[feature_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

    if nan_counts.empty:
        print("No NaNs found in any feature column. Nothing to clean.")
        return

    print("NaN counts per feature column (columns with 0 NaNs omitted):")
    print(nan_counts.to_string())

    nan_mask = df[feature_cols].isna().any(axis=1)
    n_affected = int(nan_mask.sum())
    print(f"\n{n_affected} of {len(df)} rows have at least one NaN feature value "
          f"({100 * n_affected / len(df):.2f}%).\n")

    preview_cols = [c for c in present_id_cols if c in df.columns][:6]
    print("Preview of affected rows (id columns only):")
    with pd.option_context("display.max_rows", 20, "display.width", 200):
        print(df.loc[nan_mask, preview_cols].head(20))

    if args.write_clean:
        clean_path = args.input.with_name(args.input.stem + "_clean.csv")
        cleaned = df.loc[~nan_mask].reset_index(drop=True)
        cleaned.to_csv(clean_path, sep=delimiter, decimal=decimal, index=False)
        print(f"\nWrote cleaned copy ({len(cleaned)} rows) to: {clean_path}")
        print("Original file was left untouched.")
    else:
        print("\nRun again with --write-clean to save a cleaned COPY (original file untouched).")


if __name__ == "__main__":
    main()