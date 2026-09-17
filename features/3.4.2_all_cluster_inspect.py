#!/usr/bin/env python3
"""
3.4.2_all_cluster_inspect.py

Batch runner for 3.4.1_cluster_inspect.py: reads a CSV of UMAP+HDBSCAN
parameter combinations (e.g. bayesopt_refined_ranked2.csv, the ranked
output of 3.3.2_umap_hdbscan_bayes.py) and runs 3.4.1_cluster_inspect.py once
per combination - each into its own subfolder - so you get the full
inspection output (embedding CSV, scatter, boxplots, summary) for every
combination in one go, instead of running it by hand per combo.

It also concatenates all the per-combo cluster_summary.csv files into a
single all_combos_summary.csv (tagged with each combo's parameters and
rank), so you can compare cluster counts/sizes/composition across
combinations without opening every subfolder individually.

3.4.1_cluster_inspect.py itself is NOT modified - this just calls it
repeatedly as a subprocess with different --output-dir / parameter values.

Usage:
    python 3.4.2_all_cluster_inspect.py --combos bayesopt_refined_ranked2.csv

    # only run the first N rows (as ordered in the CSV, i.e. top-ranked)
    python 3.4.2_all_cluster_inspect.py --combos bayesopt_refined_ranked2.csv --top 10

    # point at a non-default script/input/output location
    python 3.4.2_all_cluster_inspect.py --combos bayesopt_refined_ranked2.csv ^
        --script "C:\\path\\to\\3.4.1_cluster_inspect.py" ^
        --input  "C:\\path\\to\\arousal_feature_matrix_scaled.csv" ^
        --output-root "C:\\path\\to\\cluster_inspect_batch"

    # just print the commands without running them
    python 3.4.2_all_cluster_inspect.py --combos bayesopt_refined_ranked2.csv --dry-run

Output (under --output-root):
    rank01_nn17_md0.0415_nc2_mcs11_ms5/   <- one subfolder per combo, containing
        embedding_with_clusters.csv           the normal 3.4.1_cluster_inspect.py output
        cluster_scatter.png                   (scatter skipped when n_components != 2,
        cluster_boxplots.png                   exactly like the underlying script does)
        cluster_summary.csv
    rank02_.../
    ...
    all_combos_summary.csv                <- every combo's cluster_summary.csv stacked,
                                              with combo_rank / combo_dir / params prefixed
"""

import argparse
import csv as csv_module
import subprocess
import sys
from pathlib import Path

import pandas as pd

DEFAULT_SCRIPT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_LU\features\3.4.1_cluster_inspect.py"
)
DEFAULT_INPUT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
    r"\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled\arousal_feature_matrix_scaled_clean.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
    r"\THESIS_OUTPUTS\PROJECT 2\4. clustering"
)

PARAM_COLUMNS = ["n_neighbors", "min_dist", "n_components", "min_cluster_size", "min_samples"]


def detect_delimiter(path: Path, sample_lines: int = 5) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        sample = "".join(f.readline() for _ in range(sample_lines))
    try:
        dialect = csv_module.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except csv_module.Error:
        return ","

def combo_dir_name(row_dict, rank):
    return (
        f"rank{rank:02d}_nn{int(row_dict['n_neighbors'])}_"
        f"md{float(row_dict['min_dist']):.4f}_"
        f"nc{int(row_dict['n_components'])}_"
        f"mcs{int(row_dict['min_cluster_size'])}_"
        f"ms{int(row_dict['min_samples'])}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--combos", type=Path, required=True, help="CSV with parameter combinations")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT, help="Path to 3.4.1_cluster_inspect.py")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Feature matrix CSV, passed through as --input")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top", type=int, default=None, help="Only run the first N rows of the CSV")
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args()

    if not args.script.exists():
        sys.exit(f"Can't find 3.4.1_cluster_inspect.py at: {args.script}\n"
                  f"Pass its real location with --script.")

    delimiter = detect_delimiter(args.combos)
    combos = pd.read_csv(args.combos, sep=delimiter)
    combos.columns = [c.strip() for c in combos.columns]
    missing = [c for c in PARAM_COLUMNS if c not in combos.columns]
    if missing:    
        sys.exit(f"CSV is missing expected columns: {missing}")

    if args.top:
        combos = combos.head(args.top)

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_frames = []
    failures = []
    n = len(combos)

    for rank, row in enumerate(combos.to_dict(orient="records"), start=1):
        combo_out = args.output_root / combo_dir_name(row, rank)

        cmd = [
            sys.executable, str(args.script),
            "--input", str(args.input),
            "--output-dir", str(combo_out),
            "--metric", args.metric,
            "--random-state", str(args.random_state),
            "--n-neighbors", str(int(row["n_neighbors"])),
            "--min-dist", str(row["min_dist"]),
            "--n-components", str(int(row["n_components"])),
            "--min-cluster-size", str(int(row["min_cluster_size"])),
            "--min-samples", str(int(row["min_samples"])),
        ]

        print(f"\n=== [{rank}/{n}] {combo_out.name} ===")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout[-1500:])
        if result.returncode != 0:
            print(f"FAILED (exit {result.returncode}):", file=sys.stderr)
            print(result.stderr[-1500:], file=sys.stderr)
            failures.append(combo_out.name)
            continue

        summary_path = combo_out / "cluster_summary.csv"
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            for col in reversed(PARAM_COLUMNS):
                summary_df.insert(0, col, row[col])
            summary_df.insert(0, "combo_dir", combo_out.name)
            summary_df.insert(0, "combo_rank", rank)
            summary_frames.append(summary_df)

    if args.dry_run:
        print("\nDry run - nothing was executed.")
        return

    if summary_frames:
        all_summary = pd.concat(summary_frames, ignore_index=True)
        all_summary_path = args.output_root / "all_combos_summary.csv"
        all_summary.to_csv(all_summary_path, index=False)
        print(f"\nSaved combined summary across {len(summary_frames)} combo(s): {all_summary_path}")

    if failures:
        print(f"\n{len(failures)} combo(s) failed: {failures}", file=sys.stderr)
    print(f"\nDone. {n - len(failures)}/{n} combo(s) completed successfully.")


if __name__ == "__main__":
    main()
    