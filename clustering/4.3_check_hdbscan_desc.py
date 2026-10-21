"""
=============================================================================
4.3_check_hdbscan_desc.py

Quick, independent check of 5_cluster_descriptives.py: are the cluster labels
linked to the right events, and are the numbers in the table right?

Per run (tag) it checks:
  1. Link: labels and original matrix are merged again (independently, with a
     plain pandas merge) on subject_id + night_id + event_idx. Every labelled
     event must be found exactly once.
  2. Same event: columns present in both files (start_sec, end_sec, stage_rk,
     ...) must be identical after linking. If the link were shifted or
     scrambled, these would differ.
  3. Same values: for every feature present in both files (reduced_features
     runs: the scaled features), the Spearman correlation between the value
     in the labels file and the original value must be 1.000 -- scaling and
     log-transforms are monotonic, so the rank order must be identical.
  4. Counts: events per cluster in the labels file = in
     hdbscan_final_cluster_summary.csv = in the descriptives table.
  5. Numbers: median, Q1, Q3 per group x feature are recomputed (pandas
     quantile) and compared with cluster_descriptives_long.csv.
  6. Spot check: 5 random events printed side by side (labels file vs original)
     to look at by eye.

Usage:
  python 4.3_check_hdbscan_desc.py
  python 4.3_check_hdbscan_desc.py --tags whitened_single
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2"
)
HDBSCAN_DIR = BASE / r"4. clustering\4. hdbscan"
ORIGINAL_INPUT = BASE / r"reduced feature matrix\arousal_features_reduced_original.csv"
DESCRIPTIVES_DIR = BASE / r"6. cluster descriptives\hdbscan"
DEFAULT_TAGS = ["whitened", "whitened_single", "reduced_features", "reduced_features_single"]

KEY = ["subject_id", "night_id", "event_idx"]
META_CHECK = ["start_sec", "end_sec", "sec_prev_event", "stage_rk", "group"]
TOL = 1e-6


def read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    num_as_text = [c for c in df.columns if c not in ("subject_id", "group", "night_id")
                   and not pd.api.types.is_numeric_dtype(df[c])]
    if num_as_text:
        df = pd.read_csv(path, sep=None, engine="python", decimal=",")
    for k in ("subject_id", "night_id"):
        df[k] = df[k].astype(str).str.strip()
    df["event_idx"] = df["event_idx"].astype(int)
    return df


def ok(cond: bool, msg: str) -> bool:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    return cond


def check_tag(tag: str, original: pd.DataFrame, hd_dir: Path, desc_dir: Path) -> bool:
    print(f"\n=== {tag} ===")
    lab_path = hd_dir / tag / "hdbscan_final_labels.csv"
    if not lab_path.exists():
        print(f"  [SKIPPED] no labels file: {lab_path}")
        return None
    lab = read(lab_path)
    all_ok = True

    # 1. link
    m = lab.merge(original, on=KEY, how="left", suffixes=("_lab", "_orig"),
                  indicator=True, validate="one_to_one")
    all_ok &= ok((m["_merge"] == "both").all(),
                 f"all {len(lab)} labelled events found in the original matrix "
                 f"(not found: {(m['_merge'] != 'both').sum()})")
    m = m[m["_merge"] == "both"]

    # 2. same event: shared metadata identical
    for c in META_CHECK:
        a, b = f"{c}_lab", f"{c}_orig"
        if a in m and b in m:
            if pd.api.types.is_numeric_dtype(m[a]):
                same = np.isclose(m[a], m[b], atol=TOL, equal_nan=True)
            else:
                same = (m[a].astype(str) == m[b].astype(str)).to_numpy()
            all_ok &= ok(same.all(), f"{c} identical in both files ({(~same).sum()} mismatches)")

    # 3. same values: rank order of shared features identical
    shared = [c[:-4] for c in m.columns if c.endswith("_lab") and f"{c[:-4]}_orig" in m
              and c[:-4] not in META_CHECK and pd.api.types.is_numeric_dtype(m[c])]
    for f in shared:
        rho = m[f"{f}_lab"].corr(m[f"{f}_orig"], method="spearman")
        all_ok &= ok(rho > 0.9999, f"{f}: Spearman rho(labels file, original) = {rho:.4f}")
    if not shared:
        print("  [INFO] no shared features (PCA run) -- link checked via metadata above")
    dur = "duration_sec_orig" if "duration_sec_orig" in m else "duration_sec"
    if dur in m and {"start_sec_orig", "end_sec_orig"} <= set(m.columns):
        d = (m["end_sec_orig"] - m["start_sec_orig"] - m[dur]).abs()
        print(f"  [INFO] original duration_sec vs end-start: max difference {d.max():.3f} s")

    # 4. counts
    counts = m["cluster"].value_counts().sort_index()
    summ_path = hd_dir / tag / "hdbscan_final_cluster_summary.csv"
    if summ_path.exists():
        s = pd.read_csv(summ_path, sep=None, engine="python").set_index("cluster")["n_events"]
        all_ok &= ok(s.sort_index().to_dict() == counts.to_dict(),
                     f"cluster sizes match hdbscan_final_cluster_summary.csv {counts.to_dict()}")

    # 5. numbers in the descriptives
    long_path = desc_dir / tag / "cluster_descriptives_long.csv"
    if not long_path.exists():
        print(f"  [SKIPPED] no descriptives yet: {long_path}")
        return all_ok
    desc = pd.read_csv(long_path, sep=None, engine="python")
    if desc.shape[1] > 1 and not pd.api.types.is_numeric_dtype(desc.get("median", pd.Series(dtype=float))):
        desc = pd.read_csv(long_path, sep=None, engine="python", decimal=",")
    desc.columns = [c.strip().lstrip("\ufeff") for c in desc.columns]
    needed = {"group", "feature", "n", "median", "q1", "q3"}
    if not needed <= set(desc.columns):
        all_ok &= ok(False, f"{long_path.name} has unexpected columns {list(desc.columns)} "
                            f"(expected {sorted(needed)}) -- rerun 5_cluster_descriptives.py "
                            "and do not save the CSV from Excel")
        return all_ok
    groups = {f"Cluster {c}": m[m["cluster"] == c] for c in counts.index if c != -1}
    if -1 in counts.index:
        groups["Noise"] = m[m["cluster"] == -1]
    groups["All events"] = m
    max_diff, n_checked = 0.0, 0
    for _, r in desc.iterrows():
        if r["group"] not in groups:
            all_ok &= ok(False, f"group '{r['group']}' in the descriptives but not in the labels")
            continue
        col = r["feature"] + ("_orig" if f"{r['feature']}_orig" in m else "")
        v = groups[r["group"]][col].replace([np.inf, -np.inf], np.nan).dropna()
        q = v.quantile([0.25, 0.5, 0.75]).to_numpy()
        max_diff = max(max_diff, np.nanmax(np.abs(q - r[["q1", "median", "q3"]].to_numpy(float))))
        all_ok &= len(v) == r["n"]
        n_checked += 1
    all_ok &= ok(max_diff < TOL, f"{n_checked} medians/IQRs recomputed, max difference {max_diff:.2e}")

    # 6. spot check
    feats = [c for c in original.columns if c not in KEY + META_CHECK][:4]
    show = KEY + ["cluster"] + [f"start_sec_{s}" for s in ("lab", "orig") if f"start_sec_{s}" in m] \
        + [f + ("_orig" if f"{f}_orig" in m else "") for f in feats]
    print("  Spot check (5 random events):")
    print(m[show].sample(min(5, len(m)), random_state=1).to_string(index=False))
    return all_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    p.add_argument("--hdbscan-dir", type=Path, default=HDBSCAN_DIR)
    p.add_argument("--original", type=Path, default=ORIGINAL_INPUT)
    p.add_argument("--descriptives-dir", type=Path, default=DESCRIPTIVES_DIR)
    a = p.parse_args()

    original = read(a.original)
    results = {t: check_tag(t, original, a.hdbscan_dir, a.descriptives_dir) for t in a.tags}
    print("\n" + "=" * 40)
    for t, r in results.items():
        status = "SKIPPED (no labels file)" if r is None else (
            "ALL CHECKS PASSED" if r else "CHECK FAILED -- see above")
        print(f"{t:<28} {status}")


if __name__ == "__main__":
    main()