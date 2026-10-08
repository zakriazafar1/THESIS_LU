"""
4_center_features.py 

Centreer de geschaalde features per deelnemer (of per nacht) en standaardiseer 
opnieuw. Robuustheidscheck: is de clusterstructuur binnen personen hetzelfde 
als over personen heen?

Wat het script doet:
1. Leest de geschaalde feature-matrix (z-scores).
2. Trekt per groep (deelnemer, of deelnemer x nacht) het groepsgemiddelde
   (of de mediaan) af, per feature -> elk event = "hoe wijkt deze arousal 
   af van wat gebruikelijk is voor deze persoon/nacht"
3. Standaardiseert opnieuw (z-score over alle events), zodat elke feature weer SD 1 heeft.
   Nodig voor HDBSCAN (afstanden). 
4. Schrijft een nieuw CSV met precies dezelfde kolommen als de invoer, zodat
   2_GMM.py en de HDBSCAN-pipeline het via --input kunnen gebruiken.
5. Schrijft een kort rapport: per feature hoeveel variantie tussen groepen zat
   (en dus is verwijderd). 

Gebruik
-------
    python 4_center_features.py                         # per deelnemer, gemiddelde
    python 4_center_features.py --level night           # per nacht (secundaire check)
    python 4_center_features.py --stat median           # mediaan i.p.v. gemiddelde
    python 4_center_features.py --input X.csv --outdir Y

Daarna bijvoorbeeld:
    python 2_GMM.py --input "...\\arousal_features_reduced_scaled_centered_participant.csv" ^
                    --outdir "...\\4. clustering\\GMM_centered_participant"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Standaardpaden
# ---------------------------------------------------------------------------
BASE = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2")
DEFAULT_INPUT = BASE / "reduced feature matrix" / "arousal_features_reduced_scaled.csv"
DEFAULT_OUTDIR = BASE / "centered feature matrix"

FEATURES = [
    "mean_delta_ratio",
    "mean_theta_ratio",
    "mean_alpha_ratio",
    "mean_sigma_ratio",
    "mean_beta_ratio",
    "duration_sec",
    "oxy_amp_ratio",
    "motion_rms",
]
LEVELS = {
    "participant": ["subject_id"],
    "night": ["subject_id", "night_id"],
}
MIN_EVENTS_WARN = 20   # waarschuwing als een groep minder events heeft (instabiel gemiddelde)


def read_csv_nl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if df[FEATURES].dtypes.eq(object).any():
        df = pd.read_csv(path, sep=";", decimal=",")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Ontbrekende feature-kolommen: {missing}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--level", choices=list(LEVELS), default="participant",
                    help="Centreren per deelnemer (hoofdanalyse) of per nacht (secundair)")
    ap.add_argument("--stat", choices=["mean", "median"], default="mean",
                    help="Groepsgemiddelde of -mediaan aftrekken")
    ap.add_argument("--no-rescale", action="store_true", help="Niet opnieuw z-scoren na centreren")
    args = ap.parse_args()

    print(f"Inlezen: {args.input}")
    df = read_csv_nl(args.input)
    keys = LEVELS[args.level]
    miss = [k for k in keys if k not in df.columns]
    if miss:
        raise ValueError(f"Kolommen nodig voor centreren ontbreken: {miss}")

    n_before = len(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Let op: {n_before - len(df)} events met NaN verwijderd.")

    sizes = df.groupby(keys).size()
    print(f"  {len(df)} events | {len(sizes)} groepen ({args.level}) | "
          f"events per groep: min {sizes.min()}, mediaan {int(sizes.median())}, max {sizes.max()}")
    small = (sizes < MIN_EVENTS_WARN).sum()
    if small:
        print(f"  LET OP: {small} groepen met < {MIN_EVENTS_WARN} events; hun gemiddelde is onzeker.")

    X = df[FEATURES].astype(float)
    group_center = X.groupby([df[k] for k in keys]).transform(args.stat)
    Xc = X - group_center

    # Rapport: welk deel van de variantie zat tussen groepen?
    var_total = X.var(ddof=1)
    var_within = Xc.var(ddof=1)
    report = pd.DataFrame({
        "var_total": var_total,
        "var_within_after_centering": var_within,
        "pct_variance_removed": 100 * (1 - var_within / var_total),
    })

    if not args.no_rescale:
        Xc = (Xc - Xc.mean()) / Xc.std(ddof=0)

    out = df.copy()
    out[FEATURES] = Xc.to_numpy()

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"centered_{args.level}" + ("" if args.stat == "mean" else "_median")
    out_csv = args.outdir / f"{args.input.stem}_{tag}.csv"
    out.to_csv(out_csv, sep=";", index=False)
    report.round(4).to_csv(args.outdir / f"centering_report_{tag}.csv", sep=";")
    (args.outdir / f"centering_settings_{tag}.json").write_text(json.dumps({
        "input": str(args.input),
        "output": str(out_csv),
        "level": args.level,
        "grouping_columns": keys,
        "center_statistic": args.stat,
        "rescaled_after_centering": not args.no_rescale,
        "n_events": int(len(out)),
        "n_groups": int(len(sizes)),
        "events_per_group": {"min": int(sizes.min()), "median": float(sizes.median()), "max": int(sizes.max())},
    }, indent=2))

    print("\nVariantie die tussen groepen zat (en is verwijderd):")
    print(report["pct_variance_removed"].round(1).to_string())
    print("\nControle na centreren (moet ~0 en ~1 zijn):")
    print(out[FEATURES].agg(["mean", "std"]).T.round(3).to_string())
    print(f"\nOpgeslagen:\n  {out_csv}")


if __name__ == "__main__":
    main()