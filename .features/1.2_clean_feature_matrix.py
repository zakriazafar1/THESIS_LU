"""
=============================================================================
1.2_clean_feature_matrix.py

Verwijdert alle events (rijen) uit de featurematrix waarin minstens één
feature ontbreekt (NaN of inf). 

Uitzondering: sec_prev_event telt NIET mee, want die is per definitie leeg
bij het eerste event van elke nacht.
Niet-feature kolommen (subject_id, group, night_id, stage_rk, event_idx,
start_sec, end_sec) tellen ook niet mee.

event_idx wordt NIET opnieuw genummerd: zo blijft elk event terug te
koppelen aan arousal_feature_matrix_ORIGIN.csv (er ontstaan dus gaten in
de nummering waar rijen verwijderd zijn).

Daarnaast wordt een tweede versie weggeschreven met alleen events waarvan
de duur tussen MIN_DURATION_SEC en MAX_DURATION_SEC ligt (standaard 3-15 s,
grenzen inclusief), gebaseerd op de CLEAN versie.

Input : arousal_feature_matrix_ORIGIN.csv
Output: arousal_feature_matrix_CLEAN.csv      (alle duren, zonder NaN)
        arousal_feature_matrix_FILTERED.csv   (zonder NaN, 3 <= duur <= 15 s)
        (beide in dezelfde map)

Gebruik:
  python 1.2_clean_feature_matrix.py
=============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATIE
# =============================================================================

EVENTS_DIR = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices\.feature info")

INPUT_FILE = EVENTS_DIR / "arousal_feature_matrix_ORIGIN.csv"
OUTPUT_FILE = EVENTS_DIR / "arousal_feature_matrix_NO_NA.csv"
OUTPUT_FILE_FILTERED = EVENTS_DIR / "arousal_feature_matrix_FILTERED.csv"

# Duur-filter voor de _FILTERED versie (grenzen inclusief)
MIN_DURATION_SEC = 2.9
MAX_DURATION_SEC = 15.1

NON_FEATURE_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec",
]
IGNORE_MISSING_IN = ["sec_prev_event"]   # mag leeg zijn, telt niet mee

# =============================================================================
# INLEZEN 
# =============================================================================

def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Detecteert komma/puntkomma-scheiding en zet decimale komma's om."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if col in ("subject_id", "group", "night_id") or pd.api.types.is_numeric_dtype(df[col]):
            continue
        converted = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if converted.notna().any():
            df[col] = converted
    return df


# =============================================================================
# HOOFDLOOP
# =============================================================================

def main():
    if not INPUT_FILE.exists():
        print(f"Featurematrix niet gevonden: {INPUT_FILE}")
        return

    df = load_feature_matrix(INPUT_FILE)
    df = df.replace([np.inf, -np.inf], np.nan)

    check_cols = [c for c in df.columns
                  if c not in NON_FEATURE_COLUMNS
                  and c not in IGNORE_MISSING_IN
                  and pd.api.types.is_numeric_dtype(df[c])]

    if not check_cols:
        print("Geen numerieke features gevonden. Kolommen zoals ingelezen:")
        print(list(df.columns))
        return

    missing_mask = df[check_cols].isna().any(axis=1)
    n_total, n_removed = len(df), int(missing_mask.sum())

    # Overzicht: hoeveel NaN's per feature (alleen features met >0)
    missing_per_feature = df[check_cols].isna().sum()
    missing_per_feature = missing_per_feature[missing_per_feature > 0].sort_values(ascending=False)
    if len(missing_per_feature):
        print("Aantal missende waarden per feature:")
        print(missing_per_feature.to_string())
        print()

    # Overzicht: uit welke nachten worden rijen verwijderd
    if n_removed and {"subject_id", "night_id"}.issubset(df.columns):
        per_night = df[missing_mask].groupby(["subject_id", "night_id"]).size()
        print(f"Verwijderde events per nacht ({len(per_night)} nachten):")
        print(per_night.to_string())
        print()

    clean = df[~missing_mask]
    clean.to_csv(OUTPUT_FILE, index=False, float_format="%.3f")

    print(f"Events voor opschonen : {n_total}")
    print(f"Verwijderd            : {n_removed} ({100 * n_removed / n_total:.2f}%)")
    print(f"Over                  : {len(clean)}")
    print(f"Opgeslagen: {OUTPUT_FILE}")

    # ---- Tweede versie: alleen events met MIN <= duration <= MAX ----
    dur = clean["duration_sec"]
    in_range = dur.between(MIN_DURATION_SEC, MAX_DURATION_SEC, inclusive="both")
    filtered = clean[in_range]
    n_short = int((dur < MIN_DURATION_SEC).sum())
    n_long = int((dur > MAX_DURATION_SEC).sum())
    
    # Alleen arousals uit slaap: wake (0) en niet-gescoord (-1) eruit
    stage_ok = ~filtered["stage_rk"].isin([0, -1])
    print(f"  wake / niet-gescoord verwijderd: {int((~stage_ok).sum())}")
    filtered = filtered[stage_ok]

    filtered.to_csv(OUTPUT_FILE_FILTERED, index=False, float_format="%.3f")

    print(f"\nDuur-filter {MIN_DURATION_SEC:g}-{MAX_DURATION_SEC:g} s (op de CLEAN matrix):")
    print(f"  korter dan {MIN_DURATION_SEC:g} s : {n_short}")
    print(f"  langer dan {MAX_DURATION_SEC:g} s : {n_long}")
    print(f"  over                : {len(filtered)} ({100 * len(filtered) / len(clean):.1f}% van CLEAN)")
    print(f"Opgeslagen: {OUTPUT_FILE_FILTERED}")


if __name__ == "__main__":
    main()