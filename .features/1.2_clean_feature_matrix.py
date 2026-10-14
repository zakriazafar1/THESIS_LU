"""
=============================================================================
1.2_clean_feature_matrix.py

Maakt de definitieve featurematrix voor de analyse, in drie stappen:

  1. Duur-filter: alleen events met MIN_DURATION_SEC <= duur <= MAX_DURATION_SEC
     (grenzen inclusief; 2.9-15.1 s = 3-15 s met een kleine marge).
  2. Slaapstadium-filter: alleen arousals uit slaap. Wake (0), niet-gescoord (-1)
     en events zonder stadium (NaN, bv. geen hypnogram) gaan eruit.
  3. NaN-check op de gefilterde set: telt per feature hoeveel waarden ontbreken
     (NaN of inf) en verwijdert elk event met minstens één ontbrekende feature.
     Eerst filteren, dan NaN-check: NaN's door foutief gescoorde, extreem korte
     events (bv. 0.004 s) vallen zo al in het duurfilter weg.

sec_prev_event wordt helemaal uit de matrix verwijderd (wordt niet gebruikt).
Niet-feature kolommen (subject_id, group, night_id, stage_rk, event_idx,
start_sec, end_sec) tellen niet mee in de NaN-check.

event_idx wordt NIET opnieuw genummerd: zo blijft elk event terug te
koppelen aan de ORIGIN-matrix (er ontstaan gaten waar rijen verwijderd zijn).

Input : INPUT_FILE (zie configuratie)
Output: arousal_feature_matrix_FILTERED.csv (in dezelfde map)

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

INPUT_FILE = EVENTS_DIR / "arousal_feature_matrix_ORIGIN_clipped.csv"
OUTPUT_FILE_FILTERED = EVENTS_DIR / "arousal_feature_matrix_FILTERED.csv"

# Duur-filter (grenzen inclusief)
MIN_DURATION_SEC = 2.9
MAX_DURATION_SEC = 15.1

# Slaapstadia die eruit gaan: wake (0) en niet-gescoord (-1); NaN gaat ook weg
EXCLUDE_STAGES = [0, -1]

NON_FEATURE_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec",
]
DROP_COLUMNS = ["sec_prev_event"]   # wordt niet gebruikt -> uit de matrix

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


def indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# =============================================================================
# HOOFDLOOP
# =============================================================================

def main():
    if not INPUT_FILE.exists():
        print(f"Featurematrix niet gevonden: {INPUT_FILE}")
        return

    df = load_feature_matrix(INPUT_FILE)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    n_start = len(df)
    print(f"Ingeladen: {INPUT_FILE.name} - {n_start} events "
          f"({df.groupby(['subject_id', 'night_id']).ngroups} nachten)\n")

    check_cols = [c for c in df.columns
                  if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    if not check_cols:
        print("Geen numerieke features gevonden. Kolommen zoals ingelezen:")
        print(list(df.columns))
        return

    # ---- Stap 1: duur-filter ----
    dur = df["duration_sec"]
    n_short = int((dur < MIN_DURATION_SEC).sum())
    n_long = int((dur > MAX_DURATION_SEC).sum())
    n_no_dur = int(dur.isna().sum())
    df = df[dur.between(MIN_DURATION_SEC, MAX_DURATION_SEC, inclusive="both")]

    # ---- Stap 2: slaapstadium-filter ----
    stage = df["stage_rk"]
    n_wake = int((stage == 0).sum())
    n_unscored = int((stage == -1).sum())
    n_no_stage = int(stage.isna().sum())
    df = df[~stage.isin(EXCLUDE_STAGES) & stage.notna()]
    n_after_filters = len(df)

    # ---- Stap 3: NaN-check (op de gefilterde set) ----
    nan_per_feature = df[check_cols].isna().sum()
    missing_mask = df[check_cols].isna().any(axis=1)
    n_nan_rows = int(missing_mask.sum())
    n_nan_cells = int(nan_per_feature.sum())

    print("NaN-check (na duur- en stadiumfilter)")
    print(f"  features gecontroleerd     : {len(check_cols)}")
    print(f"  ontbrekende waarden totaal : {n_nan_cells} "
          f"({100 * n_nan_cells / max(1, n_after_filters * len(check_cols)):.3f}% van alle cellen)")
    print(f"  events met >=1 NaN         : {n_nan_rows} "
          f"({100 * n_nan_rows / max(1, n_after_filters):.2f}%)")
    if n_nan_cells:
        print("  NaN per feature (alleen features met >0):")
        print(indent(nan_per_feature[nan_per_feature > 0].sort_values(ascending=False).to_string()))
        print("  events met NaN:")
        print(indent(df.loc[missing_mask, ["subject_id", "night_id", "start_sec", "duration_sec"]]
                     .to_string(index=False)))
    df = df[~missing_mask]
    n_final = len(df)

    # ---- Overzicht ----
    print("\nOVERZICHT")
    print(f"  start                                    : {n_start:6d}")
    print(f"  - korter dan {MIN_DURATION_SEC:g} s                       : {n_short:6d}")
    print(f"  - langer dan {MAX_DURATION_SEC:g} s                      : {n_long:6d}")
    if n_no_dur:
        print(f"  - geen duur (NaN)                        : {n_no_dur:6d}")
    print(f"  - wake (stage 0)                         : {n_wake:6d}")
    print(f"  - niet-gescoord (stage -1)               : {n_unscored:6d}")
    print(f"  - geen stadium (NaN)                     : {n_no_stage:6d}")
    print(f"  - events met NaN in een feature          : {n_nan_rows:6d}")
    print(f"  = over                                   : {n_final:6d} "
          f"({100 * n_final / n_start:.1f}%)")
    print(f"    uit {df.groupby(['subject_id', 'night_id']).ngroups} nachten, "
          f"{df['subject_id'].nunique()} proefpersonen")
    print("  verdeling slaapstadia:",
          ", ".join(f"{int(s)}: {n}" for s, n in df["stage_rk"].value_counts().sort_index().items()))

    df.to_csv(OUTPUT_FILE_FILTERED, index=False, float_format="%.3f")
    print(f"\nOpgeslagen: {OUTPUT_FILE_FILTERED}")
    print(f"Kolommen: {df.shape[1]} (sec_prev_event verwijderd)")


if __name__ == "__main__":
    main()