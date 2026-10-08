"""
=============================================================================
1.3_feature_summary.py

Maakt een samenvattende tabel van de featurematrix die door
1_feature_matrix.py is aangemaakt (arousal_feature_matrix_ORIGIN.csv).

Output: één rij per feature, met de kolommen
  feature, n, n_missing, min, max, mean, median, q1, q3, iqr

  - n          = aantal events met een geldige (niet-NaN) waarde
  - n_missing  = aantal events met NaN
  - q1 / q3    = 25e / 75e percentiel
  - iqr        = q3 - q1

Niet-feature kolommen (ID's, tijdstempels en de slaapstage) worden
overgeslagen: subject_id, group, night_id, stage_rk, event_idx,
start_sec, end_sec. duration_sec en sec_prev_event worden WEL meegenomen.

De tabel wordt opgeslagen in dezelfde map als de featurematrix:
  arousal_feature_SUMMARY.csv

Gebruik:
  python 1.3_feature_summary.py
  python 1.3_feature_summary.py --input "pad\\naar\\arousal_feature_matrix_FILTERED.csv"

Met --input wordt de naam van het input-bestand aan de output toegevoegd,
zodat de standaard-samenvatting niet overschreven wordt:
  ..._FILTERED.csv  ->  arousal_feature_SUMMARY_FILTERED.csv
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATIE
# =============================================================================

EVENTS_DIR = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices\.feature info")

INPUT_FILE = EVENTS_DIR / "arousal_feature_matrix_NO_NA.csv"
OUTPUT_FILE = EVENTS_DIR / "arousal_feature_SUMMARY.csv"

NON_FEATURE_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec",
]

# =============================================================================
# SAMENVATTING
# =============================================================================

def load_feature_matrix(path: Path) -> pd.DataFrame:
    """
    Leest de featurematrix robuust in:
      - scheidingsteken automatisch detecteren (komma of puntkomma, bv. na
        opslaan in Excel met NL-instellingen)
    """
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


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Alle numerieke kolommen behalve ID's/tijdstempels, in de originele volgorde."""
    return [c for c in df.columns
            if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def summarize_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Berekent per feature: n, n_missing, min, max, mean, median, q1, q3, iqr."""
    rows = []
    for col in feature_cols:
        x = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(x):
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            stats = {
                "min": x.min(), "max": x.max(),
                "mean": x.mean(), "median": x.median(),
                "q1": q1, "q3": q3, "iqr": q3 - q1,
            }
        else:
            stats = {k: np.nan for k in ("min", "max", "mean", "median", "q1", "q3", "iqr")}

        rows.append({"feature": col, "n": len(x), "n_missing": len(df) - len(x), **stats})
    return pd.DataFrame(rows)


def input_tag(path: Path) -> str:
    """'arousal_feature_matrix_FILTERED' -> 'FILTERED'; andere namen blijven heel."""
    prefix = "arousal_feature_matrix_"
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None,
                        help="Pad naar een andere featurematrix (standaard: INPUT_FILE)")
    args = parser.parse_args()

    input_file = args.input if args.input else INPUT_FILE
    output_file = OUTPUT_FILE
    if args.input:
        output_file = OUTPUT_FILE.with_name(f"{OUTPUT_FILE.stem}_{input_tag(input_file)}{OUTPUT_FILE.suffix}")

    if not input_file.exists():
        print(f"Featurematrix niet gevonden: {input_file}")
        print("Draai eerst 1_feature_matrix.py.")
        return

    df = load_feature_matrix(input_file)
    feature_cols = get_feature_columns(df)
    print(f"Featurematrix ingeladen: {input_file.name} - {df.shape[0]} events, {len(feature_cols)} features")

    if not feature_cols:
        print("Geen numerieke features gevonden. Kolommen zoals ingelezen:")
        print(list(df.columns))
        print(df.head(3).to_string())
        return

    SUMMARY = summarize_features(df, feature_cols)
    SUMMARY.to_csv(output_file, index=False, float_format="%.3f")
    print(f"Samenvatting opgeslagen: {output_file}")


if __name__ == "__main__":
    main()