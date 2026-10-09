"""
add_stress_ratings.py

Voegt stress_rating_e (avond) en stress_rating_m (ochtend) uit 'stress ratings.xlsx'
toe aan nights_overview.csv.

- Alleen rijen met study = NSR worden gebruikt.
- Matching op de eerste drie kolommen van de Excel (study, subject, nacht).
  Het script bepaalt zelf welke van die drie de study-, subject- en nachtkolom is;
  overschrijf in CONFIG als dat misgaat.
- nights_overview.csv wordt als tekst ingelezen, zodat de bestaande opmaak
  (decimale komma, 2/3 decimalen) exact behouden blijft.

Output: nights_overview_stress.csv (zelfde map), plus stress_unmatched_nights.csv indien nodig.

Gebruik:
  python add_stress_ratings.py --inspect   # toont kolommen, herkenning en voorbeeld-matching
  python add_stress_ratings.py             # volledige run
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
BASE = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents"
            r"\THESIS_OUTPUTS\PROJECT 2\.data descriptives")

NIGHTS_CSV  = BASE / "nights overview" / "nights_overview.csv"
STRESS_XLSX = BASE / "stress ratings.xlsx"
OUT_CSV     = BASE / "distress data" / "nights_overview_distress.csv"

SHEET = 0                 # eerste tabblad; of de naam van het tabblad
STUDY_VALUE = "NSR"
STRESS_COLS = ["stress_rating_e", "stress_rating_m"]

# None = automatisch herkennen binnen de eerste drie kolommen
XL_STUDY_COL   = "study"
XL_SUBJECT_COL = "subject"
XL_NIGHT_COL   = "ses"

DEFAULT_TIMEPOINT = "T0"  # als de nachtkolom alleen een nachtnummer bevat
N_DECIMALS = 2
OUT_SEP, OUT_DECIMAL = ";", ","


# ============================================================
# HELPERS
# ============================================================
def norm_subject(x) -> str:
    """-> 'bnbd_nsr_17598'. Werkt voor volledige IDs en voor alleen het nummer."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    m = re.search(r"bnbd_[a-z]+_\d+", s)
    if m:
        pre, num = m.group(0).rsplit("_", 1)
        return f"{pre}_{int(num):05d}"
    d = re.findall(r"\d+", s)
    return f"bnbd_{STUDY_VALUE.lower()}_{int(d[0]):05d}" if d else s


def norm_night(x) -> str:
    """-> 'T0_N1'. Werkt voor 'T0_N1', 'N1', '1', 1.0 of een volledige ID."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    m = re.search(r"(T\d+)_?N(\d+)", s)
    if m:
        return f"{m.group(1)}_N{int(m.group(2))}"
    m = re.search(r"N(?:IGHT|ACHT)?\s*_?(\d+)", s) or re.fullmatch(r"(\d+)(?:\.0+)?", s)
    return f"{DEFAULT_TIMEPOINT}_N{int(m.group(1))}" if m else s


def detect_key_cols(xl: pd.DataFrame):
    """Bepaal study/subject/nacht binnen de eerste drie kolommen."""
    first3 = list(xl.columns[:3])
    study = XL_STUDY_COL
    if study is None:  # kolom waarin STUDY_VALUE voorkomt
        study = next((c for c in first3
                      if xl[c].astype(str).str.strip().str.upper().eq(STUDY_VALUE).any()), None)
    rest = [c for c in first3 if c != study]
    subj = XL_SUBJECT_COL
    if subj is None:  # kolom met bnbd-IDs of 4-5-cijferige nummers
        def subj_score(c):
            v = xl[c].dropna().astype(str)
            return v.str.contains(r"bnbd_|\d{4,5}", case=False).mean() if len(v) else 0
        subj = max(rest, key=subj_score)
    night = XL_NIGHT_COL or next(c for c in rest if c != subj)
    return study, subj, night


def fmt_num(v) -> str:
    if pd.isna(v):
        return ""
    v = float(v)
    if v.is_integer():
        return str(int(v))
    return f"{v:.{N_DECIMALS}f}".replace(".", OUT_DECIMAL)


# ============================================================
# MAIN
# ============================================================
def main(inspect_only=False):
    # nights_overview als tekst -> opmaak blijft exact behouden
    nights = pd.read_csv(NIGHTS_CSV, sep=OUT_SEP, dtype=str, keep_default_na=False,
                         encoding="utf-8-sig")
    xl = pd.read_excel(STRESS_XLSX, sheet_name=SHEET)
    xl.columns = [str(c).strip() for c in xl.columns]

    missing = [c for c in STRESS_COLS if c not in xl.columns]
    if missing:
        raise KeyError(f"Kolommen {missing} niet in Excel. Beschikbaar: {list(xl.columns)}")

    study_col, subj_col, night_col = detect_key_cols(xl)

    print("=" * 70)
    print(f"Excel: {len(xl)} rijen | eerste drie kolommen: {list(xl.columns[:3])}")
    print(f"  -> study='{study_col}', subject='{subj_col}', night='{night_col}'")
    if study_col is None:
        print(f"  [LET OP] geen kolom met '{STUDY_VALUE}' gevonden in de eerste drie kolommen; "
              f"zet XL_STUDY_COL in CONFIG")
    else:
        print(f"  study-waarden: {xl[study_col].astype(str).str.strip().value_counts().to_dict()}")

    if study_col is not None:
        xl = xl[xl[study_col].astype(str).str.strip().str.upper() == STUDY_VALUE].copy()
    xl["_subject"] = xl[subj_col].map(norm_subject)
    xl["_night"] = xl[night_col].map(norm_night)
    for c in STRESS_COLS:
        xl[c] = pd.to_numeric(xl[c].astype(str).str.replace(",", "."), errors="coerce")

    print(f"  NSR-rijen: {len(xl)}")
    print("  voorbeeld ruw -> genormaliseerd:")
    for _, r in xl.head(3).iterrows():
        print(f"    {r[subj_col]!r}, {r[night_col]!r} -> {r['_subject']}, {r['_night']}")
    print(f"  voorbeeld nights_overview: {nights[['subject', 'night']].head(3).values.tolist()}")

    # dubbele (subject, nacht)-combinaties in de Excel
    dup = xl.duplicated(["_subject", "_night"], keep=False)
    if dup.any():
        print(f"  [LET OP] {dup.sum()} Excel-rijen met dubbele subject+nacht; eerste rij wordt gebruikt:")
        print(xl.loc[dup, [subj_col, night_col] + STRESS_COLS].head(10).to_string(index=False))
    xl_one = xl.drop_duplicates(["_subject", "_night"], keep="first")

    key = pd.MultiIndex.from_frame(nights[["subject", "night"]])
    xl_idx = pd.MultiIndex.from_frame(xl_one[["_subject", "_night"]])
    n_match = key.isin(xl_idx).sum()
    print(f"  nachten gematcht: {n_match} / {len(nights)}")
    print("=" * 70)
    if inspect_only:
        return

    merged = nights.merge(
        xl_one[["_subject", "_night"] + STRESS_COLS],
        left_on=["subject", "night"], right_on=["_subject", "_night"], how="left"
    ).drop(columns=["_subject", "_night"])

    unmatched = merged[merged[STRESS_COLS].isna().all(axis=1)][["subject", "night"]]
    if len(unmatched):
        print(f"[LET OP] {len(unmatched)} nachten zonder stress ratings:")
        print(unmatched.to_string(index=False))
        unmatched.to_csv(OUT_CSV.parent / "stress_unmatched_nights.csv", index=False, sep=OUT_SEP)
    for c in STRESS_COLS:
        n_na = merged[c].isna().sum()
        if n_na:
            print(f"[INFO] {c}: {n_na} nachten zonder waarde")

    for c in STRESS_COLS:
        merged[c] = merged[c].map(fmt_num)
    merged.to_csv(OUT_CSV, index=False, sep=OUT_SEP, encoding="utf-8-sig")
    print(f"\nOpgeslagen: {OUT_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    main(inspect_only=ap.parse_args().inspect)