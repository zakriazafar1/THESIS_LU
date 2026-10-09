"""
subjects_overview.py

Overzicht van subjects/nachten/events in de reduced feature matrix,
gekoppeld aan gender/age uit de subjects-info file.

Outputs (in OUT_DIR):
  - per_subject_overview.csv   : 1 rij per subject (n_nachten, n_events, events/nacht, gender, age)
  - per_night_overview.csv     : 1 rij per nacht (subject, nacht, n_events)
  - summary_table.csv          : samenvattende tabel (descriptives voor Methods/Results)
  - unmatched_subjects.csv     : subjects uit de feature matrix zonder match in subjects-info (indien aanwezig)

Gebruik:
  python subjects_overview.py --inspect     # toont alleen kolommen + gedetecteerde ID-kolommen
  python subjects_overview.py               # volledige run
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
BASE = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2")

FEATURES_CSV = BASE / "reduced feature matrix" / "arousal_features_reduced_original.csv"
SUBJECTS_CSV = BASE / "data descriptives" / "subjects_timepoints_total(in).csv"
OUT_DIR      = BASE / "data descriptives" / "subjects overview" 

# Laat op None voor auto-detectie; vul in als de detectie de verkeerde kolom pakt.
FM_SUBJECT_COL = "subject_id"   # bv. "subject"
FM_NIGHT_COL   = "night_id"   # bv. "night" of "edf_file"
FM_EVENT_COL   = "event_idx"   # optioneel (bv. "event_id"); anders telt elke rij als 1 event
SI_SUBJECT_COL = "nsr_id"   # bv. "participant_id"
SI_GENDER_COL  = "female"   # bv. "gender" / "sex"
SI_AGE_COL     = "age"   # bv. "age"

# Subject-IDs matchen op alleen het numerieke deel (bv. "BNBD_012" == "12").
# Zet op False als IDs letterlijk gelijk moeten zijn.
MATCH_ON_DIGITS = True

# Output in Nederlands Excel-formaat (; en decimale komma)
OUT_SEP, OUT_DECIMAL = ";", ","

# ============================================================
# HELPERS
# ============================================================
SUBJECT_CANDIDATES = ["subject", "subject_id", "subj", "participant", "participant_id",
                      "pp", "ppn", "id", "record_id", "castor_id"]
NIGHT_CANDIDATES   = ["night", "night_id", "nacht", "session", "recording", "edf",
                      "edf_file", "file", "filename", "night_idx", "date"]
EVENT_CANDIDATES   = ["event_id", "event", "arousal_id", "arousal_idx", "event_idx"]
GENDER_CANDIDATES  = ["gender", "sex", "geslacht"]
AGE_CANDIDATES     = ["age", "leeftijd", "age_years"]


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Leest CSV met automatische detectie van ; vs , en decimale komma."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    decimal = "," if sep == ";" else "."
    df = pd.read_csv(path, sep=sep, decimal=decimal, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates, override=None, required=True, label=""):
    if override is not None:
        if override not in df.columns:
            raise KeyError(f"[{label}] kolom '{override}' niet gevonden. Beschikbaar: {list(df.columns)}")
        return override
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:                      # exacte match
        if cand in lower:
            return lower[cand]
    for cand in candidates:                      # bevat-match
        for lc, orig in lower.items():
            if cand in lc:
                return orig
    if required:
        raise KeyError(f"[{label}] geen kolom gevonden voor {candidates}. "
                       f"Zet deze handmatig in CONFIG. Beschikbaar: {list(df.columns)}")
    return None


def norm_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if MATCH_ON_DIGITS:
        digits = re.findall(r"\d+", s)
        return str(int(digits[0])) if digits else s.upper()
    return s.upper()


def describe(series: pd.Series, name: str, decimals=1) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"Variabele": name, "N": 0, "Waarde": "—"}
    return {
        "Variabele": name,
        "N": len(s),
        "Waarde": (f"M = {s.mean():.{decimals}f}, SD = {s.std(ddof=1):.{decimals}f}, "
                   f"Mdn = {s.median():.{decimals}f}, range {s.min():.{decimals}f}–{s.max():.{decimals}f}"),
    }


# ============================================================
# MAIN
# ============================================================
def main(inspect_only=False):
    fm = read_csv_auto(FEATURES_CSV)
    si = read_csv_auto(SUBJECTS_CSV)

    fm_subj  = find_col(fm, SUBJECT_CANDIDATES, FM_SUBJECT_COL, label="feature matrix subject")
    fm_night = find_col(fm, NIGHT_CANDIDATES,   FM_NIGHT_COL,   label="feature matrix night")
    fm_event = find_col(fm, EVENT_CANDIDATES,   FM_EVENT_COL,   required=False, label="feature matrix event")
    si_subj  = find_col(si, SUBJECT_CANDIDATES, SI_SUBJECT_COL, label="subjects-info subject")
    si_gen   = find_col(si, GENDER_CANDIDATES,  SI_GENDER_COL,  required=False, label="subjects-info gender")
    si_age   = find_col(si, AGE_CANDIDATES,     SI_AGE_COL,     required=False, label="subjects-info age")

    print("=" * 70)
    print(f"Feature matrix : {fm.shape[0]} rijen x {fm.shape[1]} kolommen")
    print(f"  kolommen     : {list(fm.columns)}")
    print(f"  -> subject='{fm_subj}', night='{fm_night}', event='{fm_event}'")
    print(f"Subjects info  : {si.shape[0]} rijen x {si.shape[1]} kolommen")
    print(f"  kolommen     : {list(si.columns)}")
    print(f"  -> subject='{si_subj}', gender='{si_gen}', age='{si_age}'")
    print(f"  voorbeeld IDs FM: {fm[fm_subj].dropna().unique()[:5].tolist()}")
    print(f"  voorbeeld IDs SI: {si[si_subj].dropna().unique()[:5].tolist()}")
    print("=" * 70)
    if inspect_only:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- events per nacht / subject ----------
    fm["_sid"] = fm[fm_subj].map(norm_id)
    fm["_nid"] = fm[fm_subj].astype(str) + "__" + fm[fm_night].astype(str)  # nacht uniek binnen subject

    if fm_event:
        per_night = fm.groupby(["_sid", fm_night], dropna=False)[fm_event].nunique()
    else:
        per_night = fm.groupby(["_sid", fm_night], dropna=False).size()
    per_night = per_night.rename("n_events").reset_index().rename(columns={"_sid": "subject"})

    per_subject = (per_night.groupby("subject")
                   .agg(n_nights=(fm_night, "nunique"),
                        n_events=("n_events", "sum"),
                        events_per_night_mean=("n_events", "mean"),
                        events_per_night_median=("n_events", "median"),
                        events_per_night_min=("n_events", "min"),
                        events_per_night_max=("n_events", "max"))
                   .reset_index())

    # ---------- subjects-info: 1 rij per subject ----------
    si["_sid"] = si[si_subj].map(norm_id)
    keep = {}
    if si_gen:
        keep["gender"] = si_gen
    if si_age:
        si[si_age] = pd.to_numeric(si[si_age].astype(str).str.replace(",", "."), errors="coerce")
        keep["age"] = si_age

    if keep:
        si_sub = si[["_sid"] + list(keep.values())].rename(columns={v: k for k, v in keep.items()})
        # Check op inconsistenties bij meerdere timepoints per subject
        for col in keep:
            n_var = si_sub.dropna(subset=[col]).groupby("_sid")[col].nunique()
            bad = n_var[n_var > 1]
            if len(bad):
                print(f"[LET OP] {len(bad)} subjects hebben meerdere waarden voor '{col}' "
                      f"(eerste niet-lege wordt gebruikt): {bad.index.tolist()[:10]}")
        si_one = si_sub.groupby("_sid").first().reset_index().rename(columns={"_sid": "subject"})
    else:
        si_one = pd.DataFrame({"subject": si["_sid"].unique()})

    # ---------- koppelen ----------
    overview = per_subject.merge(si_one, on="subject", how="left", indicator=True)
    unmatched = overview.loc[overview["_merge"] == "left_only", "subject"]
    overview = overview.drop(columns="_merge").sort_values("subject", key=lambda s: pd.to_numeric(s, errors="coerce"))

    print(f"Subjects in feature matrix : {len(per_subject)}")
    print(f"  gematcht met subjects-info: {len(per_subject) - len(unmatched)}")
    if len(unmatched):
        print(f"  [LET OP] niet gematcht ({len(unmatched)}): {unmatched.tolist()}")
        unmatched.to_frame().to_csv(OUT_DIR / "unmatched_subjects.csv", index=False, sep=OUT_SEP)

    # ---------- samenvattende tabel ----------
    rows = [
        {"Variabele": "Subjects", "N": len(per_subject), "Waarde": ""},
        {"Variabele": "Nachten", "N": len(per_night), "Waarde": ""},
        {"Variabele": "Arousal events", "N": int(per_night["n_events"].sum()), "Waarde": ""},
        describe(per_subject["n_nights"], "Nachten per subject"),
        describe(per_subject["n_events"], "Events per subject"),
        describe(per_night["n_events"], "Events per nacht"),
    ]
    if "age" in overview:
        rows.append(describe(overview["age"], "Leeftijd (jaren)"))
    if "gender" in overview:
        counts = overview["gender"].fillna("onbekend").value_counts()
        for g, n in counts.items():
            rows.append({"Variabele": f"Gender: {g}", "N": int(n),
                         "Waarde": f"{100 * n / len(overview):.1f}%"})
        # events/nachten per gender
        for g, grp in overview.groupby(overview["gender"].fillna("onbekend")):
            rows.append({"Variabele": f"  {g}: nachten / events", "N": len(grp),
                         "Waarde": f"{int(grp['n_nights'].sum())} nachten, {int(grp['n_events'].sum())} events"})
    summary = pd.DataFrame(rows)

    # ---------- opslaan ----------
    per_night.rename(columns={fm_night: "night"}).to_csv(
        OUT_DIR / "per_night_overview.csv", index=False, sep=OUT_SEP, decimal=OUT_DECIMAL)
    overview.round(2).to_csv(OUT_DIR / "per_subject_overview.csv", index=False, sep=OUT_SEP, decimal=OUT_DECIMAL)
    summary.to_csv(OUT_DIR / "summary_table.csv", index=False, sep=OUT_SEP, decimal=OUT_DECIMAL)

    pd.set_option("display.width", 160)
    print("\n" + summary.to_string(index=False))
    print(f"\nOutputs opgeslagen in: {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="toon alleen kolommen en gedetecteerde ID-kolommen")
    main(inspect_only=ap.parse_args().inspect)