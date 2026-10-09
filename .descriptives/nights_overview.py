"""
nights_overview.py

Beschrijft alle nachten die in de reduced feature matrix zitten:
slaaparchitectuur (uit de R&K-hypnogrammen in sleepArchitecture/) + arousal-maten.

Belangrijk:
  - Arousal-index = arousals / uur TOTAL SLEEP TIME (TST = N1+N2+N3+REM epochs).
    Wake (0) en unscored/artefact (-1) epochs zitten NIET in de noemer,
    consistent met het uitsluiten van wake-events in de feature matrix (AASM-definitie).
  - Geclipte nachten (CLIP_NIGHTS) worden in het hypnogram op dezelfde epoch afgekapt.
  - Arousal-aantallen = de events in de analyse (na 3-15 s filter en wake-exclusie).

Hypnogram-pad:
  RAW_ROOT/<GROUP>/<subject>/<subject>_<night>/sleepArchitecture/<subject>_<night>.csv
  bv. .../NSR/bnbd_nsr_17598/bnbd_nsr_17598_T0_N1/sleepArchitecture/bnbd_nsr_17598_T0_N1.csv

Outputs (in OUT_DIR):
  - nights_overview.csv      : 1 rij per nacht, geselecteerde maten (META_COLS + FEATURE_COLS), 2 decimalen (per_min: 3)
  - nights_summary_table.csv : M, SD, Mdn, IQR, range per maat over nachten
  - nights_unmatched.csv     : nachten zonder gevonden hypnogram (indien aanwezig)

Gebruik:
  python nights_overview.py --inspect   # kolommen, paden, matching + 1 test-hypnogram
  python nights_overview.py             # volledige run
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
BASE = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2")

FEATURES_CSV      = BASE / "reduced feature matrix" / "arousal_features_reduced_scaled.csv"
ORIG_FEATURES_CSV = BASE / "reduced feature matrix" / "arousal_features_reduced_original.csv"  # ongeschaald (duur in s)

RAW_ROOT = Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg")
GROUPS   = ["NSR"]

EPOCH_S = 30

# Geclipte nachten: (subject, nacht) -> aantal epochs dat behouden blijft (labels[:N])
CLIP_NIGHTS: dict[tuple[str, str], int] = {
    ("bnbd_nsr_17598", "T0_N1"): 1011,
    ("bnbd_nsr_16379", "T0_N1"): 990,
    ("bnbd_nsr_19611", "T0_N1"): 910,
}

# Stadiumcodes -> labels. R&K: S3+S4 samen = N3.
STAGE_MAP = {
    0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "N3", 5: "REM", -1: "UNS",
    "W": "W", "WAKE": "W", "N1": "N1", "N2": "N2", "N3": "N3", "N4": "N3",
    "S1": "N1", "S2": "N2", "S3": "N3", "S4": "N3", "R": "REM", "REM": "REM",
    "A": "UNS", "ART": "UNS", "?": "UNS", "U": "UNS",
}
SLEEP_STAGES = ["N1", "N2", "N3", "REM"]

# Kolommen in de feature matrix (None = auto-detectie)
FM_SUBJECT_COL = "subject_id"
FM_NIGHT_COL   = "night_id"
FM_ONSET_COL   = "start_sec"
ONSET_UNIT_S   = 1.0       # 1.0 = seconden; 1/128 = samples bij 128 Hz; 30 = epochs
ORIG_DURATION_COL = "duration_sec"   # auto: 'duration...'

OUT_DIR = BASE / "nights overview"
OUT_SEP, OUT_DECIMAL = ";", ","
N_DECIMALS = 2                                  # standaard: 2 decimalen (decimale komma, geen duizendtal-scheiding)
DECIMALS_OVERRIDE = {"arousals_per_min_tst": 3}   # uitzonderingen per kolom

# Kolommen die per nacht bewaard worden (in deze volgorde)
META_COLS = ["subject", "night", "clipped_at_epoch",
             "recording_min_edf", "recording_min_hypno", "unscored_min",
             "spt_min", "tst_min", "waso_min",
             "N1_min", "N2_min", "N3_min", "REM_min"]
FEATURE_COLS = ["n_arousals", "arousal_index_per_h_tst", "arousals_per_min_tst",
                "arousal_dur_mean_s", "arousal_dur_median_s",
                "arousal_time_total_min", "arousal_time_pct_tst"]

# ============================================================
# HELPERS
# ============================================================
SUBJECT_CANDIDATES = ["subject", "subject_id", "subj", "participant", "participant_id", "pp", "ppn", "id"]
NIGHT_CANDIDATES   = ["night", "night_id", "nacht", "recording", "session", "edf", "file", "filename"]
ONSET_CANDIDATES   = ["onset_s", "onset", "start_s", "start", "t_start", "onset_sec", "event_start"]
DUR_CANDIDATES     = ["duration_s", "duration", "dur_s", "dur"]


def read_csv_auto(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(path, sep=sep, decimal="," if sep == ";" else ".",
                     encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df, candidates, override=None, required=True, label=""):
    if override is not None:
        if override not in df.columns:
            raise KeyError(f"[{label}] '{override}' niet gevonden. Beschikbaar: {list(df.columns)}")
        return override
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for cand in candidates:
        for lc, orig in lower.items():
            if cand in lc:
                return orig
    if required:
        raise KeyError(f"[{label}] geen kolom voor {candidates}. Zet handmatig in CONFIG. "
                       f"Beschikbaar: {list(df.columns)}")
    return None


def norm_subject(x) -> str:
    """-> 'bnbd_nsr_17598'. Accepteert ook volledige paden/IDs of alleen het nummer."""
    s = str(x).strip().lower()
    m = re.search(r"bnbd_[a-z]+_\d{5}", s)
    if m:
        return m.group(0)
    d = re.findall(r"\d+", s)
    if d:  # alleen nummer -> eerste groep uit GROUPS
        return f"bnbd_{GROUPS[0].lower()}_{int(d[0]):05d}"
    return s


def norm_night(x) -> str:
    """-> 'T0_N1'. Accepteert 'T0_N1', 'N1', 1, of een volledige ID."""
    s = str(x).strip().upper()
    m = re.search(r"(T\d+)_N(\d+)", s)
    if m:
        return f"{m.group(1)}_N{int(m.group(2))}"
    m = re.search(r"N(\d+)", s) or re.fullmatch(r"(\d+)(?:\.0)?", s)
    if m:
        return f"T0_N{int(m.group(1))}"
    return s


def night_dir(subject: str, night: str):
    for g in GROUPS:
        d = RAW_ROOT / g / subject / f"{subject}_{night}"
        if d.exists():
            return d
    return None


def hypno_path(subject: str, night: str):
    d = night_dir(subject, night)
    if d is None:
        return None
    p = d / "sleepArchitecture" / f"{subject}_{night}.csv"
    if p.exists():
        return p
    hits = sorted((d / "sleepArchitecture").glob("*.csv")) if (d / "sleepArchitecture").exists() else []
    return hits[0] if len(hits) == 1 else None


def edf_duration_min(subject: str, night: str):
    """Opnameduur uit de EDF-header van 'EEG L.edf' (zonder het signaal te laden)."""
    d = night_dir(subject, night)
    if d is None:
        return np.nan
    p = d / f"{subject}_{night}_edf" / "EEG L.edf"
    if not p.exists():
        return np.nan
    try:
        with open(p, "rb") as f:
            hdr = f.read(256)
        n_rec = int(hdr[236:244].decode().strip())
        rec_dur = float(hdr[244:252].decode().strip())
        return n_rec * rec_dur / 60
    except Exception:
        return np.nan


def to_label(v):
    if pd.isna(v):
        return "UNS"
    if isinstance(v, str):
        s = v.strip().upper()
        try:
            v = int(float(s.replace(",", ".")))
        except ValueError:
            return STAGE_MAP.get(s, "UNS")
    return STAGE_MAP.get(int(v), "UNS")


def load_hypnogram(path: Path) -> np.ndarray:
    """Array met labels per epoch. Ondersteunt 1-kolom CSV, tabel met stage-kolom en JSON."""
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            key = next((k for k in obj if re.search(r"stage|hypno", k, re.I)), None)
            obj = obj[key] if key else next(v for v in obj.values() if isinstance(v, list))
        return np.array([to_label(v) for v in obj])

    df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#", dtype=str)
    first = df.iloc[0].astype(str).str.lower()
    stage_hdr = [i for i, v in enumerate(first) if re.search(r"stage|hypno|stadium|score", v)]
    if stage_hdr:
        col = df.iloc[1:, stage_hdr[0]]
    elif df.shape[1] == 1:
        col = df.iloc[:, 0]
    else:
        valid = {str(k) for k in STAGE_MAP}
        fracs = [np.mean([str(v).strip().upper() in valid for v in df[c]]) for c in df.columns]
        col = df.iloc[:, int(np.argmax(fracs))]
    col = col[col.astype(str).str.strip().str.upper().isin({str(k) for k in STAGE_MAP})
              | col.astype(str).str.strip().str.match(r"^-?\d+(\.0)?$")]  # header/rommel eruit
    return np.array([to_label(v) for v in col])


# ============================================================
# MATEN PER NACHT
# ============================================================
def sleep_metrics(labels: np.ndarray) -> dict:
    ep_min = EPOCH_S / 60
    n = len(labels)
    is_sleep = np.isin(labels, SLEEP_STAGES)
    out = {"recording_min_hypno": n * ep_min,
           "unscored_min": np.sum(labels == "UNS") * ep_min}
    if not is_sleep.any():
        return out
    first, last = np.flatnonzero(is_sleep)[[0, -1]]
    spt = labels[first:last + 1]
    tst = is_sleep.sum() * ep_min
    out.update({
        "sleep_onset_latency_min": first * ep_min,
        "spt_min": len(spt) * ep_min,
        "tst_min": tst,
        "waso_min": np.sum(spt == "W") * ep_min,
        "unscored_in_spt_min": np.sum(spt == "UNS") * ep_min,
        "wake_after_final_awakening_min": (n - 1 - last) * ep_min,
        "sleep_efficiency_pct": 100 * tst / (n * ep_min),
        "sleep_maintenance_eff_pct": 100 * tst / (len(spt) * ep_min),
    })
    for st in SLEEP_STAGES:
        m = np.sum(labels == st) * ep_min
        out[f"{st}_min"] = m
        out[f"{st}_pct_tst"] = 100 * m / tst
    out["n_awakenings"] = int(np.sum((spt[1:] == "W") & (spt[:-1] != "W")))
    shifts = int(np.sum(spt[1:] != spt[:-1]))
    out["stage_shifts"] = shifts
    out["stage_shifts_per_h_tst"] = shifts / (tst / 60)
    return out


def arousal_metrics(ev: pd.DataFrame, labels, sm: dict, onset_col, dur_col) -> dict:
    n = len(ev)
    out = {"n_arousals": n}
    tst = sm.get("tst_min", np.nan)
    if tst and tst > 0:
        out["arousal_index_per_h_tst"] = n / (tst / 60)
        out["arousals_per_min_tst"] = n / tst
    if "spt_min" in sm:
        out["arousals_per_h_spt"] = n / (sm["spt_min"] / 60)

    if onset_col and labels is not None and n:
        on_s = pd.to_numeric(ev[onset_col], errors="coerce") * ONSET_UNIT_S
        ep = (on_s // EPOCH_S)
        st = np.array([labels[int(e)] if pd.notna(e) and 0 <= e < len(labels) else "OUT" for e in ep])
        for s in SLEEP_STAGES:
            k = int(np.sum(st == s))
            out[f"n_arousals_{s}"] = k
            mins = sm.get(f"{s}_min", 0)
            out[f"arousal_index_{s}_per_h"] = k / (mins / 60) if mins > 0 else np.nan
        nrem_min = sum(sm.get(f"{s}_min", 0) for s in ["N1", "N2", "N3"])
        k_nrem = int(np.isin(st, ["N1", "N2", "N3"]).sum())
        out["arousal_index_NREM_per_h"] = k_nrem / (nrem_min / 60) if nrem_min > 0 else np.nan
        out["n_arousals_in_W_UNS_or_out"] = int(np.isin(st, ["W", "UNS", "OUT"]).sum())  # controle: ~0
        iai = np.diff(np.sort(on_s.dropna().values)) / 60
        if len(iai):
            out["inter_arousal_interval_median_min"] = float(np.median(iai))
        if "sleep_onset_latency_min" in sm:
            mid_s = (sm["sleep_onset_latency_min"] + sm["spt_min"] / 2) * 60
            out["pct_arousals_first_half_spt"] = 100 * np.mean(on_s.dropna() < mid_s)

    if dur_col and n:
        d = pd.to_numeric(ev[dur_col], errors="coerce").dropna()
        if len(d):
            out["arousal_dur_mean_s"] = d.mean()
            out["arousal_dur_median_s"] = d.median()
            out["arousal_time_total_min"] = d.sum() / 60
            if tst and tst > 0:
                out["arousal_time_pct_tst"] = 100 * (d.sum() / 60) / tst
    return out


def fmt_num(v, d: int) -> str:
    """Vaste notatie met d decimalen en decimale komma; leeg bij NaN. Nooit duizendtal-scheiding."""
    if pd.isna(v):
        return ""
    return f"{float(v):.{d}f}".replace(".", OUT_DECIMAL)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.select_dtypes(include="number").columns:
        s = df[c].dropna()
        if s.empty:
            continue
        rows.append({"Maat": c, "N_nachten": len(s), "M": s.mean(), "SD": s.std(ddof=1),
                     "Mdn": s.median(), "Q1": s.quantile(.25), "Q3": s.quantile(.75),
                     "Min": s.min(), "Max": s.max()})
    return pd.DataFrame(rows)


def attach_duration(fm, subj_col, night_col, onset_col):
    """Koppelt duur (s) uit de ongeschaalde matrix: op rij-positie als die identiek is, anders op onset."""
    if ORIG_FEATURES_CSV is None or not Path(ORIG_FEATURES_CSV).exists():
        return fm, None
    orig = read_csv_auto(Path(ORIG_FEATURES_CSV))
    o_dur = find_col(orig, DUR_CANDIDATES, ORIG_DURATION_COL, required=False, label="orig duration")
    if o_dur is None:
        print(f"[INFO] geen duur-kolom in ongeschaalde matrix (kolommen: {list(orig.columns)}) "
              f"-> arousal-duur wordt overgeslagen")
        return fm, None
    o_subj = find_col(orig, SUBJECT_CANDIDATES, FM_SUBJECT_COL, label="orig subject")
    o_night = find_col(orig, NIGHT_CANDIDATES, FM_NIGHT_COL, label="orig night")

    same_rows = (len(orig) == len(fm)
                 and (orig[o_subj].astype(str).values == fm[subj_col].astype(str).values).all()
                 and (orig[o_night].astype(str).values == fm[night_col].astype(str).values).all())
    if same_rows:
        fm["_dur_s"] = pd.to_numeric(orig[o_dur], errors="coerce").values
        print(f"Duur gekoppeld op rij-positie (bestanden zijn rij-voor-rij identiek), kolom '{o_dur}'")
    elif onset_col and onset_col in orig.columns:
        key = ["_s", "_n", "_on"]
        for d, sc, nc in ((fm, subj_col, night_col), (orig, o_subj, o_night)):
            d["_s"], d["_n"] = d[sc].map(norm_subject), d[nc].map(norm_night)
            d["_on"] = pd.to_numeric(d[onset_col], errors="coerce").round(3)
        fm = fm.merge(orig[key + [o_dur]].rename(columns={o_dur: "_dur_s"}), on=key, how="left")
        fm["_dur_s"] = pd.to_numeric(fm["_dur_s"], errors="coerce")
        print(f"Duur gekoppeld op (subject, nacht, onset): {fm['_dur_s'].notna().mean():.1%} van de events")
    else:
        print("[LET OP] ongeschaalde matrix niet rij-identiek en geen onset om op te koppelen -> duur overgeslagen")
        return fm, None
    return fm, "_dur_s"


# ============================================================
# MAIN
# ============================================================
def main(inspect_only=False):
    fm = read_csv_auto(FEATURES_CSV)
    subj_col = find_col(fm, SUBJECT_CANDIDATES, FM_SUBJECT_COL, label="subject")
    night_col = find_col(fm, NIGHT_CANDIDATES, FM_NIGHT_COL, label="night")
    onset_col = find_col(fm, ONSET_CANDIDATES, FM_ONSET_COL, required=False, label="onset")

    fm["_subject"] = fm[subj_col].map(norm_subject)
    fm["_night"] = fm[night_col].map(norm_night)
    fm, dur_col = attach_duration(fm, subj_col, night_col, onset_col)

    nights = fm[["_subject", "_night"]].drop_duplicates().reset_index(drop=True)
    nights["_hypno"] = [hypno_path(s, n) for s, n in zip(nights["_subject"], nights["_night"])]

    print("=" * 70)
    print(f"Feature matrix: {len(fm)} events, {nights['_subject'].nunique()} subjects, {len(nights)} nachten")
    print(f"  subject='{subj_col}', night='{night_col}', onset='{onset_col}', duur='{dur_col}'")
    print(f"  voorbeeld ruw -> genormaliseerd: "
          f"{fm[[subj_col, night_col]].iloc[0].tolist()} -> {nights.iloc[0, :2].tolist()}")
    print(f"  hypnogram gevonden: {nights['_hypno'].notna().sum()} / {len(nights)}")
    for _, m in nights[nights["_hypno"].isna()].iterrows():
        print(f"    ontbreekt: {m['_subject']} {m['_night']}  "
              f"(map: {night_dir(m['_subject'], m['_night'])})")
    missing_clip = [k for k in CLIP_NIGHTS if k not in set(zip(nights["_subject"], nights["_night"]))]
    if missing_clip:
        print(f"  [LET OP] CLIP_NIGHTS niet teruggevonden in feature matrix: {missing_clip}")

    if inspect_only:
        ok = nights.dropna(subset=["_hypno"])
        if len(ok):
            r = ok.iloc[0]
            lab = load_hypnogram(r["_hypno"])
            print(f"\nTest-hypnogram: {r['_hypno']}")
            print(f"  {len(lab)} epochs = {len(lab) * EPOCH_S / 60:.1f} min  |  "
                  f"EDF-duur: {edf_duration_min(r['_subject'], r['_night']):.1f} min")
            print("  stadia:", pd.Series(lab).value_counts().to_dict())
            print("  eerste 20 epochs:", lab[:20].tolist())
            with open(r["_hypno"], "r", encoding="utf-8-sig", errors="replace") as f:
                print("  ruwe eerste 5 regels:", [next(f).rstrip() for _ in range(5)])
        else:
            print(f"\n[LET OP] geen hypnogram gevonden. Eerste verwacht pad: "
                  f"{RAW_ROOT / GROUPS[0] / nights['_subject'].iloc[0] / (nights['_subject'].iloc[0] + '_' + nights['_night'].iloc[0])}")
        print("=" * 70)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in nights.iterrows():
        s, n = r["_subject"], r["_night"]
        hp = r["_hypno"] if isinstance(r["_hypno"], Path) else None   # pandas maakt van None soms NaN
        ev = fm[(fm["_subject"] == s) & (fm["_night"] == n)]
        clip = CLIP_NIGHTS.get((s, n))
        row = {"subject": s, "night": n,
               "hypnogram_found": hp is not None,
               "clipped_at_epoch": clip,
               "recording_min_edf": edf_duration_min(s, n)}
        labels, sm = None, {}
        if hp is not None:
            labels = load_hypnogram(hp)
            if clip is not None:
                labels = labels[:clip]
            sm = sleep_metrics(labels)
        row.update(sm)
        row.update(arousal_metrics(ev, labels, sm, onset_col, dur_col))
        rows.append(row)

    ov = pd.DataFrame(rows).sort_values(["subject", "night"]).reset_index(drop=True)
    ov.insert(2, "n_nights_subject", ov.groupby("subject")["night"].transform("count"))

    unmatched = ov[~ov["hypnogram_found"]]
    if len(unmatched):
        print(f"[LET OP] {len(unmatched)} nachten zonder hypnogram -> alleen n_arousals berekend")
        unmatched[["subject", "night"]].to_csv(OUT_DIR / "nights_unmatched.csv", index=False, sep=OUT_SEP)
    if "n_arousals_in_W_UNS_or_out" in ov and ov["n_arousals_in_W_UNS_or_out"].sum() > 0:
        bad = ov[ov["n_arousals_in_W_UNS_or_out"] > 0]
        print(f"[LET OP] {int(bad['n_arousals_in_W_UNS_or_out'].sum())} events in {len(bad)} nachten vallen "
              f"volgens het hypnogram in W/UNS of buiten de opname -> check ONSET_UNIT_S of clipping")
    diff = (ov["recording_min_edf"] - ov.get("recording_min_hypno", np.nan)).abs()
    if (diff > 1).any():
        print(f"[INFO] {int((diff > 1).sum())} nachten: EDF-duur en hypnogram-duur verschillen > 1 min "
              f"(o.a. geclipte nachten en een afgekapte laatste epoch zijn normaal)")

    # alleen de gewenste kolommen (ontbrekende worden gemeld en overgeslagen)
    keep = [c for c in META_COLS + FEATURE_COLS if c in ov.columns]
    missing = [c for c in META_COLS + FEATURE_COLS if c not in ov.columns]
    if missing:
        print(f"[INFO] kolommen niet berekend en dus weggelaten: {missing}")
    ov = ov[keep].copy()
    for c in ["clipped_at_epoch", "n_arousals"]:          # gehele getallen zonder ',0'
        if c in ov:
            ov[c] = ov[c].astype("Int64")

    summ = summarize(ov.drop(columns=["clipped_at_epoch"], errors="ignore"))
    header = pd.DataFrame([
        {"Maat": "Aantal nachten", "N_nachten": len(ov)},
        {"Maat": "Aantal subjects", "N_nachten": ov["subject"].nunique()},
        {"Maat": "Nachten met hypnogram", "N_nachten": int(len(ov) - len(unmatched))},
        {"Maat": "Geclipte nachten", "N_nachten": int(ov["clipped_at_epoch"].notna().sum())},
        {"Maat": "Totaal arousals", "N_nachten": int(ov["n_arousals"].sum())},
    ])
    summ = pd.concat([header, summ], ignore_index=True)

    summ["N_nachten"] = summ["N_nachten"].astype("Int64")
    # per rij het juiste aantal decimalen (bv. 3 voor arousals_per_min_tst)
    for c in ["M", "SD", "Mdn", "Q1", "Q3", "Min", "Max"]:
        summ[c] = [fmt_num(v, DECIMALS_OVERRIDE.get(m, N_DECIMALS)) for v, m in zip(summ[c], summ["Maat"])]
    # uitzonderingskolommen in de per-nacht tabel apart formatteren; de rest via float_format
    for c, d in DECIMALS_OVERRIDE.items():
        if c in ov:
            ov[c] = ov[c].map(lambda v, d=d: fmt_num(v, d))
    fmt = f"%.{N_DECIMALS}f"   # vaste notatie: geen duizendtal-scheiding, geen wetenschappelijke notatie
    ov.to_csv(OUT_DIR / "nights_overview.csv", index=False, sep=OUT_SEP, decimal=OUT_DECIMAL, float_format=fmt)
    summ.to_csv(OUT_DIR / "nights_summary_table.csv", index=False, sep=OUT_SEP, decimal=OUT_DECIMAL, float_format=fmt)

    pd.set_option("display.width", 200, "display.max_rows", 200)
    print("\n" + summ.to_string(index=False))
    print(f"\nOutputs opgeslagen in: {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    main(inspect_only=ap.parse_args().inspect)