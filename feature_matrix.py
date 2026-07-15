"""
=============================================================================
feature_matrix.py

Bouwt een featurematrix (1 rij per gescoord arousal-event) vanuit:
  1. Lucija's gescoorde arousal-events  (in de sleepArchitecture map)
  2. De ruwe EDF-kanalen van diezelfde nacht (EEG L, EEG R, dX/dY/dZ, evt. OXY_IR_AC)

Verwachte mapstructuur (subject_id en night_id zijn VARIABEL, dus we zoeken
ze dynamisch op i.p.v. hardcoded):

  RAW_ROOT/
    GROUP/                                  bv. NSR, SAV, Prezens
      bnbd_<groep>_XXXXX/                   XXXXX = 5-cijferig subject nummer
        bnbd_<groep>_XXXXX_T0_N#/           N# = nacht-nummer, variabel
          bnbd_<groep>_XXXXX_T0_N#_edf/
            EEG L.edf
            EEG R.edf
            dX.edf
            dY.edf
            dZ.edf
            OXY_IR_AC.edf                   (optioneel)
          sleepArchitecture/
            bnbd_<groep>_XXXXX_T0_N#.csv           <- hypnogram (R&K stages)  [optioneel]
            bnbd_<groep>_XXXXX_T0_N#_events.csv    <- Lucija's gescoorde events

We zoeken ALLE "sleepArchitecture" mappen onder RAW_ROOT via rglob, dus het
maakt niet uit hoe het subject-nummer of het nacht-nummer precies heet.
De bijbehorende EDF-kanalen worden gezocht in de submap "<stem>_edf" naast
de sleepArchitecture-map (met fallback naar de nachtmap zelf als die submap
er toch niet is).

BELANGRIJK — nog te verifiëren aannames:
  - De naam van het events-bestand bevat "_events" (zoals eerder gezien:
    bnbd_nsr_01272_T0_N3_events.csv, kolommen: event, start, stop, duration, channel)
  - Het hypnogram-bestand heeft dezelfde naam als de nacht-map zelf, zonder
    "_events" suffix (bv. bnbd_nsr_01272_T0_N2.csv)
  - Als deze aannames niet kloppen, draai dit script eerst met --inspect
    zodat je precies ziet welke bestanden er per nacht gevonden worden,
    voordat je de volledige featureberekening draait.

Gebruik:
  python build_feature_matrix.py --inspect --limit 5     # eerst checken
  python build_feature_matrix.py --limit 5                # test op 5 nachten
  python build_feature_matrix.py                          # volledige run
=============================================================================
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, filtfilt, hilbert, iirnotch

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# =============================================================================
# CONFIGURATIE
# =============================================================================

RAW_ROOT   = Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg")
GROUPS     = ["NSR", "Prezens", "SAV"]
EVENTS_DIR = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices")

TARGET_SFREQ = 128.0          # Hz, zelfde als in de detectiepipeline
NOTCH_HZ = 50.0
HIGHPASS_HZ = 0.1
LOWPASS_HZ = 35.0

BASELINE_SEC = 90.0            # causale baseline-window vóór event-onset
EEG_CHANNELS = ["EEG L", "EEG R"]
MOTION_CHANNELS = ["dX", "dY", "dZ"]
OXY_CHANNEL = "OXY_IR_AC"      # optioneel, wordt geladen indien aanwezig

BANDS = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}

# =============================================================================
# SECTIE 1 — NACHTEN VINDEN EN IDENTIFICEREN
# =============================================================================

def find_night_dirs(raw_root: Path) -> list[Path]:
    """
    Zoekt alle nacht-mappen door te zoeken naar 'sleepArchitecture' submappen.
    Dit omzeilt het probleem dat subject-nummer en nacht-nummer variabel zijn.
    Filtert daarna op GROUPS, en houdt alleen T0_N# nachten over (geen T1, T2, ...).
    """
    arch_dirs = sorted(raw_root.rglob("sleepArchitecture"))
    night_dirs = [d.parent for d in arch_dirs if d.is_dir()]

    filtered = []
    for nd in night_dirs:
        group_in_path = next((g for g in GROUPS if g.upper() in [p.upper() for p in nd.parts]), None)
        if group_in_path is None:
            continue
        if not re.search(r"_T0_N\d+$", nd.name):
            continue
        filtered.append(nd)
    return filtered


def parse_ids(night_dir: Path) -> dict:
    """
    Parseert group / subject_id / night_id robuust uit de mapnaam,
    ongeacht het exacte subject-nummer of nacht-nummer.

    Voorbeeld: bnbd_nsr_01272_T0_N2  ->
      group      = NSR
      subject_id = bnbd_nsr_01272
      night_id   = T0_N2
    """
    stem = night_dir.name  # bv. "bnbd_nsr_01272_T0_N2"

    m = re.match(r"(bnbd_([a-zA-Z]+)_\d+)_((?:T\d+)_(?:N\d+))", stem)
    if m:
        subject_id = m.group(1)
        group = m.group(2).upper()
        night_id = m.group(3)
    else:
        # Fallback: minder strikt, gewoon op underscores splitsen
        parts = stem.split("_")
        subject_id = "_".join(parts[:3])
        night_id = "_".join(parts[3:])
        group = parts[1].upper() if len(parts) > 1 else "UNKNOWN"

    return {"subject_id": subject_id, "night_id": night_id, "group": group, "stem": stem}


# =============================================================================
# SECTIE 2 — EVENTS EN HYPNOGRAM INLADEN
# =============================================================================

def find_events_file(arch_dir: Path, stem: str) -> Path | None:
    """Zoekt het bestand met Lucija's gescoorde arousal-events."""
    candidates = sorted(arch_dir.glob("*_events.csv")) or sorted(arch_dir.glob("*events*.csv"))
    if candidates:
        return candidates[0]
    return None


def find_hypnogram_file(arch_dir: Path, stem: str) -> Path | None:
    """Zoekt het hypnogram-bestand (R&K sleep stages), indien aanwezig."""
    exact = arch_dir / f"{stem}.csv"
    if exact.exists():
        return exact
    # Fallback: elk ander csv-bestand in de map dat NIET het events-bestand is
    others = [f for f in arch_dir.glob("*.csv") if "event" not in f.name.lower()]
    return others[0] if others else None


def _read_csv_flex(path: Path) -> pd.DataFrame:
    """Leest csv, probeert zowel komma- als puntkomma-scheiding."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_events(path: Path) -> pd.DataFrame:
    """
    Normaliseert Lucija's events-bestand naar kolommen: start_sec, end_sec, duration_sec.
    Vast formaat: event, start, stop, duration, channel.
    """
    df = _read_csv_flex(path)

    df = df.rename(columns={"start": "start_sec", "stop": "end_sec", "duration": "duration_sec"})

    keep_cols = ["start_sec", "end_sec", "duration_sec"]
    if "channel" in df.columns:
        keep_cols.append("channel")
    if "event" in df.columns:
        keep_cols.append("event")
    return df[keep_cols].reset_index(drop=True)


def load_hypnogram(path: Path) -> pd.DataFrame | None:
    """
    Laadt het hypnogram, indien aanwezig. Verwacht kolommen die epoch-tijd
    en R&K-stage aangeven; namen kunnen variëren dus we zoeken op trefwoorden.
    Geeft None terug als het bestand niet gevonden of niet leesbaar is.
    """
    if path is None:
        return None
    try:
        df = _read_csv_flex(path)
    except Exception:
        return None

    onset_col = next((c for c in df.columns if c in ("onset", "start", "start_sec", "time")), None)
    stage_col = next((c for c in df.columns if "stage" in c or c in ("rk", "score")), None)

    if onset_col is None or stage_col is None:
        return None

    df = df.rename(columns={onset_col: "onset_sec", stage_col: "stage_rk"})
    return df[["onset_sec", "stage_rk"]].sort_values("onset_sec").reset_index(drop=True)


def get_stage_at(hypnogram: pd.DataFrame | None, t_sec: float):
    """Geeft de R&K-stage terug die geldt op tijdstip t_sec, of NaN."""
    if hypnogram is None or len(hypnogram) == 0:
        return np.nan
    idx = hypnogram["onset_sec"].searchsorted(t_sec, side="right") - 1
    if idx < 0:
        return np.nan
    return hypnogram["stage_rk"].iloc[idx]


# =============================================================================
# SECTIE 3 — EDF KANALEN INLADEN
# =============================================================================

def get_edf_dir(night_dir: Path, stem: str) -> Path:
    """
    Geeft de map terug waarin de EDF-kanaalbestanden staan.
    Structuur: night_dir / <stem>_edf / EEG L.edf, EEG R.edf, ...
    Valt terug op night_dir zelf als de _edf submap niet bestaat
    (voor het geval de structuur toch per nacht verschilt).
    """
    edf_dir = night_dir / f"{stem}_edf"
    return edf_dir if edf_dir.exists() else night_dir


def load_channel(edf_dir: Path, name: str) -> tuple[np.ndarray, float] | None:
    """Laadt één EDF-kanaalbestand, geeft (data_in_uV, sfreq) terug of None."""
    edf_path = edf_dir / f"{name}.edf"
    if not edf_path.exists():
        return None
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()[0] * 1e6  # V -> µV
    return data, raw.info["sfreq"]


def resample_to_target(data: np.ndarray, sfreq: float) -> np.ndarray:
    if abs(sfreq - TARGET_SFREQ) < 1e-6:
        return data
    n_target = int(round(len(data) * TARGET_SFREQ / sfreq))
    return mne.filter.resample(data, npad="auto", up=TARGET_SFREQ, down=sfreq)[:n_target] \
        if False else _resample_scipy(data, sfreq)


def _resample_scipy(data: np.ndarray, sfreq: float) -> np.ndarray:
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(TARGET_SFREQ), int(sfreq))
    up, down = int(TARGET_SFREQ // g), int(sfreq // g)
    return resample_poly(data, up, down)


def preprocess_eeg(data: np.ndarray, sfreq: float) -> np.ndarray:
    """DC-removal, notch, high-pass, low-pass, resample -> zelfde stappen als detectiepipeline."""
    data = data - np.median(data)

    b_notch, a_notch = iirnotch(NOTCH_HZ, Q=30, fs=sfreq)
    data = filtfilt(b_notch, a_notch, data)

    b_hp, a_hp = butter(4, HIGHPASS_HZ / (sfreq / 2), btype="high")
    data = filtfilt(b_hp, a_hp, data)

    b_lp, a_lp = butter(4, LOWPASS_HZ / (sfreq / 2), btype="low")
    data = filtfilt(b_lp, a_lp, data)

    return _resample_scipy(data, sfreq)


def band_envelope(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Bandpass filter + Hilbert-envelope voor één band."""
    lo, hi = band
    b, a = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype="band")
    filtered = filtfilt(b, a, data)
    return np.abs(hilbert(filtered))


def load_night_signals(night_dir: Path, stem: str) -> dict:
    """Laadt en preprocesst alle beschikbare kanalen voor één nacht."""
    signals = {}
    edf_dir = get_edf_dir(night_dir, stem)

    for ch in EEG_CHANNELS:
        loaded = load_channel(edf_dir, ch)
        if loaded is not None:
            data, sfreq = loaded
            signals[ch] = preprocess_eeg(data, sfreq)

    for ch in MOTION_CHANNELS:
        loaded = load_channel(edf_dir, ch)
        if loaded is not None:
            data, sfreq = loaded
            signals[ch] = _resample_scipy(data, sfreq)

    loaded = load_channel(edf_dir, OXY_CHANNEL)
    if loaded is not None:
        data, sfreq = loaded
        signals[OXY_CHANNEL] = _resample_scipy(data, sfreq)

    return signals


# =============================================================================
# SECTIE 4 — FEATURES PER EVENT BEREKENEN
# =============================================================================

def extract_event_features(signals: dict, start_sec: float, end_sec: float) -> dict:
    """
    Berekent features voor één event, gebaseerd op de EEG-signalen.
    Baseline = causale mediaan van de band-envelope in de BASELINE_SEC vóór het event.
    """
    feats = {}
    sf = TARGET_SFREQ

    start_i = int(start_sec * sf)
    end_i = int(end_sec * sf)
    base_start_i = max(0, int((start_sec - BASELINE_SEC) * sf))

    eeg_band_data = {}  # per kanaal, per band: (baseline_median, during_mean, during_peak, during_std)

    for ch in EEG_CHANNELS:
        if ch not in signals:
            continue
        sig = signals[ch]
        if end_i > len(sig) or start_i >= end_i or base_start_i >= start_i:
            continue

        for band_name, band_range in BANDS.items():
            baseline_segment = sig[base_start_i:start_i]
            event_segment = sig[start_i:end_i]

            env_baseline = band_envelope(baseline_segment, sf, band_range)
            env_event = band_envelope(event_segment, sf, band_range)

            baseline_med = np.median(env_baseline) if len(env_baseline) else np.nan
            during_mean = np.mean(env_event) if len(env_event) else np.nan
            during_peak = np.max(env_event) if len(env_event) else np.nan
            during_std = np.std(env_event) if len(env_event) else np.nan

            eeg_band_data[(ch, band_name)] = (baseline_med, during_mean, during_peak, during_std)

            ratio = during_mean / baseline_med if baseline_med not in (0, np.nan) and not np.isnan(baseline_med) else np.nan
            feats[f"{ch}_{band_name}_ratio"] = ratio
            feats[f"{ch}_{band_name}_peak_ratio"] = (
                during_peak / baseline_med if baseline_med not in (0, np.nan) and not np.isnan(baseline_med) else np.nan
            )
            feats[f"{ch}_{band_name}_cv"] = (
                during_std / during_mean if during_mean not in (0, np.nan) and not np.isnan(during_mean) else np.nan
            )

    # Gemiddelde over kanalen (voor als er maar 1 kanaal beschikbaar is, of ter samenvatting)
    for band_name in BANDS:
        ratios = [feats.get(f"{ch}_{band_name}_ratio") for ch in EEG_CHANNELS if f"{ch}_{band_name}_ratio" in feats]
        ratios = [r for r in ratios if r is not None and not (isinstance(r, float) and np.isnan(r))]
        feats[f"mean_{band_name}_ratio"] = np.mean(ratios) if ratios else np.nan

    # Bilateraliteit: hoe gelijk zijn L en R tijdens het event (alpha+beta band)
    if "EEG L" in signals and "EEG R" in signals:
        l_sig = signals["EEG L"][start_i:end_i] if end_i <= len(signals["EEG L"]) else None
        r_sig = signals["EEG R"][start_i:end_i] if end_i <= len(signals["EEG R"]) else None
        if l_sig is not None and r_sig is not None and len(l_sig) > 1 and len(r_sig) > 1:
            feats["bilateral_corr"] = np.corrcoef(l_sig, r_sig)[0, 1]
        else:
            feats["bilateral_corr"] = np.nan
    else:
        feats["bilateral_corr"] = np.nan

    # Rise time: tijd tot piek van de alpha+beta envelope binnen het event
    if "EEG L" in signals:
        sig = signals["EEG L"]
        if end_i <= len(sig) and start_i < end_i:
            seg = sig[start_i:end_i]
            env = band_envelope(seg, sf, (8.0, 30.0))
            peak_idx = np.argmax(env)
            feats["rise_time_sec"] = peak_idx / sf
        else:
            feats["rise_time_sec"] = np.nan
    else:
        feats["rise_time_sec"] = np.nan

    # Motion features (accelerometer), als proxy voor beweging tijdens het event
    motion_rms = []
    for ch in MOTION_CHANNELS:
        if ch in signals:
            sig = signals[ch]
            if end_i <= len(sig) and start_i < end_i:
                seg = sig[start_i:end_i]
                motion_rms.append(np.sqrt(np.mean(seg ** 2)))
    feats["motion_rms"] = np.mean(motion_rms) if motion_rms else np.nan

    # Pulse-oximetrie amplitude verandering (cardiovasculaire arousal proxy), indien beschikbaar
    if OXY_CHANNEL in signals:
        sig = signals[OXY_CHANNEL]
        if base_start_i < start_i and end_i <= len(sig) and start_i < end_i:
            baseline_amp = np.std(sig[base_start_i:start_i])
            event_amp = np.std(sig[start_i:end_i])
            feats["oxy_amp_ratio"] = event_amp / baseline_amp if baseline_amp else np.nan
        else:
            feats["oxy_amp_ratio"] = np.nan
    else:
        feats["oxy_amp_ratio"] = np.nan

    return feats


# =============================================================================
# SECTIE 5 — HOOFDLOOP
# =============================================================================

def process_night(night_dir: Path, ids: dict, inspect: bool = False) -> pd.DataFrame | None:
    arch_dir = night_dir / "sleepArchitecture"
    events_path = find_events_file(arch_dir, ids["stem"])
    hyp_path = find_hypnogram_file(arch_dir, ids["stem"])

    if inspect:
        print(f"\n--- {ids['stem']} ---")
        print(f"  night_dir     : {night_dir}")
        print(f"  events_file   : {events_path}")
        print(f"  hypnogram_file: {hyp_path}")
        if arch_dir.exists():
            print(f"  sleepArchitecture inhoud: {[f.name for f in arch_dir.iterdir()]}")
        edf_dir = get_edf_dir(night_dir, ids["stem"])
        print(f"  edf_dir       : {edf_dir}")
        print(f"  edf bestanden : {[f.name for f in edf_dir.glob('*.edf')]}")
        return None

    if events_path is None:
        print(f"  [SKIP] geen events-bestand gevonden voor {ids['stem']}")
        return None

    events = load_events(events_path)
    if len(events) == 0:
        return None

    hypnogram = load_hypnogram(hyp_path)
    signals = load_night_signals(night_dir, ids["stem"])

    if "EEG L" not in signals and "EEG R" not in signals:
        print(f"  [SKIP] geen EEG-kanalen geladen voor {ids['stem']}")
        return None

    rows = []
    prev_end = None
    for i, ev in events.iterrows():
        feats = extract_event_features(signals, ev["start_sec"], ev["end_sec"])
        feats.update({
            "subject_id": ids["subject_id"],
            "group": ids["group"],
            "night_id": ids["night_id"],
            "event_idx": i,
            "start_sec": ev["start_sec"],
            "end_sec": ev["end_sec"],
            "duration_sec": ev["duration_sec"],
            "stage_rk": get_stage_at(hypnogram, ev["start_sec"]),
            "time_since_prev_event_sec": (ev["start_sec"] - prev_end) if prev_end is not None else np.nan,
        })
        rows.append(feats)
        prev_end = ev["end_sec"]

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true", help="Alleen paden/bestanden tonen, niets berekenen")
    parser.add_argument("--limit", type=int, default=None, help="Beperk tot N nachten (voor testen)")
    args = parser.parse_args()

    night_dirs = find_night_dirs(RAW_ROOT)
    print(f"Gevonden: {len(night_dirs)} nachten met een sleepArchitecture map")

    if args.limit:
        night_dirs = night_dirs[: args.limit]

    all_rows = []
    for night_dir in night_dirs:
        ids = parse_ids(night_dir)
        try:
            df = process_night(night_dir, ids, inspect=args.inspect)
        except Exception as e:
            print(f"  [ERROR] {ids['stem']}: {e}")
            continue
        if df is not None:
            all_rows.append(df)
            print(f"  [OK] {ids['stem']}: {len(df)} events verwerkt")

            # Per-nacht featurematrix apart opslaan
            EVENTS_DIR.mkdir(parents=True, exist_ok=True)
            night_out_path = EVENTS_DIR / f"{ids['stem']}_fm.csv"
            df.to_csv(night_out_path, index=False)

    if args.inspect:
        print("\nInspectie klaar. Pas find_events_file / load_events / load_hypnogram")
        print("aan als de gevonden bestandsnamen of kolommen niet kloppen.")
        return

    if not all_rows:
        print("Geen events verwerkt.")
        return

    feature_matrix = pd.concat(all_rows, ignore_index=True)

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVENTS_DIR / "arousal_feature_matrix.csv"
    feature_matrix.to_csv(out_path, index=False)
    print(f"\nFeaturematrix opgeslagen: {out_path}")
    print(f"Shape: {feature_matrix.shape}")


if __name__ == "__main__":
    main()