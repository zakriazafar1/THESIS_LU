"""
=============================================================================
PHASE 9 — Candidate event pipeline direct vanuit losse ZMax EDF bestanden
           (geen tussenliggende .fif of samengevoegde EDF nodig)

Per nacht worden de losse kanaalbestanden geladen:
  EEG L.edf, EEG R.edf, dX.edf, dY.edf, dZ.edf  (OXY_IR_AC optioneel)

Mappenstructuur verwacht:
  RAW_ROOT / GROUP / subject_id / subject_id_T0_N1 / subject_id_T0_N1_edf /
      EEG L.edf
      EEG R.edf
      dX.edf
      dY.edf
      dZ.edf

Hypnogram (verplicht):
  Per nacht moet een human-rated hypnogram CSV aanwezig zijn:
    .../subject_id_T0_N1/sleepArchitecture/subject_id_T0_N1.csv
  Codering: Rechtschaffen & Kales
    0 = Wake, 1 = N1, 2 = N2, 3 = N3/SWS, 5 = REM
  Nachten zonder hypnogram worden overgeslagen.

  Het hypnogram wordt gebruikt voor:
    - Gate A: wake epochs (score=0) uitsluiten als onset-locatie
    - Gate B: post-event epoch checken op wake (score=0)
    - Elke geaccepteerd event krijgt de bijbehorende slaapfase als label

Pipeline per nacht:
  1. Hypnogram laden — geen hypnogram = skip
  2. Losse EDF bestanden laden en samenvoegen in geheugen
  3. Preprocessing: DC removal, notch 50 Hz, bandpass EEG + EMG, resample 128 Hz
  4. Band-envelopes (theta / alpha / sigma / beta)
  5. Lokale baseline + ratio (90 s causale rollende mediaan)
  6. Candidate activation mask
  7. Micro-arousal gates (A: wake epoch, B: post-event wake epoch, C: REM EMG)
  8. Event boundaries + DataFrame → CSV

Output:
  EVENTS_DIR / GROUP / subject_id / candidate_events_{subject_id}_{night_id}.csv

Gebruik:
  python phase9_from_raw.py             # volledige batch
  python phase9_from_raw.py --limit 3   # testen op 3 nachten
  python phase9_from_raw.py --jobs 4    # 4 parallelle workers
=============================================================================
"""

import io
import contextlib
import traceback
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.signal import hilbert
from joblib import Parallel, delayed
from tqdm.auto import tqdm


# =============================================================================
# SECTIE 1 — CONFIGURATIE
# =============================================================================

# ── Mappen ────────────────────────────────────────────────────────────────────
RAW_ROOT   = Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg")
GROUPS     = ["NSR", "Prezens", "SAV"]   # de drie te doorzoeken groepsmappen
EVENTS_DIR = Path(r"C:\Users\zafar\Documents\THESIS_OUTPUTS\2_candidate_events")
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Kanaalnamen intern (wat Phase 9 verwacht) → EDF bestandsnaam ──────────────
# EEG L en R worden twee keer ingeladen:
#   psg-lp  → krijgt EEG bandpass (0.1–35 Hz)
#   psg-emg → krijgt EMG bandpass (10–100 Hz)
CHANNEL_FILES = {
    "EEG L psg-lp":  "EEG L",
    "EEG R psg-lp":  "EEG R",
    "EEG L psg-emg": "EEG L",
    "EEG R psg-emg": "EEG R",
    "dX":            "dX",
    "dY":            "dY",
    "dZ":            "dZ",
}

CH_F7    = "EEG L psg-lp"
CH_F8    = "EEG R psg-lp"
CH_EMG_L = "EEG L psg-emg"
CH_EMG_R = "EEG R psg-emg"
EEG_CH   = [CH_F7, CH_F8]
EMG_CH   = [CH_EMG_L, CH_EMG_R]
MOV_CH   = ["dX", "dY", "dZ"]
ALL_CH   = EEG_CH + EMG_CH + MOV_CH

# ── Preprocessing parameters ──────────────────────────────────────────────────
TARGET_SFREQ          = 128
NOTCH_HZ              = 50.0
EEG_L_FREQ            = 0.1
EEG_H_FREQ            = 35.0
EMG_L_FREQ            = 10.0
EMG_H_FREQ            = 100.0
MOVEMENT_THRESHOLD_UV = 1000.0

# ── Frequentiebanden ──────────────────────────────────────────────────────────
BANDS = {
    "theta": (4.0,  7.0),
    "alpha": (8.0,  12.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}

# ── Signaalverwerking ─────────────────────────────────────────────────────────
SMOOTH_SEC           = 0.5
BASELINE_SEC         = 90.0    # Popovic gebruikte 90 s
ACTIVATION_THRESHOLD = 2.0
SPINDLE_THRESHOLD    = 2.0
MERGE_GAP_SEC        = 1.0
MIN_DUR_SEC          = 1.0
MAX_DUR_SEC          = 15.0

# ── Wake mask ─────────────────────────────────────────────────────────────────
WAKE_THRESHOLD_MULT  = 2.5
WAKE_MIN_DUR_SEC     = 20.0

# ── Post-event wake check ─────────────────────────────────────────────────────
POST_EVENT_CHECK_SEC = 15.0
POST_EVENT_WAKE_FRAC = 0.5

# ── REM gate ──────────────────────────────────────────────────────────────────
REM_EMG_SUPPRESSION_FRAC = 0.8

N_JOBS = -1

# ── Hypnogram ─────────────────────────────────────────────────────────────────
# Rechtschaffen & Kales codering
STAGE_WAKE = 0
STAGE_REM  = 5   # R&K gebruikt 5 voor REM, niet 4 (AASM)
EPOCH_SEC  = 30  # standaard PSG epoch lengte in seconden

# Bestandsstructuur hypnogram:
#   RAW_ROOT / GROUP / subject_id / stem / sleepArchitecture / stem.csv
# Voorbeeld:
#   .../NSR/bnbd_nsr_03554/bnbd_nsr_03554_T0_N1/sleepArchitecture/
#       bnbd_nsr_03554_T0_N1.csv


# =============================================================================
# SECTIE 2 — MAPPENSTRUCTUUR
# =============================================================================

def find_night_dirs(raw_root: Path) -> list:
    """
    Zoekt alle *_edf mappen onder de drie groepsmappen (NSR, Prezens, SAV).

    Structuur:
      raw_root / GROUP / bnbd_xxx_XXXXX / bnbd_xxx_XXXXX_T0_N1 /
          bnbd_xxx_XXXXX_T0_N1_edf /    ← dit zoeken we
              EEG L.edf
              EEG R.edf
              ...

    Geeft gesorteerde lijst van *_edf Path objecten.
    """
    edf_dirs = []
    for group in GROUPS:
        group_path = raw_root / group
        if not group_path.exists():
            continue
        edf_dirs.extend(
            d for d in group_path.rglob("*_edf") if d.is_dir()
        )
    return sorted(edf_dirs)


def parse_ids(edf_dir: Path) -> dict:
    """
    Parseert IDs uit de naam van de *_edf map.

    .../NSR/bnbd_nsr_01272/bnbd_nsr_01272_T0_N2/bnbd_nsr_01272_T0_N2_edf
    → subject_id = bnbd_nsr_01272
    → night_id   = T0_N2
    → group      = NSR
    → stem       = bnbd_nsr_01272_T0_N2
    """
    stem  = edf_dir.name.replace("_edf", "")   # "bnbd_nsr_01272_T0_N2"
    parts = stem.split("_")                     # ["bnbd","nsr","01272","T0","N2"]
    return {
        "subject_id": "_".join(parts[:3]),      # bnbd_nsr_01272
        "night_id":   "_".join(parts[3:]),      # T0_N2
        "group":      parts[1].upper(),         # NSR
        "stem":       stem,
    }


def get_csv_path(ids: dict) -> Path:
    return (
        EVENTS_DIR
        / ids["group"]
        / ids["subject_id"]
        / f"candidate_events_{ids['subject_id']}_{ids['night_id']}.csv"
    )


def get_hypnogram_path(ids: dict) -> Path:
    """
    Berekent het hypnogram pad.

    Structuur:
      RAW_ROOT / GROUP / subject_id / stem / sleepArchitecture / stem.csv

    Voorbeeld:
      .../NSR/bnbd_nsr_01272/bnbd_nsr_01272_T0_N2/
          sleepArchitecture/bnbd_nsr_01272_T0_N2.csv
    """
    return (
        RAW_ROOT
        / ids["group"]
        / ids["subject_id"]
        / ids["stem"]
        / "sleepArchitecture"
        / f"{ids['stem']}.csv"
    )


# =============================================================================
# SECTIE 3 — LADEN EN PREPROCESSING
# Losse EDF bestanden laden, samenvoegen in geheugen, preprocessing toepassen.
# Geeft signals dict terug: kanaalnaam → 1D numpy array (µV), plus sfreq.
# =============================================================================

def load_and_preprocess(night_dir: Path) -> dict:
    """
    Laadt losse EDF kanaalbestanden direct uit de nacht-map en past
    preprocessing toe.

    Stappen:
      1. Elk kanaalbestand afzonderlijk laden
      2. Samenvoegen tot één MNE RawArray in geheugen
      3. DC removal, notch 50 Hz, bandpass EEG + EMG, resample 128 Hz
      4. Numpy arrays teruggeven als signals dict

    Returns
    -------
    dict: kanaalnaam → 1D numpy array (µV), plus "sfreq" key
    """
    loaded    = {}
    sfreq_ref = None

    # ── Elk uniek EDF bestand één keer laden ─────────────────────────────────
    unique_files = set(CHANNEL_FILES.values())
    for raw_file in unique_files:
        edf_path = night_dir / f"{raw_file}.edf"
        if not edf_path.exists():
            continue
        raw   = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        data  = raw.get_data()[0] * 1e6   # V → µV
        sfreq = raw.info["sfreq"]
        loaded[raw_file] = (data, sfreq)
        if sfreq_ref is None:
            sfreq_ref = sfreq

    if not loaded:
        raise ValueError(f"Geen kanaalbestanden gevonden in {night_dir}")

    # ── Kanaalnamen toewijzen + lengtes gelijkschakelen ───────────────────────
    arrays = {}
    for ch_name, raw_file in CHANNEL_FILES.items():
        if raw_file in loaded:
            arrays[ch_name] = loaded[raw_file][0].copy()

    min_len = min(len(d) for d in arrays.values())
    for k in arrays:
        arrays[k] = arrays[k][:min_len]

    # ── MNE RawArray bouwen ───────────────────────────────────────────────────
    ch_names = list(arrays.keys())
    ch_types = []
    for name in ch_names:
        if "psg-lp"  in name: ch_types.append("eeg")
        elif "psg-emg" in name: ch_types.append("emg")
        else: ch_types.append("misc")

    data_mx = np.stack(list(arrays.values())) * 1e-6   # µV → V voor MNE
    info    = mne.create_info(ch_names=ch_names, sfreq=sfreq_ref,
                               ch_types=ch_types, verbose=False)
    raw     = mne.io.RawArray(data_mx, info, verbose=False)
    raw._data = raw._data.astype(np.float64)

    eeg_lp_chs = [ch for ch in raw.ch_names if "psg-lp"  in ch]
    emg_chs    = [ch for ch in raw.ch_names if "psg-emg" in ch]
    mov_chs    = [ch for ch in raw.ch_names if ch in MOV_CH]

    # ── Preprocessing ─────────────────────────────────────────────────────────

    # 1. DC removal EEG + EMG
    for ch in eeg_lp_chs + emg_chs:
        idx = raw.ch_names.index(ch)
        raw._data[idx] -= np.mean(raw._data[idx])

    # 2. Notch 50 Hz op EEG
    if sfreq_ref > NOTCH_HZ * 2 and eeg_lp_chs:
        raw.notch_filter(freqs=NOTCH_HZ, picks=eeg_lp_chs, verbose=False)

    # 3. Bandpass EEG: 0.1–35 Hz
    if eeg_lp_chs:
        raw.filter(l_freq=EEG_L_FREQ, h_freq=EEG_H_FREQ,
                   picks=eeg_lp_chs, method="fir",
                   fir_window="hamming", verbose=False)

    # 4. Bandpass EMG: 10–100 Hz
    if emg_chs:
        h_emg = min(EMG_H_FREQ, sfreq_ref / 2 - 1)
        raw.filter(l_freq=EMG_L_FREQ, h_freq=h_emg,
                   picks=emg_chs, verbose=False)

    # 5. DC removal accelerometer
    if mov_chs:
        raw.apply_function(lambda x: x - np.mean(x),
                           picks=mov_chs, verbose=False)

    # 6. Resample naar 128 Hz
    if sfreq_ref != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    # ── Teruggeven als signals dict (µV) ──────────────────────────────────────
    signals = {ch: raw.get_data(picks=ch)[0] * 1e6
               for ch in raw.ch_names}
    signals["sfreq"] = TARGET_SFREQ

    return signals


# =============================================================================
# SECTIE 4 — BEWEGINGSARTEFACTEN
# =============================================================================

def remove_movement_artifacts(signals: dict,
                               sfreq: int = TARGET_SFREQ) -> tuple:
    n_samples     = len(signals[CH_F7])
    combined_mask = np.zeros(n_samples, dtype=bool)
    buffer        = int(0.5 * sfreq)
    stats         = {}

    for ch in EEG_CH:
        if ch not in signals:
            continue
        mask         = np.abs(signals[ch]) > MOVEMENT_THRESHOLD_UV
        mask_dilated = np.zeros(n_samples, dtype=bool)
        for idx in np.where(mask)[0]:
            mask_dilated[
                max(0, idx - buffer):min(n_samples, idx + buffer)
            ] = True
        stats[ch]     = round(mask_dilated.mean() * 100, 2)
        combined_mask |= mask_dilated

    return combined_mask, stats


# =============================================================================
# SECTIE 5 — BAND-ENVELOPES
# =============================================================================

def bandpass_filter(signal: np.ndarray, lo: float, hi: float,
                    sfreq: float = TARGET_SFREQ) -> np.ndarray:
    filtered = mne.filter.filter_data(
        signal[np.newaxis, :], sfreq=sfreq, l_freq=lo, h_freq=hi,
        method="fir", fir_window="hamming", verbose=False,
    )
    return filtered[0]


def compute_envelope(signal: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(signal)).astype(np.float32)


def smooth_envelope(envelope: np.ndarray, smooth_sec: float = SMOOTH_SEC,
                    sfreq: float = TARGET_SFREQ) -> np.ndarray:
    window = int(smooth_sec * sfreq)
    return np.convolve(envelope, np.ones(window) / window,
                       mode="same").astype(np.float32)


def compute_band_envelopes(signals: dict,
                            sfreq: float = TARGET_SFREQ) -> dict:
    time_axis = np.arange(len(signals[CH_F7])) / sfreq
    result    = {}
    for ch_label, ch_name in [("F7", CH_F7), ("F8", CH_F8)]:
        if ch_name not in signals:
            continue
        sig  = signals[ch_name]
        rows = {"time": time_axis}
        for band_name, (lo, hi) in BANDS.items():
            env             = compute_envelope(bandpass_filter(sig, lo, hi, sfreq))
            rows[band_name] = smooth_envelope(env, SMOOTH_SEC, sfreq)
        result[ch_label] = pd.DataFrame(rows)
    return result


def compute_emg_envelopes(signals: dict,
                           sfreq: float = TARGET_SFREQ) -> dict:
    result = {}
    for ch_label, ch_name in [("F7", CH_EMG_L), ("F8", CH_EMG_R)]:
        if ch_name not in signals:
            continue
        env = compute_envelope(signals[ch_name])
        result[ch_label] = smooth_envelope(env, SMOOTH_SEC, sfreq)
    return result


# =============================================================================
# SECTIE 6 — BASELINE + RATIO
# =============================================================================

def _rolling_median_causal(arr: np.ndarray, sfreq: float,
                            window_s: float) -> np.ndarray:
    """
    Causale rollende mediaan via pandas rolling.
    Kijkt alleen terug in de tijd (geen toekomst).
    Robuust tegen uitschieters en compatibel met alle NumPy versies.
    """
    w      = max(1, int(window_s * sfreq))
    result = (pd.Series(arr.astype(np.float64))
              .rolling(window=w, min_periods=1)
              .median())
    return result.values.astype(np.float32)


def compute_rolling_baseline(envelopes: dict,
                              movement_mask: np.ndarray = None,
                              sfreq: float = TARGET_SFREQ) -> dict:
    baselines = {}
    for ch_label, df in envelopes.items():
        rows = {"time": df["time"].values}
        for band in BANDS:
            env = df[band].copy().astype(np.float64)
            if movement_mask is not None:
                env[movement_mask] = np.nan
            series     = pd.Series(env).ffill().bfill().values.astype(np.float32)
            rows[band] = _rolling_median_causal(series, sfreq, BASELINE_SEC)
        baselines[ch_label] = pd.DataFrame(rows)
    return baselines


def compute_ratio(envelopes: dict, baselines: dict) -> dict:
    ratios = {}
    for ch_label in envelopes:
        rows = {"time": envelopes[ch_label]["time"].values}
        for band in BANDS:
            rows[band] = (
                envelopes[ch_label][band].values /
                (baselines[ch_label][band].values + 1e-6)
            ).astype(np.float32)
        ratios[ch_label] = pd.DataFrame(rows)
    return ratios


def compute_emg_baseline(emg_envelopes: dict,
                          movement_mask: np.ndarray = None,
                          sfreq: float = TARGET_SFREQ) -> dict:
    baselines = {}
    for ch_label, env in emg_envelopes.items():
        arr = env.copy().astype(np.float32)
        if movement_mask is not None:
            arr[movement_mask] = np.nan
        arr = pd.Series(arr).ffill().bfill().values.astype(np.float32)
        baselines[ch_label] = _rolling_median_causal(arr, sfreq, BASELINE_SEC)
    return baselines


# =============================================================================
# SECTIE 7 — HYPNOGRAM LADEN + WAKE MASK
#
# Het hypnogram (human-rated, R&K codering) wordt gebruikt als wake mask:
#   - Elk epoch met score 0 (Wake) → alle samples in dat epoch = wake
#   - De wake mask is een boolean array op sample-niveau
#
# Voordeel t.o.v. signaal-afgeleide wake mask:
#   - Menselijke scorer identificeert wake betrouwbaarder dan het algoritme
#   - Geen risico op vals-positieve wake-classificatie bij hoge-beta slapers
#     (veel voorkomend bij angst/PTSD in de BNBD populatie)
#
# Geen hypnogram aanwezig → nacht wordt overgeslagen (optie A).
# =============================================================================

def load_hypnogram(hyp_path: Path) -> np.ndarray:
    """
    Laadt het hypnogram CSV bestand.

    Verwacht één kolom zonder header, één score per rij (30-s epoch).
    Rechtschaffen & Kales codering: 0=Wake, 1=N1, 2=N2, 3=N3, 5=REM.

    Returns
    -------
    np.ndarray van integers, lengte = aantal epochs
    """
    df = pd.read_csv(hyp_path, header=None)
    return df.iloc[:, 0].values.astype(int)


def hypnogram_to_wake_mask(hypnogram: np.ndarray,
                            n_samples: int,
                            sfreq: float = TARGET_SFREQ,
                            epoch_sec: int = EPOCH_SEC) -> np.ndarray:
    """
    Zet een epoch-niveau hypnogram om naar een sample-niveau wake mask.

    Elk epoch van epoch_sec seconden wordt uitgebreid naar
    epoch_sec × sfreq samples. Wake epoch (score=0) → True.

    Parameters
    ----------
    hypnogram : array van R&K scores per epoch
    n_samples : totaal aantal samples in het EEG signaal
                (voor afstemming als hypnogram iets korter/langer is)
    sfreq     : sample frequentie
    epoch_sec : epoch duur in seconden (standaard 30)

    Returns
    -------
    boolean array, lengte = n_samples, True = wake sample
    """
    samples_per_epoch = int(epoch_sec * sfreq)
    wake_mask         = np.zeros(n_samples, dtype=bool)

    for epoch_idx, score in enumerate(hypnogram):
        start = epoch_idx * samples_per_epoch
        end   = min(start + samples_per_epoch, n_samples)
        if start >= n_samples:
            break
        if score == STAGE_WAKE:
            wake_mask[start:end] = True

    return wake_mask


def hypnogram_to_stage_array(hypnogram: np.ndarray,
                              n_samples: int,
                              sfreq: float = TARGET_SFREQ,
                              epoch_sec: int = EPOCH_SEC) -> np.ndarray:
    """
    Zet hypnogram om naar sample-niveau stage array (R&K integer per sample).
    Wordt gebruikt om elk gedetecteerd event een slaapfase label te geven.
    """
    samples_per_epoch = int(epoch_sec * sfreq)
    stage_array       = np.full(n_samples, -1, dtype=int)  # -1 = onbekend

    for epoch_idx, score in enumerate(hypnogram):
        start = epoch_idx * samples_per_epoch
        end   = min(start + samples_per_epoch, n_samples)
        if start >= n_samples:
            break
        stage_array[start:end] = score

    return stage_array


# R&K stage labels voor in de output DataFrame
RK_LABELS = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 5: "REM", -1: "Unscored"}


# =============================================================================
# SECTIE 8 — CANDIDATE DETECTIE + SPINDLE SCORE
# =============================================================================

def compute_activation_mask(ratios: dict) -> dict:
    masks = {}
    for ch_label, df in ratios.items():
        masks[ch_label] = (
            (df["alpha"].values > ACTIVATION_THRESHOLD) |
            (df["beta"].values  > ACTIVATION_THRESHOLD)
        )
    return masks


def combine_channels(masks: dict) -> dict:
    F7, F8 = masks["F7"], masks["F8"]
    return {
        "F7_only":   F7 & ~F8,
        "F8_only":   F8 & ~F7,
        "bilateral": F7 & F8,
        "combined":  F7 | F8,
    }


def compute_spindle_score(ratios: dict, envelopes: dict) -> np.ndarray:
    scores = []
    for ch in ("F7", "F8"):
        env = envelopes[ch]
        rat = ratios[ch]
        c1  = (env["sigma"].values > env["theta"].values).astype(np.float32)
        c2  = (env["sigma"].values > env["alpha"].values).astype(np.float32)
        c3  = (env["sigma"].values > env["beta"].values ).astype(np.float32)
        c4  = (rat["sigma"].values > SPINDLE_THRESHOLD  ).astype(np.float32)
        scores.append((c1 + c2 + c3 + c4) / 4.0)
    return np.mean(scores, axis=0).astype(np.float32)


# =============================================================================
# SECTIE 9 — MICRO-AROUSAL GATES
# =============================================================================

def _is_rem_like_window(onset, offset, emg_envelopes, emg_baselines):
    fracs = []
    for ch in ("F7", "F8"):
        if ch not in emg_envelopes or ch not in emg_baselines:
            continue
        suppressed = (emg_envelopes[ch][onset:offset] <
                      REM_EMG_SUPPRESSION_FRAC * emg_baselines[ch][onset:offset])
        fracs.append(suppressed.mean())
    return np.mean(fracs) > 0.5 if fracs else False


def _has_emg_elevation(onset, offset, emg_envelopes, emg_baselines):
    fracs = []
    for ch in ("F7", "F8"):
        if ch not in emg_envelopes or ch not in emg_baselines:
            continue
        fracs.append(
            (emg_envelopes[ch][onset:offset] >
             emg_baselines[ch][onset:offset]).mean()
        )
    return np.mean(fracs) >= 0.5 if fracs else True


def mask_to_events(mask):
    events, in_event, start = [], False, 0
    for i, active in enumerate(mask):
        if active and not in_event:
            start, in_event = i, True
        elif not active and in_event:
            in_event = False
            events.append((start, i))
    if in_event:
        events.append((start, len(mask)))
    return events


def merge_events(events, sfreq=TARGET_SFREQ):
    if not events:
        return []
    gap    = int(MERGE_GAP_SEC * sfreq)
    merged = [events[0]]
    for s, e in events[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def apply_duration_filter(events, sfreq=TARGET_SFREQ):
    lo = int(MIN_DUR_SEC * sfreq)
    hi = int(MAX_DUR_SEC * sfreq)
    return [(s, e) for s, e in events if lo <= (e - s) <= hi]


def apply_microarousal_gates(events, wake_mask,
                              emg_envelopes, emg_baselines,
                              sfreq=TARGET_SFREQ):
    post_s    = int(POST_EVENT_CHECK_SEC * sfreq)
    n         = len(wake_mask)
    accepted  = []
    counts    = {"gate_a_wake_onset": 0,
                 "gate_b_post_wake":  0,
                 "gate_c_rem_emg":    0}

    for start, end in events:

        # Gate A — onset in wake-like periode
        if wake_mask[start]:
            counts["gate_a_wake_onset"] += 1
            continue

        # Gate B — post-event venster wake-like
        post_window = wake_mask[end:min(end + post_s, n)]
        if len(post_window) > 0 and post_window.mean() >= POST_EVENT_WAKE_FRAC:
            counts["gate_b_post_wake"] += 1
            continue

        # Gate C — REM venster zonder EMG activatie
        if (_is_rem_like_window(start, end, emg_envelopes, emg_baselines) and
                not _has_emg_elevation(start, end, emg_envelopes, emg_baselines)):
            counts["gate_c_rem_emg"] += 1
            continue

        accepted.append((start, end))

    return accepted, counts


# =============================================================================
# SECTIE 10 — EVENTS NAAR DATAFRAME
# =============================================================================

def events_to_dataframe(events, channel_masks, spindle_score,
                         emg_envelopes, emg_baselines,
                         wake_mask, stage_array,
                         movement_mask=None,
                         sfreq=TARGET_SFREQ):
    rows     = []
    post_s   = int(POST_EVENT_CHECK_SEC * sfreq)
    n_signal = len(wake_mask)

    for start, end in events:
        dur_sec  = (end - start) / sfreq
        post_end = min(end + post_s, n_signal)

        # ── Per-sample activatie in dit event venster ──────────────────────
        # F7_only: samples waar alleen F7 actief was (niet F8)
        # F8_only: samples waar alleen F8 actief was (niet F7)
        # bilateral: samples waar F7 EN F8 tegelijk actief waren
        f7_seg    = channel_masks["F7_only"][start:end]
        f8_seg    = channel_masks["F8_only"][start:end]
        bilat_seg = channel_masks["bilateral"][start:end]

        frac_f7    = float(f7_seg.mean())
        frac_f8    = float(f8_seg.mean())
        frac_bilat = float(bilat_seg.mean())

        # Dominant kanaal = het patroon met de grootste fractie van het event
        if frac_bilat >= frac_f7 and frac_bilat >= frac_f8:
            dominant = "bilateral"
        elif frac_f7 >= frac_f8:
            dominant = "F7"
        else:
            dominant = "F8"

        rows.append({
            "start_sample":         start,
            "end_sample":           end,
            "start_sec":            round(start / sfreq, 4),
            "end_sec":              round(end   / sfreq, 4),
            "duration_sec":         round(dur_sec, 4),
            "duration_category":    ("short" if dur_sec <= 3.0 else "arousal"),
            # Was dit kanaal ergens actief in het event? (ook 1 sample telt)
            "F7_active":            bool(f7_seg.any() or bilat_seg.any()),
            "F8_active":            bool(f8_seg.any() or bilat_seg.any()),
            # Was de MEERDERHEID (>50%) van het event op dit patroon?
            # F7_only en bilateral sluiten elkaar uit — max één kan True zijn
            "F7_only":              bool(frac_f7    > 0.5),
            "F8_only":              bool(frac_f8    > 0.5),
            "bilateral":            bool(frac_bilat > 0.5),
            # Dominant kanaal: welk patroon had de grootste fractie?
            "dominant_channel":     dominant,
            # Ruwe fracties voor verdere analyse
            "frac_F7_only":         round(frac_f7,    4),
            "frac_F8_only":         round(frac_f8,    4),
            "bilateral_overlap":    round(frac_bilat, 4),
            "spindle_score_mean":   round(float(spindle_score[start:end].mean()), 4),
            "spindle_score_max":    round(float(spindle_score[start:end].max()),  4),
            "artifact_overlap_pct": round(float(movement_mask[start:end].mean() * 100), 2)
                                    if movement_mask is not None else 0.0,
            "rem_like":             _is_rem_like_window(start, end,
                                                         emg_envelopes,
                                                         emg_baselines),
            "post_wake_frac":       round(float(wake_mask[end:post_end].mean()), 4),
            "wake_onset":           bool(wake_mask[max(0, start - int(5*sfreq)):start].mean() > 0.5),
            "stage_rk":             int(pd.Series(stage_array[start:end]).mode()[0]),
            "stage_label":          RK_LABELS.get(
                                        int(pd.Series(stage_array[start:end]).mode()[0]),
                                        "Unscored"),
        })

    return pd.DataFrame(rows)


# =============================================================================
# SECTIE 11 — VERWERKING VAN ÉÉN NACHT
# =============================================================================

def _process_night(edf_dir: Path) -> dict:
    ids      = parse_ids(edf_dir)
    out_csv  = get_csv_path(ids)
    hyp_path = get_hypnogram_path(ids)

    log = {
        "group":              ids["group"],
        "subject_id":         ids["subject_id"],
        "night_id":           ids["night_id"],
        "edf_dir":            str(edf_dir),
        "output_csv":         str(out_csv),
        "hypnogram_path":     str(hyp_path),
        "status":             "failed",
        "n_events":           "",
        "gate_a_wake_onset":  "",
        "gate_b_post_wake":   "",
        "gate_c_rem_emg":     "",
        "movement_pct_L":     "",
        "movement_pct_R":     "",
        "error":              "",
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Skip als output al bestaat ────────────────────────────────────────────
    if out_csv.exists():
        log["status"] = "skipped"
        return log

    # ── Skip als hypnogram ontbreekt (optie A) ────────────────────────────────
    if not hyp_path.exists():
        log["status"] = "no_hypnogram"
        log["error"]  = f"Hypnogram niet gevonden: {hyp_path}"
        return log

    try:
        with contextlib.redirect_stdout(io.StringIO()):

            # ── Hypnogram laden ───────────────────────────────────────────────
            hypnogram = load_hypnogram(hyp_path)

            # ── Laden + preprocessing ─────────────────────────────────────────
            # edf_dir is de *_edf map met de losse kanaalbestanden
            signals   = load_and_preprocess(edf_dir)
            sfreq   = signals["sfreq"]
            n_samples = len(signals[CH_F7])

            # ── Wake mask en stage array vanuit hypnogram ─────────────────────
            wake_mask   = hypnogram_to_wake_mask(hypnogram, n_samples, sfreq)
            stage_array = hypnogram_to_stage_array(hypnogram, n_samples, sfreq)

            # ── Bewegingsmasker ───────────────────────────────────────────────
            movement_mask, mov_stats = remove_movement_artifacts(signals, sfreq)

            # ── Band-envelopes ────────────────────────────────────────────────
            envelopes     = compute_band_envelopes(signals, sfreq)
            emg_envelopes = compute_emg_envelopes(signals, sfreq)

            # ── Baseline + ratio ──────────────────────────────────────────────
            baselines     = compute_rolling_baseline(envelopes, movement_mask, sfreq)
            ratios        = compute_ratio(envelopes, baselines)
            emg_baselines = compute_emg_baseline(emg_envelopes, movement_mask, sfreq)

            # ── Candidate detectie ────────────────────────────────────────────
            activation_masks = compute_activation_mask(ratios)
            channel_masks    = combine_channels(activation_masks)
            spindle_score    = compute_spindle_score(ratios, envelopes)

            candidate_mask = channel_masks["combined"].copy()
            candidate_mask = candidate_mask & ~movement_mask

            # ── Event grenzen ─────────────────────────────────────────────────
            events = mask_to_events(candidate_mask)
            events = merge_events(events, sfreq)
            events = apply_duration_filter(events, sfreq)

            # ── Micro-arousal gates ───────────────────────────────────────────
            events, gate_counts = apply_microarousal_gates(
                events, wake_mask, emg_envelopes, emg_baselines, sfreq
            )

            # ── DataFrame bouwen ──────────────────────────────────────────────
            df = events_to_dataframe(
                events, channel_masks, spindle_score,
                emg_envelopes, emg_baselines,
                wake_mask, stage_array,
                movement_mask, sfreq
            )

        # ── Identifiers toevoegen en opslaan ──────────────────────────────────
        df.insert(0, "night_id",   ids["night_id"])
        df.insert(0, "subject_id", ids["subject_id"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        log.update({
            "status":            "ok",
            "n_events":          len(df),
            "gate_a_wake_onset": gate_counts["gate_a_wake_onset"],
            "gate_b_post_wake":  gate_counts["gate_b_post_wake"],
            "gate_c_rem_emg":    gate_counts["gate_c_rem_emg"],
            "movement_pct_L":    mov_stats.get(CH_F7, ""),
            "movement_pct_R":    mov_stats.get(CH_F8, ""),
        })

    except Exception:
        log["error"] = traceback.format_exc()[-800:]

    return log


# =============================================================================
# SECTIE 12 — BATCH RUNNER
# =============================================================================

def run_batch(limit: int = None, n_jobs: int = N_JOBS) -> pd.DataFrame:
    t_start = datetime.now()
    print("=" * 65)
    print("  PHASE 9 — Direct vanuit losse ZMax EDF bestanden")
    print(f"  Start  : {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Bron   : {RAW_ROOT}")
    print(f"  Output : {EVENTS_DIR}")
    print("=" * 65)

    all_dirs = find_night_dirs(RAW_ROOT)
    if not all_dirs:
        print(f"\n  [FOUT] Geen *_edf mappen gevonden onder {RAW_ROOT}")
        return pd.DataFrame()

    if limit:
        all_dirs = all_dirs[:limit]

    todo   = [d for d in all_dirs if not get_csv_path(parse_ids(d)).exists()]
    n_skip = len(all_dirs) - len(todo)

    print(f"\n  Nachten gevonden : {len(all_dirs)}")
    print(f"  Al verwerkt      : {n_skip}")
    print(f"  Te verwerken     : {len(todo)}")
    print(f"  Workers          : {'alle cores' if n_jobs == -1 else n_jobs}\n")

    results = list(tqdm(
        Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(_process_night)(d) for d in todo
        ),
        total=len(todo), desc="Nachten", unit="nacht",
    ))

    log_df   = pd.DataFrame(results).sort_values(
        ["group", "subject_id", "night_id"]
    )
    log_path = EVENTS_DIR / "batch_log.csv"
    log_df.to_csv(log_path, index=False)

    ok_df    = log_df[log_df["status"] == "ok"]
    n_ok     = len(ok_df)
    n_skip   = (log_df["status"] == "skipped").sum()
    n_nohyp  = (log_df["status"] == "no_hypnogram").sum()
    n_fail   = (log_df["status"] == "failed").sum()
    mins     = (datetime.now() - t_start).total_seconds() / 60

    print(f"\n{'=' * 65}")
    print(f"  Succesvol        : {n_ok}")
    print(f"  Overgeslagen     : {n_skip}")
    print(f"  Geen hypnogram   : {n_nohyp}  (overgeslagen)")
    print(f"  Mislukt          : {n_fail}")
    print(f"  Tijd             : {mins:.1f} min")

    if n_ok > 0:
        to_num = lambda c: pd.to_numeric(ok_df[c], errors="coerce").sum()
        print(f"  Totaal events        : {int(to_num('n_events'))}")
        print(f"  Gate A (wake onset)  : {int(to_num('gate_a_wake_onset'))}")
        print(f"  Gate B (post-event)  : {int(to_num('gate_b_post_wake'))}")
        print(f"  Gate C (REM EMG)     : {int(to_num('gate_c_rem_emg'))}")

    print(f"  Log : {log_path}")
    print(f"{'=' * 65}")

    if n_fail > 0:
        for _, row in log_df[log_df["status"] == "failed"].iterrows():
            print(f"\n  FAIL: {row['subject_id']} / {row['night_id']}")
            print(f"  {row['error'][:300]}")

    return log_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 9 direct vanuit losse ZMax EDF bestanden"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Verwerk alleen de eerste N nachten (voor testen)")
    parser.add_argument("--jobs",  type=int, default=-1,
                        help="Aantal parallelle workers (-1 = alle cores)")
    args = parser.parse_args()
    run_batch(limit=args.limit, n_jobs=args.jobs)
    