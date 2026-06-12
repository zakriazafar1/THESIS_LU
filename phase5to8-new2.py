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

Output:
  EVENTS_DIR / GROUP / subject_id / candidate_events_{subject_id}_{night_id}.csv

Gebruik:
  python phase5to8-new.py             # volledige batch
  python phase5to8-new.py --limit 3   # testen op 3 nachten
  python phase5to8-new.py --jobs 4    # 4 parallelle workers
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
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.fft import next_fast_len
from joblib import Parallel, delayed
from tqdm.auto import tqdm

try:
    import bottleneck as bn
    _HAS_BOTTLENECK = True
except ImportError:
    _HAS_BOTTLENECK = False


# =============================================================================
# SECTIE 1 — CONFIGURATIE
# =============================================================================

RAW_ROOT   = Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg")
GROUPS     = ["NSR", "Prezens", "SAV"]
EVENTS_DIR = Path(r"C:\Users\zafar\Documents\THESIS_OUTPUTS\2_candidate_events")
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

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

TARGET_SFREQ          = 128
NOTCH_HZ              = 50.0
EEG_L_FREQ            = 0.1
EEG_H_FREQ            = 35.0
EMG_L_FREQ            = 10.0
EMG_H_FREQ            = 100.0
MOVEMENT_THRESHOLD_UV = 1000.0

BANDS = {
    "theta": (4.0,  7.0),
    "alpha": (8.0,  12.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}

SMOOTH_SEC           = 0.5
BASELINE_SEC         = 90.0
ACTIVATION_THRESHOLD = 2.0
SPINDLE_THRESHOLD    = 2.0
MERGE_GAP_SEC        = 1.0
MIN_DUR_SEC          = 1.0
MAX_DUR_SEC          = 15.0

POST_EVENT_CHECK_SEC     = 15.0
POST_EVENT_WAKE_FRAC     = 0.5
REM_EMG_SUPPRESSION_FRAC = 0.8

N_JOBS     = -1
STAGE_WAKE = 0
STAGE_REM  = 5
EPOCH_SEC  = 30

# ── Arousal score gewichten ───────────────────────────────────────────────────
# Gewogen gemiddelde van 6 signaalkenmerken (0–1 elk).
# Gewichten worden automatisch genormaliseerd.
AROUSAL_SCORE_WEIGHTS = {
    "alpha_ratio":      0.25,
    "beta_ratio":       0.20,
    "bilateral":        0.15,
    "emg_consistency":  0.15,
    "sigma_suppress":   0.10,
    "signal_stability": 0.15,
}


# =============================================================================
# SECTIE 2 — MAPPENSTRUCTUUR
# =============================================================================

def find_night_dirs(raw_root: Path) -> list:
    edf_dirs = []
    for group in GROUPS:
        group_path = raw_root / group
        if not group_path.exists():
            continue
        edf_dirs.extend(
            d for d in group_path.rglob("*_edf")
            if d.is_dir() and "_T0_" in d.name
        )
    return sorted(edf_dirs)


def parse_ids(edf_dir: Path) -> dict:
    stem  = edf_dir.name.replace("_edf", "")
    parts = stem.split("_")
    return {
        "subject_id": "_".join(parts[:3]),
        "night_id":   "_".join(parts[3:]),
        "group":      parts[1].upper(),
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
# Robust: slechte/ontbrekende kanalen worden per kanaal afgevangen zodat
# één corrupt bestand niet de hele nacht laat falen.
# =============================================================================

def load_and_preprocess(edf_dir: Path) -> dict:
    """
    Laadt losse EDF kanaalbestanden en past preprocessing toe.
    Elk kanaalbestand wordt individueel geladen met foutafvang zodat
    een corrupt of ontbrekend bestand de nacht niet volledig laat falen.
    Vereist: EEG L.edf en EEG R.edf moeten aanwezig en leesbaar zijn.
    """
    loaded    = {}   # raw_filename → (data µV, sfreq)
    sfreq_ref = None
    load_errors = []

    unique_files = set(CHANNEL_FILES.values())
    for raw_file in unique_files:
        edf_path = edf_dir / f"{raw_file}.edf"
        if not edf_path.exists():
            load_errors.append(f"{raw_file}.edf: bestand niet gevonden")
            continue
        try:
            raw   = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            # Controleer dat het bestand minstens 1 kanaal heeft
            if raw.n_times == 0:
                load_errors.append(f"{raw_file}.edf: leeg bestand (0 samples)")
                continue
            data  = raw.get_data()[0] * 1e6   # V → µV
            sfreq = raw.info["sfreq"]
            loaded[raw_file] = (data, sfreq)
            if sfreq_ref is None:
                sfreq_ref = sfreq
        except Exception as e:
            load_errors.append(f"{raw_file}.edf: {e}")

    # EEG L en R zijn verplicht — faal als die ontbreken
    for required in ["EEG L", "EEG R"]:
        if required not in loaded:
            missing_info = "\n".join(load_errors)
            raise ValueError(
                f"Verplicht kanaal '{required}' kon niet geladen worden "
                f"uit {edf_dir}.\nLaadfouten:\n{missing_info}"
            )

    # ── Kanaalnamen toewijzen ─────────────────────────────────────────────────
    arrays = {}
    for ch_name, raw_file in CHANNEL_FILES.items():
        if raw_file in loaded:
            arrays[ch_name] = loaded[raw_file][0].copy()

    # ── Lengtes gelijkschakelen (EDF bestanden kunnen iets afwijken) ──────────
    if not arrays:
        raise ValueError(f"Geen bruikbare kanalen in {edf_dir}")

    min_len = min(len(d) for d in arrays.values())

    # Controleer of het signaal lang genoeg is voor de filters
    # Butterworth orde 4 heeft ~3x padlen nodig: min ~1000 samples bij 128 Hz
    min_required = max(1000, int(10 * (sfreq_ref or TARGET_SFREQ)))
    if min_len < min_required:
        raise ValueError(
            f"Signaal te kort voor filtering: {min_len} samples "
            f"(minimum {min_required}). Nacht waarschijnlijk incompleet."
        )

    for k in arrays:
        arrays[k] = arrays[k][:min_len]

    # ── MNE RawArray bouwen ───────────────────────────────────────────────────
    ch_names = list(arrays.keys())
    ch_types = []
    for name in ch_names:
        if   "psg-lp"  in name: ch_types.append("eeg")
        elif "psg-emg" in name: ch_types.append("emg")
        else:                   ch_types.append("misc")

    data_mx = np.stack(list(arrays.values())) * 1e-6   # µV → V
    info    = mne.create_info(ch_names=ch_names, sfreq=sfreq_ref,
                               ch_types=ch_types, verbose=False)
    raw     = mne.io.RawArray(data_mx, info, verbose=False)
    raw._data = raw._data.astype(np.float64)

    eeg_lp_chs = [ch for ch in raw.ch_names if "psg-lp"  in ch]
    emg_chs    = [ch for ch in raw.ch_names if "psg-emg" in ch]
    mov_chs    = [ch for ch in raw.ch_names if ch in MOV_CH]

    # 1. DC removal
    for ch in eeg_lp_chs + emg_chs:
        idx = raw.ch_names.index(ch)
        raw._data[idx] -= np.mean(raw._data[idx])

    # 2. Notch 50 Hz
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
    nyq = sfreq / 2.0
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def compute_envelope(signal: np.ndarray) -> np.ndarray:
    n      = next_fast_len(len(signal))
    result = np.abs(hilbert(signal, N=n))[:len(signal)]
    return result.astype(np.float32)


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
    w = max(1, int(window_s * sfreq))
    if _HAS_BOTTLENECK:
        result = bn.move_median(arr.astype(np.float64), window=w, min_count=1)
    else:
        result = (pd.Series(arr.astype(np.float64))
                  .rolling(window=w, min_periods=1)
                  .median()
                  .values)
    return result.astype(np.float32)


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
# =============================================================================

def load_hypnogram(hyp_path: Path) -> np.ndarray:
    df = pd.read_csv(hyp_path, header=None)
    return df.iloc[:, 0].values.astype(int)


def hypnogram_to_wake_mask(hypnogram: np.ndarray, n_samples: int,
                            sfreq: float = TARGET_SFREQ,
                            epoch_sec: int = EPOCH_SEC) -> np.ndarray:
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


def hypnogram_to_stage_array(hypnogram: np.ndarray, n_samples: int,
                              sfreq: float = TARGET_SFREQ,
                              epoch_sec: int = EPOCH_SEC) -> np.ndarray:
    samples_per_epoch = int(epoch_sec * sfreq)
    stage_array       = np.full(n_samples, -1, dtype=int)
    for epoch_idx, score in enumerate(hypnogram):
        start = epoch_idx * samples_per_epoch
        end   = min(start + samples_per_epoch, n_samples)
        if start >= n_samples:
            break
        stage_array[start:end] = score
    return stage_array


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
    """
    Tags every event with a gate label instead of filtering them out.
    All events are returned so you can inspect and filter yourself.

    gate values:
      'accepted' — passed all gates, this is a microarousal
      'A'        — started during a wake epoch
      'B'        — person woke up in the 15 s after the event
      'C'        — REM-like window without EMG activation
    """
    post_s  = int(POST_EVENT_CHECK_SEC * sfreq)
    n       = len(wake_mask)
    tagged  = []
    counts  = {"gate_a_wake_onset": 0,
               "gate_b_post_wake":  0,
               "gate_c_rem_emg":    0}

    for start, end in events:

        # Gate A — onset in wake epoch
        if wake_mask[start]:
            counts["gate_a_wake_onset"] += 1
            tagged.append((start, end, "A"))
            continue

        # Gate B — post-event window is predominantly wake
        post_window = wake_mask[end:min(end + post_s, n)]
        if len(post_window) > 0 and post_window.mean() >= POST_EVENT_WAKE_FRAC:
            counts["gate_b_post_wake"] += 1
            tagged.append((start, end, "B"))
            continue

        # Gate C — REM-like window without EMG activation
        if (_is_rem_like_window(start, end, emg_envelopes, emg_baselines) and
                not _has_emg_elevation(start, end, emg_envelopes, emg_baselines)):
            counts["gate_c_rem_emg"] += 1
            tagged.append((start, end, "C"))
            continue

        tagged.append((start, end, "accepted"))

    return tagged, counts


# =============================================================================
# SECTIE 9b — AROUSAL SCORE
#
# Composiet score (0.0–1.0) per event op basis van 6 signaalkenmerken.
# Geen ML, geen labels. Gebaseerd op hetzelfde principe als YASA spindle
# detectie (meerdere drempels combineren).
#
# 1. alpha_ratio   : sigmoid van gem. alpha ratio tijdens event (center=2×)
#                    ratio=1× → 0.18 | ratio=2× → 0.5 | ratio=4× → 0.95
#                    Meest gevalideerde EEG marker van NREM arousal.
#
# 2. beta_ratio    : zelfde sigmoid voor beta (18–30 Hz)
#                    Onafhankelijk van alpha — relevant bij PTSD/angst
#                    waar bèta ook verhoogd is bij hyperarousal.
#
# 3. bilateral     : fractie event waarbij F7 én F8 tegelijk actief waren
#                    Echte arousals zijn bilateraal; unilateraal = vaker artefact.
#
# 4. emg_consistency: fractie samples met EMG > baseline
#                    Spiertonus neemt kort toe bij echte arousal.
#                    Onafhankelijke niet-EEG dimensie.
#
# 5. sigma_suppress: 1 − sigmoid(sigma_ratio, center=1.0)
#                    Hoog als sigma DAALDE → spindles gestopt = arousal.
#                    Inversere relatie met spindle_score.
#
# 6. signal_stability: fractie samples in event boven detectiedrempel
#                    Duur-agnostisch kwaliteitsmaat.
#                    2 s volledig verhoogd = 15 s volledig verhoogd.
# =============================================================================

def _sigmoid(x, center, steepness=1.5):
    """Sigmoid gecentreerd op 'center'. Mapt ratio-waarden naar [0, 1]."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))


def compute_arousal_score(start: int, end: int,
                           envelopes: dict, baselines: dict,
                           emg_envelopes: dict, emg_baselines: dict,
                           ratios: dict, channel_masks: dict,
                           sfreq: float = TARGET_SFREQ) -> dict:
    """
    Berekent de 6 component scores en de gewogen arousal_score voor één event.
    Returns dict met keys: arousal_score, alpha_ratio, beta_ratio,
    bilateral, emg_consistency, sigma_suppress, signal_stability.
    """
    scores = {}

    # 1. Alpha ratio score
    alpha_ratios = []
    for ch in ("F7", "F8"):
        if ch in ratios:
            seg = ratios[ch]["alpha"].values[start:end]
            if len(seg) > 0:
                alpha_ratios.append(float(np.mean(seg)))
    scores["alpha_ratio"] = float(
        _sigmoid(np.mean(alpha_ratios), center=ACTIVATION_THRESHOLD)
    ) if alpha_ratios else 0.0

    # 2. Beta ratio score
    beta_ratios = []
    for ch in ("F7", "F8"):
        if ch in ratios:
            seg = ratios[ch]["beta"].values[start:end]
            if len(seg) > 0:
                beta_ratios.append(float(np.mean(seg)))
    scores["beta_ratio"] = float(
        _sigmoid(np.mean(beta_ratios), center=ACTIVATION_THRESHOLD)
    ) if beta_ratios else 0.0

    # 3. Bilateral score
    bilat_seg = channel_masks["bilateral"][start:end]
    scores["bilateral"] = float(bilat_seg.mean()) if len(bilat_seg) > 0 else 0.0

    # 4. EMG consistency score
    emg_fracs = []
    for ch in ("F7", "F8"):
        if ch in emg_envelopes and ch in emg_baselines:
            emg_seg = emg_envelopes[ch][start:end]
            bas_seg = emg_baselines[ch][start:end]
            if len(emg_seg) > 0:
                emg_fracs.append(float((emg_seg > bas_seg).mean()))
    scores["emg_consistency"] = float(np.mean(emg_fracs)) if emg_fracs else 0.5

    # 5. Sigma suppression score (inverseer: lage sigma = hoge score)
    sigma_ratios = []
    for ch in ("F7", "F8"):
        if ch in ratios:
            seg = ratios[ch]["sigma"].values[start:end]
            if len(seg) > 0:
                sigma_ratios.append(float(np.mean(seg)))
    scores["sigma_suppress"] = float(
        1.0 - _sigmoid(np.mean(sigma_ratios), center=1.0)
    ) if sigma_ratios else 0.5

    # 6. Signal stability score (duur-agnostisch)
    combined_seg = channel_masks["combined"][start:end]
    scores["signal_stability"] = float(
        combined_seg.mean()
    ) if len(combined_seg) > 0 else 0.0

    # Gewogen gemiddelde
    weights       = AROUSAL_SCORE_WEIGHTS
    total_w       = sum(weights.values())
    arousal_score = sum(scores[k] * weights[k] for k in weights) / total_w

    # Clip alle component scores naar [0, 1] voor opslag
    # Voorkomt waarden > 1 door numerieke randgevallen (bijv. aan het begin
    # van de opname voordat de 90s baseline gevuld is)
    for k in list(scores.keys()):
        scores[k] = round(float(np.clip(scores[k], 0.0, 1.0)), 4)

    # Arousal score ook clippen en opnieuw berekenen na clipping
    arousal_score = sum(scores[k] * weights[k] for k in weights) / total_w
    scores["arousal_score"] = round(float(np.clip(arousal_score, 0.0, 1.0)), 4)

    return scores


# =============================================================================
# SECTIE 10 — EVENTS NAAR DATAFRAME
# =============================================================================

_OUTPUT_COLUMNS = [
    "start_sample", "end_sample", "start_sec", "end_sec",
    "duration_sec", "duration_category", "F7_active", "F8_active",
    "spindle_score",
    "arousal_score",
    "score_alpha_ratio", "score_beta_ratio", "score_bilateral",
    "score_emg_consistency", "score_sigma_suppress", "score_signal_stability",
    "post_wake_frac", "stage_rk", "stage_label", "gate",
]


def events_to_dataframe(tagged_events, channel_masks,
                         spindle_score, wake_mask, stage_array,
                         envelopes, baselines,
                         emg_envelopes, emg_baselines,
                         ratios,
                         sfreq=TARGET_SFREQ):
    """
    Bouwt de output DataFrame van ALLE events inclusief gefilterde.
    Seconds kolommen opgeslagen met komma als decimaalteken zodat Excel
    (Nederlandse locale) ze correct weergeeft.

    Filter later met: df[df["gate"] == "accepted"]
    """
    if not tagged_events:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    rows     = []
    post_s   = int(POST_EVENT_CHECK_SEC * sfreq)
    n_signal = len(wake_mask)

    for start, end, gate_label in tagged_events:
        dur_sec  = (end - start) / sfreq
        post_end = min(end + post_s, n_signal)

        f7_seg    = channel_masks["F7_only"][start:end]
        f8_seg    = channel_masks["F8_only"][start:end]
        bilat_seg = channel_masks["bilateral"][start:end]

        stage_val = int(pd.Series(stage_array[start:end]).mode()[0])

        ar = compute_arousal_score(
            start, end,
            envelopes, baselines,
            emg_envelopes, emg_baselines,
            ratios, channel_masks, sfreq
        )

        rows.append({
            "start_sample":      start,
            "end_sample":        end,
            "start_sec":         f"{start / sfreq:.4f}".replace(".", ","),
            "end_sec":           f"{end   / sfreq:.4f}".replace(".", ","),
            "duration_sec":      f"{dur_sec:.4f}".replace(".", ","),
            "duration_category": "micro" if dur_sec <= 3.0 else "arousal",
            "F7_active":         bool(f7_seg.any() or bilat_seg.any()),
            "F8_active":         bool(f8_seg.any() or bilat_seg.any()),
            "spindle_score":          f"{spindle_score[start:end].mean():.4f}".replace(".", ","),
            "arousal_score":          f"{ar['arousal_score']:.4f}".replace(".", ","),
            "score_alpha_ratio":      f"{ar['alpha_ratio']:.4f}".replace(".", ","),
            "score_beta_ratio":       f"{ar['beta_ratio']:.4f}".replace(".", ","),
            "score_bilateral":        f"{ar['bilateral']:.4f}".replace(".", ","),
            "score_emg_consistency":  f"{ar['emg_consistency']:.4f}".replace(".", ","),
            "score_sigma_suppress":   f"{ar['sigma_suppress']:.4f}".replace(".", ","),
            "score_signal_stability": f"{ar['signal_stability']:.4f}".replace(".", ","),
            "post_wake_frac":    f"{wake_mask[end:post_end].mean():.4f}".replace(".", ","),
            "stage_rk":          stage_val,
            "stage_label":       RK_LABELS.get(stage_val, "Unscored"),
            "gate":              gate_label,
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
        "n_events_total":     "",
        "gate_a_wake_onset":  "",
        "gate_b_post_wake":   "",
        "gate_c_rem_emg":     "",
        "movement_pct_L":     "",
        "movement_pct_R":     "",
        "error":              "",
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if out_csv.exists():
        log["status"] = "skipped"
        return log

    if not hyp_path.exists():
        log["status"] = "no_hypnogram"
        log["error"]  = f"Hypnogram niet gevonden: {hyp_path}"
        return log

    try:
        with contextlib.redirect_stdout(io.StringIO()):

            hypnogram = load_hypnogram(hyp_path)
            signals   = load_and_preprocess(edf_dir)
            sfreq     = signals["sfreq"]
            n_samples = len(signals[CH_F7])

            wake_mask   = hypnogram_to_wake_mask(hypnogram, n_samples, sfreq)
            stage_array = hypnogram_to_stage_array(hypnogram, n_samples, sfreq)

            movement_mask, mov_stats = remove_movement_artifacts(signals, sfreq)

            envelopes     = compute_band_envelopes(signals, sfreq)
            emg_envelopes = compute_emg_envelopes(signals, sfreq)

            baselines     = compute_rolling_baseline(envelopes, movement_mask, sfreq)
            ratios        = compute_ratio(envelopes, baselines)
            emg_baselines = compute_emg_baseline(emg_envelopes, movement_mask, sfreq)

            activation_masks = compute_activation_mask(ratios)
            channel_masks    = combine_channels(activation_masks)
            spindle_score    = compute_spindle_score(ratios, envelopes)

            candidate_mask = channel_masks["combined"].copy()
            candidate_mask = candidate_mask & ~movement_mask

            events = mask_to_events(candidate_mask)
            events = merge_events(events, sfreq)
            events = apply_duration_filter(events, sfreq)

            tagged_events, gate_counts = apply_microarousal_gates(
                events, wake_mask, emg_envelopes, emg_baselines, sfreq
            )

            df = events_to_dataframe(
                tagged_events, channel_masks,
                spindle_score, wake_mask, stage_array,
                envelopes, baselines,
                emg_envelopes, emg_baselines,
                ratios,
                sfreq
            )

        df.insert(0, "night_id",   ids["night_id"])
        df.insert(0, "subject_id", ids["subject_id"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, sep=";", decimal=".", float_format="%.4f")

        n_accepted = (df["gate"] == "accepted").sum() if "gate" in df.columns else 0
        log.update({
            "status":            "ok",
            "n_events":          n_accepted,
            "n_events_total":    len(df),
            "gate_a_wake_onset": gate_counts["gate_a_wake_onset"],
            "gate_b_post_wake":  gate_counts["gate_b_post_wake"],
            "gate_c_rem_emg":    gate_counts["gate_c_rem_emg"],
            "movement_pct_L":    mov_stats.get(CH_F7, ""),
            "movement_pct_R":    mov_stats.get(CH_F8, ""),
        })

    except Exception:
        # Full traceback — not truncated so we can diagnose any error
        log["error"] = traceback.format_exc()

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

    ok_df   = log_df[log_df["status"] == "ok"]
    n_ok    = len(ok_df)
    n_skip  = (log_df["status"] == "skipped").sum()
    n_nohyp = (log_df["status"] == "no_hypnogram").sum()
    n_fail  = (log_df["status"] == "failed").sum()
    mins    = (datetime.now() - t_start).total_seconds() / 60

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
            print(f"  {row['error']}")   # full traceback, not truncated

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