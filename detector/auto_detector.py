"""
=============================================================================
PHASE 9 — Candidate event pipeline direct vanuit losse ZMax EDF bestanden
           (geen tussenliggende .fif of samengevoegde EDF nodig)

Per nacht worden de losse kanaalbestanden geladen:
  EEG L.edf, EEG R.edf, dX.edf, dY.edf, dZ.edf
  OXY_IR_AC.edf wordt ingeladen maar nog niet gebruikt in de detectie.

Mappenstructuur verwacht:
  RAW_ROOT / GROUP / subject_id / subject_id_T0_N1 / subject_id_T0_N1_edf /
      EEG L.edf
      EEG R.edf
      dX.edf
      dY.edf
      dZ.edf
      OXY_IR_AC.edf   (optioneel)

Hypnogram (verplicht):
  Per nacht moet een human-rated hypnogram CSV aanwezig zijn:
    .../subject_id_T0_N1/sleepArchitecture/subject_id_T0_N1.csv
  Codering: Rechtschaffen & Kales
    0 = Wake, 1 = N1, 2 = N2, 3 = N3/SWS, 5 = REM
  Nachten zonder hypnogram worden overgeslagen.

Gates:
  Gate A — Meerderheid (>= GATE_A_WAKE_FRAC) van het event in een wake epoch
            (Lucija's scoring) → verwijderd.
  Gate B — Post-event: >= 50% van de 15 s na het event is wake (Lucija's scoring)
            → patient werd wakker = volledige arousal, geen micro-arousal.

Output:
  EVENTS_DIR / GROUP / subject_id / candidate_events_{subject_id}_{night_id}.csv

Gebruik:
  python auto_detector.py             # volledige batch
  python auto_detector.py --limit 3   # testen op 3 nachten
  python auto_detector.py --jobs 4    # 4 parallelle workers
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
EVENTS_DIR = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\2_candidate_events")

EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# OXY_IR_AC wordt ingeladen maar nog niet gebruikt in de detectie of scoring.
CHANNEL_FILES = {
    "EEG L psg-lp": "EEG L",
    "EEG R psg-lp": "EEG R",
    "dX":           "dX",
    "dY":           "dY",
    "dZ":           "dZ",
}

CH_F7  = "EEG L psg-lp"
CH_F8  = "EEG R psg-lp"
EEG_CH = [CH_F7, CH_F8]
MOV_CH = ["dX", "dY", "dZ"]
ALL_CH = EEG_CH + MOV_CH

# OXY_IR_AC: apart van CHANNEL_FILES omdat het optioneel is en
# nog niet in de pipeline wordt gebruikt.
CH_OXY     = "OXY_IR_AC"
OXY_FILE   = "OXY_IR_AC"

TARGET_SFREQ          = 128
NOTCH_HZ              = 50.0
EEG_L_FREQ            = 0.1
EEG_H_FREQ            = 35.0
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
MERGE_GAP_SEC        = 3.0
MIN_DUR_SEC          = 3.0
MAX_DUR_SEC          = 20.0

POST_EVENT_CHECK_SEC = 15.0
POST_EVENT_WAKE_FRAC = 0.5

# Gate A: fractie van het event die in wake moet liggen om te worden verwijderd.
GATE_A_WAKE_FRAC = 0.5

N_JOBS     = -1
STAGE_WAKE = 0
STAGE_REM  = 5
EPOCH_SEC  = 30

# ── Arousal score gewichten ───────────────────────────────────────────────────
AROUSAL_SCORE_WEIGHTS = {
    "alpha_ratio":      0.30,
    "beta_ratio":       0.25,
    "bilateral":        0.20,
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
# =============================================================================

def load_and_preprocess(edf_dir: Path) -> dict:
    """
    Laadt losse EDF kanaalbestanden en past preprocessing toe.
    OXY_IR_AC wordt apart ingeladen en doorgegeven als ruwe array
    in signals["OXY_IR_AC"] zonder verdere verwerking.
    Vereist: EEG L.edf en EEG R.edf moeten aanwezig en leesbaar zijn.
    """
    loaded      = {}
    sfreq_ref   = None
    load_errors = []

    unique_files = set(CHANNEL_FILES.values())
    for raw_file in unique_files:
        edf_path = edf_dir / f"{raw_file}.edf"
        if not edf_path.exists():
            load_errors.append(f"{raw_file}.edf: bestand niet gevonden")
            continue
        try:
            raw  = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            if raw.n_times == 0:
                load_errors.append(f"{raw_file}.edf: leeg bestand (0 samples)")
                continue
            data  = raw.get_data()[0] * 1e6
            sfreq = raw.info["sfreq"]
            loaded[raw_file] = (data, sfreq)
            if sfreq_ref is None:
                sfreq_ref = sfreq
        except Exception as e:
            load_errors.append(f"{raw_file}.edf: {e}")

    for required in ["EEG L", "EEG R"]:
        if required not in loaded:
            raise ValueError(
                f"Verplicht kanaal '{required}' kon niet geladen worden "
                f"uit {edf_dir}.\nLaadfouten:\n{chr(10).join(load_errors)}"
            )

    arrays = {}
    for ch_name, raw_file in CHANNEL_FILES.items():
        if raw_file in loaded:
            arrays[ch_name] = loaded[raw_file][0].copy()

    if not arrays:
        raise ValueError(f"Geen bruikbare kanalen in {edf_dir}")

    min_len      = min(len(d) for d in arrays.values())
    min_required = max(1000, int(10 * (sfreq_ref or TARGET_SFREQ)))
    if min_len < min_required:
        raise ValueError(
            f"Signaal te kort voor filtering: {min_len} samples "
            f"(minimum {min_required})."
        )

    for k in arrays:
        arrays[k] = arrays[k][:min_len]

    ch_names = list(arrays.keys())
    ch_types = ["eeg" if "psg-lp" in n else "misc" for n in ch_names]

    data_mx = np.stack(list(arrays.values())) * 1e-6
    info    = mne.create_info(ch_names=ch_names, sfreq=sfreq_ref,
                               ch_types=ch_types, verbose=False)
    raw     = mne.io.RawArray(data_mx, info, verbose=False)
    raw._data = raw._data.astype(np.float64)

    eeg_lp_chs = [ch for ch in raw.ch_names if "psg-lp" in ch]
    mov_chs    = [ch for ch in raw.ch_names if ch in MOV_CH]

    # 1. DC removal
    for ch in eeg_lp_chs:
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

    # 4. DC removal accelerometer
    if mov_chs:
        raw.apply_function(lambda x: x - np.mean(x),
                           picks=mov_chs, verbose=False)

    # 5. Resample naar 128 Hz
    if sfreq_ref != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    signals = {ch: raw.get_data(picks=ch)[0] * 1e6
               for ch in raw.ch_names}
    signals["sfreq"] = TARGET_SFREQ

    # ── OXY_IR_AC: apart inladen, ruw opslaan, nog niet gebruiken ────────────
    oxy_path = edf_dir / f"{OXY_FILE}.edf"
    if oxy_path.exists():
        try:
            oxy_raw = mne.io.read_raw_edf(oxy_path, preload=True, verbose=False)
            signals[CH_OXY] = oxy_raw.get_data()[0]   # in V, ruw
        except Exception as e:
            signals[CH_OXY] = None
            # Niet fataal — OXY_IR_AC is optioneel
    else:
        signals[CH_OXY] = None

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


# =============================================================================
# SECTIE 7 — HYPNOGRAM LADEN + MASKERS
# =============================================================================

def load_hypnogram(hyp_path: Path) -> np.ndarray:
    df = pd.read_csv(hyp_path, header=None)
    return df.iloc[:, 0].values.astype(int)


def _hypnogram_to_mask(hypnogram: np.ndarray, n_samples: int,
                        target_stage: int,
                        sfreq: float = TARGET_SFREQ,
                        epoch_sec: int = EPOCH_SEC) -> np.ndarray:
    samples_per_epoch = int(epoch_sec * sfreq)
    mask              = np.zeros(n_samples, dtype=bool)
    for epoch_idx, score in enumerate(hypnogram):
        start = epoch_idx * samples_per_epoch
        end   = min(start + samples_per_epoch, n_samples)
        if start >= n_samples:
            break
        if score == target_stage:
            mask[start:end] = True
    return mask


def hypnogram_to_wake_mask(hypnogram: np.ndarray, n_samples: int,
                            sfreq: float = TARGET_SFREQ,
                            epoch_sec: int = EPOCH_SEC) -> np.ndarray:
    return _hypnogram_to_mask(hypnogram, n_samples, STAGE_WAKE, sfreq, epoch_sec)


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


# =============================================================================
# SECTIE 9 — MICRO-AROUSAL GATES
#
# Gate A — Wake uitsluiting (Lucija's scoring)
#   Als >= GATE_A_WAKE_FRAC van de samples in het event in een wake epoch
#   valt → verwijderen. Strenger dan alleen de onset sample checken.
#
# Gate B — Post-event wake check (Lucija's scoring)
#   Kijk POST_EVENT_CHECK_SEC (15 s) na het event. Als >= POST_EVENT_WAKE_FRAC
#   (50%) wake is → patient werd wakker = volledige arousal, geen micro-arousal.
# =============================================================================

def apply_microarousal_gates(events, wake_mask, sfreq=TARGET_SFREQ):
    """
    Past twee gates toe op de candidate event lijst.
    Alle events worden teruggegeven met een gate label.

    gate waarden in output:
      'accepted' — beide gates gepasseerd, dit is een candidate micro-arousal
      'A'        — meerderheid event in wake (Lucija's scoring)
      'B'        — patient werd wakker na event (Lucija's scoring)
    """
    post_s  = int(POST_EVENT_CHECK_SEC * sfreq)
    n       = len(wake_mask)
    tagged  = []
    counts  = {"gate_a_wake_onset": 0,
               "gate_b_post_wake":  0}

    for start, end in events:

        # Gate A — Wake uitsluiting
        if wake_mask[start:end].mean() >= GATE_A_WAKE_FRAC:
            counts["gate_a_wake_onset"] += 1
            tagged.append((start, end, "A"))
            continue

        # Gate B — Post-event wake check
        post_window = wake_mask[end:min(end + post_s, n)]
        if len(post_window) > 0 and post_window.mean() >= POST_EVENT_WAKE_FRAC:
            counts["gate_b_post_wake"] += 1
            tagged.append((start, end, "B"))
            continue

        tagged.append((start, end, "accepted"))

    return tagged, counts


# =============================================================================
# SECTIE 9b — AROUSAL SCORE
#
# 5 componenten:
#
# 1. alpha_ratio      (0.30) sigmoid van gem. alpha ratio tijdens event
# 2. beta_ratio       (0.25) sigmoid van gem. beta ratio tijdens event
# 3. bilateral        (0.20) fractie event waarbij F7 en F8 tegelijk actief
# 4. sigma_suppress   (0.10) 1 - sigmoid(sigma_ratio): spindles gestopt = arousal
# 5. signal_stability (0.15) fractie samples boven detectiedrempel
# =============================================================================

def _sigmoid(x, center, steepness=1.5):
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))


def compute_arousal_score(start: int, end: int,
                           envelopes: dict, baselines: dict,
                           ratios: dict, channel_masks: dict,
                           sfreq: float = TARGET_SFREQ) -> dict:
    scores = {}

    # 1. Alpha ratio
    alpha_ratios = []
    for ch in ("F7", "F8"):
        if ch in ratios:
            seg = ratios[ch]["alpha"].values[start:end]
            if len(seg) > 0:
                alpha_ratios.append(float(np.mean(seg)))
    scores["alpha_ratio"] = float(
        _sigmoid(np.mean(alpha_ratios), center=ACTIVATION_THRESHOLD)
    ) if alpha_ratios else 0.0

    # 2. Beta ratio
    beta_ratios = []
    for ch in ("F7", "F8"):
        if ch in ratios:
            seg = ratios[ch]["beta"].values[start:end]
            if len(seg) > 0:
                beta_ratios.append(float(np.mean(seg)))
    scores["beta_ratio"] = float(
        _sigmoid(np.mean(beta_ratios), center=ACTIVATION_THRESHOLD)
    ) if beta_ratios else 0.0

    # 3. Bilateral
    bilat_seg = channel_masks["bilateral"][start:end]
    scores["bilateral"] = float(bilat_seg.mean()) if len(bilat_seg) > 0 else 0.0

    # 4. Sigma suppression
    sigma_ratios = []
    for ch in ("F7", "F8"):
        if ch in ratios:
            seg = ratios[ch]["sigma"].values[start:end]
            if len(seg) > 0:
                sigma_ratios.append(float(np.mean(seg)))
    scores["sigma_suppress"] = float(
        1.0 - _sigmoid(np.mean(sigma_ratios), center=1.0)
    ) if sigma_ratios else 0.5

    # 5. Signal stability
    combined_seg = channel_masks["combined"][start:end]
    scores["signal_stability"] = float(
        combined_seg.mean()
    ) if len(combined_seg) > 0 else 0.0

    weights       = AROUSAL_SCORE_WEIGHTS
    total_w       = sum(weights.values())
    arousal_score = sum(scores[k] * weights[k] for k in weights) / total_w

    for k in list(scores.keys()):
        scores[k] = round(float(np.clip(scores[k], 0.0, 1.0)), 4)

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
    "score_sigma_suppress", "score_signal_stability",
    "post_wake_frac", "stage_rk", "stage_label", "gate",
]


def events_to_dataframe(tagged_events, channel_masks,
                         spindle_score, wake_mask, stage_array,
                         envelopes, baselines, ratios,
                         sfreq=TARGET_SFREQ):
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
            ratios, channel_masks, sfreq
        )

        rows.append({
            "start_sample":           start,
            "end_sample":             end,
            "start_sec":              f"{start / sfreq:.4f}".replace(".", ","),
            "end_sec":                f"{end   / sfreq:.4f}".replace(".", ","),
            "duration_sec":           f"{dur_sec:.4f}".replace(".", ","),
            "duration_category":      "micro" if dur_sec <= 3.0 else "arousal",
            "F7_active":              bool(f7_seg.any() or bilat_seg.any()),
            "F8_active":              bool(f8_seg.any() or bilat_seg.any()),
            "spindle_score":          f"{spindle_score[start:end].mean():.4f}".replace(".", ","),
            "arousal_score":          f"{ar['arousal_score']:.4f}".replace(".", ","),
            "score_alpha_ratio":      f"{ar['alpha_ratio']:.4f}".replace(".", ","),
            "score_beta_ratio":       f"{ar['beta_ratio']:.4f}".replace(".", ","),
            "score_bilateral":        f"{ar['bilateral']:.4f}".replace(".", ","),
            "score_sigma_suppress":   f"{ar['sigma_suppress']:.4f}".replace(".", ","),
            "score_signal_stability": f"{ar['signal_stability']:.4f}".replace(".", ","),
            "post_wake_frac":         f"{wake_mask[end:post_end].mean():.4f}".replace(".", ","),
            "stage_rk":               stage_val,
            "stage_label":            RK_LABELS.get(stage_val, "Unscored"),
            "gate":                   gate_label,
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
        "group":             ids["group"],
        "subject_id":        ids["subject_id"],
        "night_id":          ids["night_id"],
        "edf_dir":           str(edf_dir),
        "output_csv":        str(out_csv),
        "hypnogram_path":    str(hyp_path),
        "status":            "failed",
        "n_events":          "",
        "n_events_total":    "",
        "gate_a_wake_onset": "",
        "gate_b_post_wake":  "",
        "oxy_available":     "",
        "movement_pct_L":    "",
        "movement_pct_R":    "",
        "error":             "",
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

            envelopes = compute_band_envelopes(signals, sfreq)
            baselines = compute_rolling_baseline(envelopes, movement_mask, sfreq)
            ratios    = compute_ratio(envelopes, baselines)

            activation_masks = compute_activation_mask(ratios)
            channel_masks    = combine_channels(activation_masks)
            spindle_score    = compute_spindle_score(ratios, envelopes)

            candidate_mask = channel_masks["combined"].copy()
            candidate_mask = candidate_mask & ~movement_mask

            events = mask_to_events(candidate_mask)
            events = merge_events(events, sfreq)
            events = apply_duration_filter(events, sfreq)

            tagged_events, gate_counts = apply_microarousal_gates(
                events, wake_mask, sfreq
            )

            df = events_to_dataframe(
                tagged_events, channel_masks,
                spindle_score, wake_mask, stage_array,
                envelopes, baselines, ratios, sfreq
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
            "oxy_available":     signals[CH_OXY] is not None,
            "movement_pct_L":    mov_stats.get(CH_F7, ""),
            "movement_pct_R":    mov_stats.get(CH_F8, ""),
        })

    except Exception:
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
        n_oxy  = ok_df["oxy_available"].sum() if "oxy_available" in ok_df else "?"
        print(f"  Totaal events        : {int(to_num('n_events'))}")
        print(f"  Gate A (wake)        : {int(to_num('gate_a_wake_onset'))}")
        print(f"  Gate B (post-wake)   : {int(to_num('gate_b_post_wake'))}")
        print(f"  OXY_IR_AC aanwezig   : {n_oxy} / {n_ok} nachten")

    print(f"  Log : {log_path}")
    print(f"{'=' * 65}")

    if n_fail > 0:
        for _, row in log_df[log_df["status"] == "failed"].iterrows():
            print(f"\n  FAIL: {row['subject_id']} / {row['night_id']}")
            print(f"  {row['error']}")

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


    