"""
=============================================================================
PHASE 9 — Batch candidate event pipeline (Fasen 5 t/m 8)

Verwerkt alle preprocessed nachten (.fif) parallel:
  - Fase 5: band-envelopes (theta / alpha / sigma / beta)
  - Fase 6: lokale baseline + ratio
  - Fase 7: candidate activation mask  ← uitgebreid met:
              * wake mask (signaal-afgeleid, geen hypnogram nodig)
              * REM-like detectie via EMG suppressie
  - Fase 7b: post-event wake check     ← NIEUW (micro-arousal definitie)
              * Na een event mag de slaap NIET overgaan in wake
              * Als het 15 s na het event wake-like is → verwijderen
  - Fase 8: event boundaries + DataFrame

Wijzigingen t.o.v. origineel:
  1. BASELINE_SEC: 30 → 90 s  (stabieler voor BNBD populatie met
     gefragmenteerde slaap; Popovic gebruikte ook 90 s)
  2. Wake mask (signaal-afgeleid):
       - beta EN emg gezamenlijk verhoogd > WAKE_THRESHOLD_MULT × baseline
       - aanhoudend ≥ WAKE_MIN_DUR_SEC   → wake-like periode
       - Candidates die starten in wake-like periode worden verwijderd
  3. Post-event wake check (kern van micro-arousal definitie):
       - Kijk POST_EVENT_CHECK_SEC na het einde van het event
       - Als ≥ POST_EVENT_WAKE_FRAC van dat venster wake-like is
         → dit was een volledige arousal (patiënt werd wakker), geen
           micro-arousal → verwijderen
  4. REM-specific EMG gate (Popovic regel):
       - Als het event valt in een REM-like venster (EMG gesupprimeerd)
         moet het event zelf ook EMG-activatie tonen
       - Zonder EMG-activatie in REM = normale sawtooth activiteit,
         geen echte arousal → verwijderen

Output per nacht:
  EVENTS_DIR / GROUP / subject_id / candidate_events_{subject_id}_{night_id}.csv

Bestaande CSV-bestanden worden overgeslagen (idempotent).

Gebruik:
  uv run phase5to8-new.py     # volledige batch
  uv run phase5to8-new.py --limit 3  # eerst testen op 3 nachten
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
# SECTIE 1 — CONSTANTEN
# Alle instelbare parameters op één plek.
# =============================================================================

# ── Kanalen ───────────────────────────────────────────────────────────────────
EEG_CH = ["EEG L psg-lp", "EEG R psg-lp"]
EMG_CH = ["EEG L psg-emg", "EEG R psg-emg"]
MOV_CH = ["dX", "dY", "dZ"]
ALL_CH = EEG_CH + EMG_CH + MOV_CH

CH_F7 = "EEG L psg-lp"
CH_F8 = "EEG R psg-lp"
CH_EMG_L = "EEG L psg-emg"
CH_EMG_R = "EEG R psg-emg"

# ── Signaalparameters ─────────────────────────────────────────────────────────
TARGET_SFREQ          = 128
MOVEMENT_THRESHOLD_UV = 1000.0

# ── Frequentiebanden ──────────────────────────────────────────────────────────
BANDS = {
    "theta": (4.0,  7.0),
    "alpha": (8.0,  12.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}

# ── Envelope smoothing ────────────────────────────────────────────────────────
SMOOTH_SEC = 0.5

# ── Baseline
# GEWIJZIGD: 30 → 90 s (Popovic gebruikte 90 s; stabieler voor BNBD populatie
# met gefragmenteerde slaap en verhoogde beta bij angst/PTSD)
BASELINE_SEC = 90.0

# ── Activatiedrempel voor arousal detectie ────────────────────────────────────
ACTIVATION_THRESHOLD = 2.0   # envelope > 2× baseline → kandidaat
SPINDLE_THRESHOLD    = 2.0   # sigma ratio drempel voor spindle score

# ── Event grenzen ─────────────────────────────────────────────────────────────
MERGE_GAP_SEC = 1.0    # gaten < 1 s tussen segmenten worden samengevoegd
MIN_DUR_SEC   = 1.0    # < 1 s = ruis → verwijderen
MAX_DUR_SEC   = 15.0   # > 15 s = volledige arousal / wake → verwijderen

# ── Wake mask parameters (signaal-afgeleid, geen hypnogram nodig) ─────────────
# Een venster is 'wake-like' als beta EN emg beide boven drempel zijn
# gedurende minstens WAKE_MIN_DUR_SEC seconden aaneengesloten.
# Hogere multiplier (2.5 vs 2.0) reduceert vals-positieve wake-classificatie
# bij hoge-beta slapers (veel voorkomend bij angststoornissen / PTSD).
WAKE_THRESHOLD_MULT = 2.5
WAKE_MIN_DUR_SEC    = 20.0   # aanhoudende activatie nodig om als wake te tellen
                              # korte burst = arousal; langdurig = wakker

# ── Post-event wake check (kern van micro-arousal definitie) ──────────────────
# Na een micro-arousal keert de patiënt terug naar slaap.
# Na een volledige arousal blijft de patiënt wakker → verwijderen.
POST_EVENT_CHECK_SEC  = 15.0  # hoe ver na het event wordt gekeken
POST_EVENT_WAKE_FRAC  = 0.5   # ≥ 50% van dat venster wake-like → verwijderen

# ── REM-specific EMG gate ─────────────────────────────────────────────────────
# In REM is EMG gesupprimeerd. Als een event valt in een REM-like venster
# (EMG < REM_EMG_SUPPRESSION_FRAC × baseline), moet het event zelf ook
# EMG-activatie tonen om als echte arousal te tellen.
REM_EMG_SUPPRESSION_FRAC = 0.8  # EMG < 80% baseline = REM-like venster

# ── Mappen ────────────────────────────────────────────────────────────────────
PREP_DIR   = Path(r"C:\Users\zafar\Documents\THESIS_OUTPUTS\1_preprocessing_EEG")
EVENTS_DIR = Path(r"C:\Users\zafar\Documents\THESIS_OUTPUTS\2_candidate_events")
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

N_JOBS = -1   # -1 = alle beschikbare CPU-cores


# =============================================================================
# SECTIE 2 — HULPFUNCTIES (ongewijzigd)
# =============================================================================

def extract_ids(edf_path: Path) -> dict:
    stem  = edf_path.stem
    parts = stem.replace("_psg", "").split("_")
    return {
        "subject_id": f"bnbd_{parts[1]}_{parts[2]}",
        "night_id":   f"{parts[3]}_{parts[4]}",
        "group":      parts[1].upper(),
    }


def preprocess_signals(raw) -> dict:
    return {ch: raw.get_data(picks=ch)[0] for ch in ALL_CH if ch in raw.ch_names}


def remove_movement_artifacts(signals: dict,
                               sfreq: int = TARGET_SFREQ) -> tuple:
    n_samples     = len(next(iter(signals.values())))
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
# SECTIE 3 — FASE 5: Band-envelopes (ongewijzigd)
# =============================================================================

def bandpass_filter(signal: np.ndarray, lo: float, hi: float,
                    sfreq: float = TARGET_SFREQ) -> np.ndarray:
    filtered = mne.filter.filter_data(
        signal[np.newaxis, :],
        sfreq=sfreq, l_freq=lo, h_freq=hi,
        method="fir", fir_window="hamming", verbose=False,
    )
    return filtered[0]


def compute_envelope(signal: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(signal)).astype(np.float32)


def smooth_envelope(envelope: np.ndarray, smooth_sec: float = SMOOTH_SEC,
                    sfreq: float = TARGET_SFREQ) -> np.ndarray:
    window = int(smooth_sec * sfreq)
    kernel = np.ones(window) / window
    return np.convolve(envelope, kernel, mode="same").astype(np.float32)


def compute_band_envelopes(signals: dict,
                            sfreq: float = TARGET_SFREQ) -> dict:
    """
    Berekent gefilterde + gesmoothe envelopes voor theta/alpha/sigma/beta
    voor zowel F7 (links) als F8 (rechts).
    """
    time_axis = np.arange(len(signals[CH_F7])) / sfreq
    result    = {}
    for ch_label, ch_name in [("F7", CH_F7), ("F8", CH_F8)]:
        if ch_name not in signals:
            continue
        sig  = signals[ch_name]
        rows = {"time": time_axis}
        for band_name, (lo, hi) in BANDS.items():
            filtered        = bandpass_filter(sig, lo, hi, sfreq)
            env             = compute_envelope(filtered)
            rows[band_name] = smooth_envelope(env, SMOOTH_SEC, sfreq)
        result[ch_label] = pd.DataFrame(rows)
    return result


def compute_emg_envelopes(signals: dict,
                           sfreq: float = TARGET_SFREQ) -> dict:
    """
    Berekent EMG envelopes voor L en R kanaal.
    ZMax EMG kanalen zijn al hardware high-pass gefilterd (~32 Hz),
    dus alleen envelope berekening is nodig.
    Wordt gebruikt voor wake mask en REM gate.
    """
    result = {}
    for ch_label, ch_name in [("F7", CH_EMG_L), ("F8", CH_EMG_R)]:
        if ch_name not in signals:
            continue
        env = compute_envelope(signals[ch_name])
        result[ch_label] = smooth_envelope(env, SMOOTH_SEC, sfreq)
    return result


# =============================================================================
# SECTIE 4 — FASE 6: Lokale baseline + ratio (BASELINE_SEC gewijzigd naar 90 s)
# =============================================================================

def _rolling_median_causal(arr: np.ndarray, sfreq: float,
                            window_s: float) -> np.ndarray:
    """
    Causale rollende mediaan: kijkt alleen terug in de tijd (geen toekomst).
    Gebruikt strided array voor efficiëntie na de eerste window.

    Mediaan i.p.v. gemiddelde maakt de baseline robuust tegen uitschieters
    (K-complexen, bewegingspieken) — belangrijk voor BNBD populatie.
    """
    w   = int(window_s * sfreq)
    n   = len(arr)
    out = np.empty(n, dtype=np.float32)

    # Bootstrap: gebruik alle beschikbare data voor het eerste volledige venster
    for i in range(min(w, n)):
        out[i] = np.median(arr[:i + 1])

    # Efficiënte strided mediaan voor de rest
    shape   = (n - w, w)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape,
                                               strides=strides)
    out[w:] = np.median(windows, axis=1)
    return out


def compute_rolling_baseline(envelopes: dict,
                              movement_mask: np.ndarray = None,
                              sfreq: float = TARGET_SFREQ,
                              baseline_sec: float = BASELINE_SEC) -> dict:
    """
    Berekent causale rollende mediaan baseline per band per kanaal.
    Artefact-samples (movement_mask=True) worden op NaN gezet voor de
    baseline berekening zodat ze de baseline niet verstoren.

    GEWIJZIGD: gebruikt nu _rolling_median_causal in plaats van
    pandas rolling (explicieter causaal, zelfde resultaat).
    """
    baselines = {}
    for ch_label, df in envelopes.items():
        rows = {"time": df["time"].values}
        for band in BANDS:
            env = df[band].copy().astype(np.float64)
            if movement_mask is not None:
                env[movement_mask] = np.nan
            # NaN opvullen met forward/backward fill voor de mediaan berekening
            series = pd.Series(env).ffill().bfill().values.astype(np.float32)
            rows[band] = _rolling_median_causal(series, sfreq, baseline_sec)
        baselines[ch_label] = pd.DataFrame(rows)
    return baselines


def compute_ratio(envelopes: dict, baselines: dict) -> dict:
    """Berekent envelope / baseline ratio per band per kanaal."""
    ratios = {}
    for ch_label in envelopes:
        env_df  = envelopes[ch_label]
        base_df = baselines[ch_label]
        rows    = {"time": env_df["time"].values}
        for band in BANDS:
            rows[band] = (
                env_df[band].values / (base_df[band].values + 1e-6)
            ).astype(np.float32)
        ratios[ch_label] = pd.DataFrame(rows)
    return ratios


def compute_emg_baseline(emg_envelopes: dict,
                          movement_mask: np.ndarray = None,
                          sfreq: float = TARGET_SFREQ,
                          baseline_sec: float = BASELINE_SEC) -> dict:
    """
    Berekent causale rollende mediaan baseline voor EMG envelopes.
    Zelfde methode als voor EEG banden.
    """
    baselines = {}
    for ch_label, env in emg_envelopes.items():
        arr = env.copy().astype(np.float32)
        if movement_mask is not None:
            arr[movement_mask] = np.nan
        arr = pd.Series(arr).ffill().bfill().values.astype(np.float32)
        baselines[ch_label] = _rolling_median_causal(arr, sfreq, baseline_sec)
    return baselines


# =============================================================================
# SECTIE 5 — FASE 7a: Wake mask (NIEUW — signaal-afgeleid)
#
# Bepaalt welke samples 'wake-like' zijn op basis van het EEG/EMG signaal,
# zonder een externe hypnogram te gebruiken.
#
# Logica:
#   1. Beta EN EMG zijn beide boven hun wake drempel (WAKE_THRESHOLD_MULT ×
#      rollende mediaan baseline) tegelijkertijd verhoogd
#   2. Deze gezamenlijke verhoging duurt minstens WAKE_MIN_DUR_SEC seconden
#      (korte burst = arousal; aanhoudend = wake)
#
# De wake mask wordt gebruikt voor twee gates:
#   Gate A: verwijder candidates die starten in wake-like periode
#   Gate B: post-event check (zie Fase 7b)
# =============================================================================

def compute_wake_mask(envelopes: dict, emg_envelopes: dict,
                      baselines: dict, emg_baselines: dict,
                      sfreq: float = TARGET_SFREQ) -> np.ndarray:
    """
    Geeft een boolean array terug (True = wake-like sample).
    Gebruikt het gemiddelde van F7 en F8 voor robuustheid.

    Parameters
    ----------
    envelopes     : EEG band envelopes (uit compute_band_envelopes)
    emg_envelopes : EMG envelopes (uit compute_emg_envelopes)
    baselines     : EEG baselines (uit compute_rolling_baseline)
    emg_baselines : EMG baselines (uit compute_emg_baseline)
    sfreq         : sample frequentie
    """
    n = len(envelopes["F7"]["time"])

    # ── Beta verhoogd (gemiddeld over F7 en F8) ───────────────────────────────
    beta_up_list = []
    for ch in ("F7", "F8"):
        if ch not in envelopes or ch not in baselines:
            continue
        beta_env = envelopes[ch]["beta"].values
        beta_bas = baselines[ch]["beta"].values
        beta_up_list.append(beta_env > WAKE_THRESHOLD_MULT * beta_bas)

    # ── EMG verhoogd (gemiddeld over F7 en F8) ────────────────────────────────
    emg_up_list = []
    for ch in ("F7", "F8"):
        if ch not in emg_envelopes or ch not in emg_baselines:
            continue
        emg_env = emg_envelopes[ch]
        emg_bas = emg_baselines[ch]
        emg_up_list.append(emg_env > WAKE_THRESHOLD_MULT * emg_bas)

    if not beta_up_list:
        # Geen beta beschikbaar: geen wake mask mogelijk
        return np.zeros(n, dtype=bool)

    # Meerderheid van kanalen moet verhoogd zijn
    beta_up = np.mean(beta_up_list, axis=0) >= 0.5

    if emg_up_list:
        emg_up   = np.mean(emg_up_list, axis=0) >= 0.5
        joint_up = beta_up & emg_up
    else:
        # Geen EMG beschikbaar: alleen beta gebruiken (minder specifiek)
        joint_up = beta_up

    # ── Run-length filter: aanhoudende activatie vereist ─────────────────────
    min_samples = int(WAKE_MIN_DUR_SEC * sfreq)
    wake_mask   = np.zeros(n, dtype=bool)
    in_run      = False
    run_start   = 0

    for i in range(n):
        if joint_up[i] and not in_run:
            in_run, run_start = True, i
        elif not joint_up[i] and in_run:
            in_run = False
            if (i - run_start) >= min_samples:
                wake_mask[run_start:i] = True

    # Vang run die doorloopt tot einde opname
    if in_run and (n - run_start) >= min_samples:
        wake_mask[run_start:n] = True

    return wake_mask


# =============================================================================
# SECTIE 6 — FASE 7b: Candidate activation mask + gates (uitgebreid)
#
# Originele logica behouden (alpha/beta ratio drempel, kanaalcombinaties,
# spindle score) en drie nieuwe gates toegevoegd:
#
# Gate A — Wake onset uitsluiting
#   Verwijder candidates die starten in een wake-like periode.
#   Geen vereiste van ≥10 s slaap voor het event (relaxed AASM).
#
# Gate B — Post-event wake check  ← KERN VAN MICRO-AROUSAL DEFINITIE
#   Kijk POST_EVENT_CHECK_SEC na het einde van het event.
#   Als ≥ POST_EVENT_WAKE_FRAC van dat venster wake-like is:
#     → patiënt werd wakker na het event = volledige arousal → verwijderen.
#   Bij een micro-arousal keert de patiënt terug naar slaap.
#
# Gate C — REM-specific EMG gate (Popovic regel)
#   Als het event valt in een REM-like venster (EMG gesupprimeerd):
#     het event zelf moet ook EMG-activatie tonen.
#   Geen EMG-activatie in REM = normale sawtooth activiteit → verwijderen.
# =============================================================================

def compute_activation_mask(ratios: dict,
                             threshold: float = ACTIVATION_THRESHOLD) -> dict:
    """Alpha OF beta ratio boven drempel → kandidaat sample."""
    masks = {}
    for ch_label, df in ratios.items():
        masks[ch_label] = (
            (df["alpha"].values > threshold) |
            (df["beta"].values  > threshold)
        )
    return masks


def combine_channels(masks: dict) -> dict:
    """Combineert F7 en F8 masks tot eenzijdige en bilaterale maskers."""
    F7 = masks["F7"]
    F8 = masks["F8"]
    return {
        "F7_only":   F7 & ~F8,
        "F8_only":   F8 & ~F7,
        "bilateral": F7 & F8,
        "combined":  F7 | F8,
    }


def compute_spindle_score(ratios: dict, envelopes: dict) -> np.ndarray:
    """
    Composite spindle score: sigma dominantie over andere banden
    gecombineerd met sigma ratio drempel.
    Ongewijzigd t.o.v. origineel.
    """
    scores = []
    for ch_label in ("F7", "F8"):
        env = envelopes[ch_label]
        rat = ratios[ch_label]
        c1  = (env["sigma"].values > env["theta"].values).astype(np.float32)
        c2  = (env["sigma"].values > env["alpha"].values).astype(np.float32)
        c3  = (env["sigma"].values > env["beta"].values ).astype(np.float32)
        c4  = (rat["sigma"].values > SPINDLE_THRESHOLD  ).astype(np.float32)
        scores.append((c1 + c2 + c3 + c4) / 4.0)
    return np.mean(scores, axis=0).astype(np.float32)


def _is_rem_like_window(onset: int, offset: int,
                         emg_envelopes: dict,
                         emg_baselines: dict) -> bool:
    """
    Geeft True als het venster [onset, offset] REM-like is:
    EMG envelope is gesupprimeerd (< REM_EMG_SUPPRESSION_FRAC × baseline)
    voor de meerderheid van het venster, gemiddeld over beide kanalen.
    """
    suppressed_fracs = []
    for ch in ("F7", "F8"):
        if ch not in emg_envelopes or ch not in emg_baselines:
            continue
        emg_seg = emg_envelopes[ch][onset:offset]
        bas_seg = emg_baselines[ch][onset:offset]
        suppressed = emg_seg < (REM_EMG_SUPPRESSION_FRAC * bas_seg)
        suppressed_fracs.append(suppressed.mean())

    if not suppressed_fracs:
        return False
    return np.mean(suppressed_fracs) > 0.5


def _has_emg_elevation(onset: int, offset: int,
                        emg_envelopes: dict,
                        emg_baselines: dict) -> bool:
    """
    Geeft True als EMG verhoogd is boven baseline voor de meerderheid
    van het venster [onset, offset] (Popovic REM arousal regel).
    """
    elevated_fracs = []
    for ch in ("F7", "F8"):
        if ch not in emg_envelopes or ch not in emg_baselines:
            continue
        emg_seg = emg_envelopes[ch][onset:offset]
        bas_seg = emg_baselines[ch][onset:offset]
        elevated_fracs.append((emg_seg > bas_seg).mean())

    if not elevated_fracs:
        return True   # geen EMG beschikbaar → niet gaten, event accepteren
    return np.mean(elevated_fracs) >= 0.5


def run_fase7(ratios: dict, envelopes: dict,
              emg_envelopes: dict, emg_baselines: dict,
              wake_mask: np.ndarray,
              movement_mask: np.ndarray = None) -> dict:
    """
    Fase 7: candidate activation mask met alle gates.

    Parameters
    ----------
    ratios        : EEG band ratios (envelope / baseline)
    envelopes     : EEG band envelopes
    emg_envelopes : EMG envelopes per kanaal
    emg_baselines : EMG baselines per kanaal
    wake_mask     : boolean array, True = wake-like sample
    movement_mask : boolean array, True = bewegingsartefact sample

    Returns
    -------
    dict met:
      candidate_mask  : boolean array na bewegingsmasker
      channel_masks   : F7_only, F8_only, bilateral, combined
      spindle_score   : composite spindle score array
      wake_mask       : doorgegeven voor gebruik in Fase 8
      emg_envelopes   : doorgegeven voor REM gate in Fase 8
      emg_baselines   : doorgegeven voor REM gate in Fase 8
      time            : tijdas
    """
    activation_masks = compute_activation_mask(ratios)
    channel_masks    = combine_channels(activation_masks)
    spindle_score    = compute_spindle_score(ratios, envelopes)

    # Combineer kanalen en verwijder bewegingsartefacten
    candidate_mask = channel_masks["combined"].copy()
    if movement_mask is not None:
        candidate_mask = candidate_mask & ~movement_mask

    return {
        "candidate_mask": candidate_mask,
        "channel_masks":  channel_masks,
        "spindle_score":  spindle_score,
        "wake_mask":      wake_mask,
        "emg_envelopes":  emg_envelopes,
        "emg_baselines":  emg_baselines,
        "time":           ratios["F7"]["time"].values,
    }


# =============================================================================
# SECTIE 7 — FASE 8: Event boundaries + DataFrame (uitgebreid met gates)
# =============================================================================

def mask_to_events(candidate_mask: np.ndarray) -> list:
    """Zet binair masker om naar lijst van (start, end) sample-tuples."""
    events   = []
    in_event = False
    start    = 0
    for i, active in enumerate(candidate_mask):
        if active and not in_event:
            start, in_event = i, True
        elif not active and in_event:
            in_event = False
            events.append((start, i))
    if in_event:
        events.append((start, len(candidate_mask)))
    return events


def merge_events(events: list, sfreq: float = TARGET_SFREQ) -> list:
    """Voeg events samen die minder dan MERGE_GAP_SEC van elkaar liggen."""
    if not events:
        return []
    merge_gap = int(MERGE_GAP_SEC * sfreq)
    merged    = [events[0]]
    for start, end in events[1:]:
        if start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def apply_duration_filter(events: list,
                           sfreq: float = TARGET_SFREQ) -> list:
    """Verwijder events buiten het [MIN_DUR_SEC, MAX_DUR_SEC] venster."""
    min_s = int(MIN_DUR_SEC * sfreq)
    max_s = int(MAX_DUR_SEC * sfreq)
    return [(s, e) for s, e in events if min_s <= (e - s) <= max_s]


def apply_microarousal_gates(events: list,
                              wake_mask: np.ndarray,
                              emg_envelopes: dict,
                              emg_baselines: dict,
                              sfreq: float = TARGET_SFREQ) -> tuple:
    """
    Past de drie nieuwe gates toe op de event lijst.

    Gate A — Wake onset uitsluiting:
      Event start in wake-like periode → verwijderen.

    Gate B — Post-event wake check (micro-arousal definitie):
      Na het event wordt POST_EVENT_CHECK_SEC gekeken.
      Als ≥ POST_EVENT_WAKE_FRAC van dat venster wake-like is
      → patiënt werd wakker = volledige arousal → verwijderen.

    Gate C — REM-specific EMG gate (Popovic):
      Event in REM-like venster zonder EMG activatie → verwijderen.

    Returns
    -------
    accepted : lijst van geaccepteerde (start, end) tuples
    gate_counts : dict met aantallen verwijderd per gate
    """
    post_samples = int(POST_EVENT_CHECK_SEC * sfreq)
    n_signal     = len(wake_mask)

    accepted = []
    gate_counts = {"gate_a_wake_onset": 0,
                   "gate_b_post_wake":  0,
                   "gate_c_rem_emg":    0}

    for start, end in events:

        # ── Gate A: onset mag niet in wake-like periode vallen ────────────────
        if wake_mask[start]:
            gate_counts["gate_a_wake_onset"] += 1
            continue

        # ── Gate B: post-event venster mag niet wake-like zijn ────────────────
        # Dit is de kern van de micro-arousal definitie:
        # na een micro-arousal keert de patiënt terug naar slaap.
        post_end    = min(end + post_samples, n_signal)
        post_window = wake_mask[end:post_end]
        if len(post_window) > 0:
            if post_window.mean() >= POST_EVENT_WAKE_FRAC:
                gate_counts["gate_b_post_wake"] += 1
                continue

        # ── Gate C: REM-specific EMG gate ────────────────────────────────────
        if _is_rem_like_window(start, end, emg_envelopes, emg_baselines):
            if not _has_emg_elevation(start, end, emg_envelopes,
                                       emg_baselines):
                gate_counts["gate_c_rem_emg"] += 1
                continue

        accepted.append((start, end))

    return accepted, gate_counts


def events_to_dataframe(events: list, channel_masks: dict,
                         spindle_score: np.ndarray,
                         emg_envelopes: dict,
                         emg_baselines: dict,
                         wake_mask: np.ndarray,
                         movement_mask: np.ndarray = None,
                         sfreq: float = TARGET_SFREQ) -> pd.DataFrame:
    """
    Zet geaccepteerde events om naar een rijke DataFrame.
    Originele kolommen behouden + nieuwe kolommen toegevoegd:
      - rem_like          : bool, event valt in REM-like venster
      - post_wake_frac    : float, fractie wake-like samples na event
      - wake_onset        : bool, event startte vlak na wake periode
                            (True als de 5 s vóór onset > 50% wake-like was)
    """
    rows = []
    post_samples = int(POST_EVENT_CHECK_SEC * sfreq)
    n_signal     = len(wake_mask)

    for start, end in events:
        dur_sec    = (end - start) / sfreq
        is_short   = 1.0  <= dur_sec <= 3.0
        is_arousal = 3.01 <= dur_sec <= 15.0
        cat = ("both"    if (is_short and is_arousal) else
               "short"   if is_short else
               "arousal")

        bilat_seg         = channel_masks["bilateral"][start:end]
        bilateral_overlap = float(bilat_seg.mean())

        # Post-event wake fractie (informatief, event is al geaccepteerd)
        post_end        = min(end + post_samples, n_signal)
        post_wake_frac  = float(wake_mask[end:post_end].mean())

        # Vlak voor onset: was de patiënt net wakker?
        pre_start       = max(0, start - int(5 * sfreq))
        wake_onset      = bool(wake_mask[pre_start:start].mean() > 0.5)

        # REM-like venster
        rem_like = _is_rem_like_window(start, end, emg_envelopes,
                                        emg_baselines)

        rows.append({
            # ── Originele kolommen (ongewijzigd) ──────────────────────────────
            "start_sample":         start,
            "end_sample":           end,
            "start_sec":            round(start / sfreq, 4),
            "end_sec":              round(end   / sfreq, 4),
            "duration_sec":         round(dur_sec, 4),
            "duration_category":    cat,
            "F7_active":            bool(
                (channel_masks["F7_only"][start:end].any()) or
                bilat_seg.any()
            ),
            "F8_active":            bool(
                (channel_masks["F8_only"][start:end].any()) or
                bilat_seg.any()
            ),
            "F7_only":              bool(
                channel_masks["F7_only"][start:end].mean() > 0.5
            ),
            "F8_only":              bool(
                channel_masks["F8_only"][start:end].mean() > 0.5
            ),
            "bilateral":            bool(bilateral_overlap > 0.1),
            "bilateral_overlap":    round(bilateral_overlap, 4),
            "spindle_score_mean":   round(
                float(spindle_score[start:end].mean()), 4
            ),
            "spindle_score_max":    round(
                float(spindle_score[start:end].max()), 4
            ),
            "artifact_overlap_pct": round(
                float(movement_mask[start:end].mean() * 100), 2
            ) if movement_mask is not None else 0.0,
            # ── Nieuwe kolommen ───────────────────────────────────────────────
            "rem_like":             rem_like,
            "post_wake_frac":       round(post_wake_frac, 4),
            "wake_onset":           wake_onset,
        })

    return pd.DataFrame(rows)


def run_fase8(fase7_output: dict,
              movement_mask: np.ndarray = None,
              sfreq: float = TARGET_SFREQ) -> tuple:
    """
    Fase 8: event grenzen bepalen, gates toepassen, DataFrame bouwen.

    Parameters
    ----------
    fase7_output  : dict van run_fase7
    movement_mask : boolean bewegingsartefact masker
    sfreq         : sample frequentie

    Returns
    -------
    df          : DataFrame met geaccepteerde micro-arousals
    gate_counts : dict met aantallen verwijderd per gate (voor logging)
    """
    candidate_mask = fase7_output["candidate_mask"]
    channel_masks  = fase7_output["channel_masks"]
    spindle_score  = fase7_output["spindle_score"]
    wake_mask      = fase7_output["wake_mask"]
    emg_envelopes  = fase7_output["emg_envelopes"]
    emg_baselines  = fase7_output["emg_baselines"]

    # ── Stap 1: masker → events ───────────────────────────────────────────────
    events = mask_to_events(candidate_mask)

    # ── Stap 2: gaten sluiten ─────────────────────────────────────────────────
    events = merge_events(events, sfreq)

    # ── Stap 3: duur filter (1–15 s) ─────────────────────────────────────────
    events = apply_duration_filter(events, sfreq)

    # ── Stap 4: micro-arousal gates (A, B, C) ────────────────────────────────
    events, gate_counts = apply_microarousal_gates(
        events, wake_mask, emg_envelopes, emg_baselines, sfreq
    )

    # ── Stap 5: DataFrame bouwen ──────────────────────────────────────────────
    df = events_to_dataframe(
        events        = events,
        channel_masks = channel_masks,
        spindle_score = spindle_score,
        emg_envelopes = emg_envelopes,
        emg_baselines = emg_baselines,
        wake_mask     = wake_mask,
        movement_mask = movement_mask,
        sfreq         = sfreq,
    )

    return df, gate_counts


# =============================================================================
# SECTIE 8 — Pad-helpers (ongewijzigd)
# =============================================================================

def _ids_from_fif(fif_path: Path) -> dict:
    stem = fif_path.stem.replace("_prep_raw", "")
    return extract_ids(Path(stem))


def _csv_path(ids: dict) -> Path:
    return (
        EVENTS_DIR
        / ids["group"]
        / ids["subject_id"]
        / f"candidate_events_{ids['subject_id']}_{ids['night_id']}.csv"
    )


def _find_fif_files() -> list:
    return sorted(PREP_DIR.rglob("*_psg_prep_raw.fif"))


# =============================================================================
# SECTIE 9 — Verwerking van één nacht (worker functie)
# =============================================================================

def _process_night(fif_path: Path) -> dict:
    """
    Volledige pipeline (Fasen 5-8) voor één nacht.
    Draait stil — prints onderdrukt zodat parallelle output leesbaar blijft.
    Geeft een log-dict terug: status 'ok' / 'skipped' / 'failed'.
    """
    ids     = _ids_from_fif(fif_path)
    out_csv = _csv_path(ids)

    log = {
        "group":               ids["group"],
        "subject_id":          ids["subject_id"],
        "night_id":            ids["night_id"],
        "fif_path":            str(fif_path),
        "output_csv":          str(out_csv),
        "status":              "failed",
        "n_events":            "",
        "gate_a_wake_onset":   "",
        "gate_b_post_wake":    "",
        "gate_c_rem_emg":      "",
        "error":               "",
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if out_csv.exists():
        log["status"] = "skipped"
        return log

    try:
        with contextlib.redirect_stdout(io.StringIO()):

            # Fase 5: band-envelopes
            raw           = mne.io.read_raw_fif(fif_path, preload=True,
                                                 verbose=False)
            signals       = preprocess_signals(raw)
            envelopes     = compute_band_envelopes(signals)
            emg_envelopes = compute_emg_envelopes(signals)

            # Fase 6: bewegingsmasker, baseline, ratio
            movement_mask, _ = remove_movement_artifacts(signals)
            baselines         = compute_rolling_baseline(envelopes,
                                                          movement_mask)
            ratios            = compute_ratio(envelopes, baselines)
            emg_baselines     = compute_emg_baseline(emg_envelopes,
                                                      movement_mask)

            # Fase 7a: wake mask (signaal-afgeleid)
            wake_mask = compute_wake_mask(
                envelopes, emg_envelopes, baselines, emg_baselines
            )

            # Fase 7b: candidate activation mask
            fase7 = run_fase7(
                ratios        = ratios,
                envelopes     = envelopes,
                emg_envelopes = emg_envelopes,
                emg_baselines = emg_baselines,
                wake_mask     = wake_mask,
                movement_mask = movement_mask,
            )

            # Fase 8: event grenzen + gates + DataFrame
            df, gate_counts = run_fase8(fase7, movement_mask=movement_mask)

        # Voeg identifiers toe als eerste kolommen
        df.insert(0, "night_id",   ids["night_id"])
        df.insert(0, "subject_id", ids["subject_id"])

        # Opslaan
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        log["status"]             = "ok"
        log["n_events"]           = len(df)
        log["gate_a_wake_onset"]  = gate_counts["gate_a_wake_onset"]
        log["gate_b_post_wake"]   = gate_counts["gate_b_post_wake"]
        log["gate_c_rem_emg"]     = gate_counts["gate_c_rem_emg"]

    except Exception:
        log["error"] = traceback.format_exc()[-800:]

    return log


# =============================================================================
# SECTIE 10 — Batch runner (uitgebreid met gate statistieken in overzicht)
# =============================================================================

def run_batch_events(limit: int = None,
                     n_jobs: int = N_JOBS) -> pd.DataFrame:
    """
    Verwerk alle preprocessed nachten parallel (Fasen 5-8).

    Parameters
    ----------
    limit  : int, optioneel — verwerk alleen de eerste N nachten (voor testen)
    n_jobs : int            — parallelle workers; -1 = alle CPU-cores

    Returns
    -------
    pd.DataFrame met verwerkingslog inclusief gate statistieken per nacht
    """
    t_start = datetime.now()
    print("=" * 65)
    print("  CANDIDATE EVENT BATCH  |  Fasen 5 t/m 8")
    print(f"  Start : {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Baseline: 90 s  |  Gates: A (wake onset), "
          "B (post-event wake), C (REM EMG)")
    print("=" * 65)

    fif_files = _find_fif_files()
    if limit:
        fif_files = fif_files[:limit]

    ids_list = [_ids_from_fif(f) for f in fif_files]
    n_skip   = sum(1 for ids in ids_list if _csv_path(ids).exists())

    print(f"\n  .fif bestanden gevonden : {len(fif_files)}")
    print(f"  Al verwerkt (skip)      : {n_skip}")
    print(f"  Te verwerken            : {len(fif_files) - n_skip}")
    print(f"  Workers                 : "
          f"{'alle cores' if n_jobs == -1 else n_jobs}\n")

    results = list(tqdm(
        Parallel(
            n_jobs    = n_jobs,
            backend   = "loky",
            return_as = "generator_unordered",
        )(delayed(_process_night)(f) for f in fif_files),
        total = len(fif_files),
        desc  = "Nachten",
        unit  = "nacht",
    ))

    log_df   = pd.DataFrame(results).sort_values(
        ["group", "subject_id", "night_id"]
    )
    log_path = EVENTS_DIR / "batch_log.csv"
    log_df.to_csv(log_path, index=False)

    # ── Eindoverzicht ─────────────────────────────────────────────────────────
    ok_df  = log_df[log_df["status"] == "ok"]
    n_ok   = len(ok_df)
    n_skip = (log_df["status"] == "skipped").sum()
    n_fail = (log_df["status"] == "failed").sum()
    mins   = (datetime.now() - t_start).total_seconds() / 60

    print(f"\n{'=' * 65}")
    print(f"  Succesvol     : {n_ok}")
    print(f"  Overgeslagen  : {n_skip}")
    print(f"  Mislukt       : {n_fail}")
    print(f"  Tijdsduur     : {mins:.1f} minuten")

    if n_ok > 0:
        total_events = pd.to_numeric(ok_df["n_events"], errors="coerce").sum()
        total_gate_a = pd.to_numeric(ok_df["gate_a_wake_onset"],
                                      errors="coerce").sum()
        total_gate_b = pd.to_numeric(ok_df["gate_b_post_wake"],
                                      errors="coerce").sum()
        total_gate_c = pd.to_numeric(ok_df["gate_c_rem_emg"],
                                      errors="coerce").sum()
        print(f"\n  Totaal geaccepteerde events : {int(total_events)}")
        print(f"  Verwijderd Gate A (wake onset)   : {int(total_gate_a)}")
        print(f"  Verwijderd Gate B (post-event)   : {int(total_gate_b)}")
        print(f"  Verwijderd Gate C (REM EMG)      : {int(total_gate_c)}")

    print(f"\n  Log     : {log_path}")
    print(f"  Output  : {EVENTS_DIR}")
    print(f"{'=' * 65}")

    if n_fail > 0:
        print(f"\n  [ERRORS — {n_fail} nachten mislukt]")
        for _, row in log_df[log_df["status"] == "failed"].iterrows():
            print(f"\n    {row['subject_id']} / {row['night_id']}")
            print(f"    {row['error'][:300]}")

    return log_df


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch candidate event pipeline (Fasen 5-8)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Verwerk alleen de eerste N nachten (voor testen)"
    )
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Aantal parallelle workers (-1 = alle cores)"
    )
    args = parser.parse_args()
    run_batch_events(limit=args.limit, n_jobs=args.jobs)