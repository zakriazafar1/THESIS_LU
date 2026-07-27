"""
=============================================================================
build_feature_matrix.py

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
EVENTS_DIR = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. feature matrices ridge")

TARGET_SFREQ = 128.0          # Hz, zelfde als in de detectiepipeline
NOTCH_HZ = 50.0
HIGHPASS_HZ = 0.1
LOWPASS_HZ = 35.0

EPOCH_SEC = 30.0               # standaard R&K/AASM epoch-lengte in het hypnogram (epoch 1 = t=0s)

EEG_CHANNELS = ["EEG L", "EEG R"]      # zoals de EDF-bestandsnamen heten
CHANNEL_LABELS = {"EEG L": "L", "EEG R": "R"}  # korte labels voor kolomnamen in de featurematrix
MOTION_CHANNELS = ["dX", "dY", "dZ"]
OXY_CHANNEL = "OXY_IR_AC"      # optioneel, wordt geladen indien aanwezig

BANDS = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}

# Ridge-extractie (Morlet CWT): dominante frequentie over tijd binnen het event
RIDGE_FREQ_MIN = 4.0    # Hz, ondergrens van het gescande frequentiebereik
RIDGE_FREQ_MAX = 30.0   # Hz, bovengrens
RIDGE_N_FREQS = 40      # aantal frequentiestappen tussen min en max
RIDGE_N_CYCLES = None   # None = variabele n_cycles (max(3, freq/2)), zelfde default als ScoringHero
RIDGE_PAD_SEC = 2.0     # padding aan weerszijden van het event, om FFT-wrap-around-randeffecten te beperken

# Vaste kolomvolgorde van de featurematrix (per band: L_ratio, L_peak_ratio, R_ratio, R_peak_ratio,
# dan de volgende band; daarna de mean_*_ratio's van alle banden naast elkaar).
FEATURE_COLUMN_ORDER = (
    ["subject_id", "group", "night_id", "stage_rk", "event_idx",
     "start_sec", "end_sec", "duration_sec", "sec_prev_event"]
    + [f"{label}_{band}_{metric}"
       for band in BANDS
       for label in CHANNEL_LABELS.values()
       for metric in ("ratio", "peak_ratio")]
    + [f"mean_{band}_ratio" for band in BANDS]
    + ["ridge_onset_hz", "ridge_peak_hz", "ridge_end_hz", "ridge_drift"]
    + ["motion_rms", "oxy_amp_ratio"]
)

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
    Laadt het hypnogram. Vast formaat, GEEN header:
      kolom 1 = R&K-stage per epoch (0=wake, 1, 2, 3, 5=REM)
      kolom 2 = ongebruikt (altijd 0 in de bestanden die we tot nu toe zagen)
    Rij N (1-based) = epoch N; epoch 1 begint bij t=0s van de opname, en elke
    epoch duurt EPOCH_SEC seconden (standaard R&K/AASM-epochlengte, 30s).
    Geeft None terug als het bestand niet gevonden of niet leesbaar is.
    """
    if path is None:
        return None
    try:
        df = pd.read_csv(path, header=None, sep=None, engine="python")
    except Exception:
        return None

    if df.shape[1] < 1 or len(df) == 0:
        return None

    stage = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    onset_sec = np.arange(len(df)) * EPOCH_SEC
    return pd.DataFrame({"onset_sec": onset_sec, "stage_rk": stage.values})


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
            data = data - np.median(data)  # DC-removal: anders meet motion_rms grotendeels sensor-offset
            signals[ch] = _resample_scipy(data, sfreq)

    loaded = load_channel(edf_dir, OXY_CHANNEL)
    if loaded is not None:
        data, sfreq = loaded
        signals[OXY_CHANNEL] = _resample_scipy(data, sfreq)

    return signals


# =============================================================================
# SECTIE 4 — FEATURES PER EVENT BEREKENEN
# =============================================================================

def compute_morlet_tf(signal, srate, freqs, n_cycles=None, L2normalize=False):
    """Compute time-frequency power via Morlet wavelet convolution (FFT-based).
    For each frequency the signal is convolved with a complex Morlet wavelet
    whose bandwidth is set by n_cycles. The FFT of the signal is computed
    once and then multiplied by each wavelet's frequency-domain Gaussian,
    making this O(F * N log N) in total.

    (Overgenomen ongewijzigd van de supervisor's ScoringHero-implementatie,
    zodat de ridge-features hier dezelfde tijd-frequentie-methode gebruiken
    als de tool waarmee Lucija scoort.)

    Parameters
    ----------
    signal : 1-D ndarray, shape (n_samples,)
        EEG signal for one epoch (may include extension on either side).
    srate : float
        Sampling rate in Hz.
    freqs : 1-D ndarray
        Centre frequencies in Hz.
    n_cycles : float, array-like, or None
        Number of wavelet cycles per frequency.  When None (default) a
        variable scheme is used: n_cycles = max(3, freq / 2).  This gives
        better temporal resolution at low frequencies (individual slow
        waves are sharply localised) while preserving good frequency
        resolution at higher frequencies (spindles, beta, gamma).
        Pass a scalar to use the same value for every frequency.
    normalize : bool
        If True, L2-normalize each wavelet so it has unit energy. This
        makes power values comparable across frequencies. If False, the
        raw (unnormalized) power is returned.

    Returns
    -------
    power : 2-D ndarray, shape (n_freqs, n_samples)
        Instantaneous power (squared magnitude of the analytic signal).
    """
    freqs = np.asarray(freqs)
    if n_cycles is None:
        n_cycles_arr = np.maximum(3.0, freqs / 2.0)
    elif np.isscalar(n_cycles):
        n_cycles_arr = np.full(len(freqs), float(n_cycles))
    else:
        n_cycles_arr = np.asarray(n_cycles, dtype=float)
    n_samples = len(signal)
    signal = signal - np.mean(signal)  # remove DC offset to avoid leakage into low-frequency wavelets
    signal_fft = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(n_samples, d=1.0 / srate)
    power = np.empty((len(freqs), n_samples), dtype=np.float64)
    for i, freq in enumerate(freqs):
        sigma_f = freq / n_cycles_arr[i]
        wavelet_fft = np.exp(-0.5 * ((fft_freqs - freq) / sigma_f) ** 2)
        if L2normalize:
            wavelet_fft /= np.sqrt(np.sum(wavelet_fft ** 2))
        analytic = np.fft.ifft(signal_fft * wavelet_fft)
        power[i] = np.abs(analytic) ** 2
    return power


# Minimale segmentlengte (samples) om band_envelope veilig te kunnen draaien.
# filtfilt vereist een signaal langer dan zijn interne padding-lengte;
# voor onze 4e-orde bandpass-filters is dat orde ~27 samples, dus 64 is een veilige marge.
MIN_FILTER_SAMPLES = 64


def compute_night_band_envelopes(signals: dict) -> dict:
    """
    Berekent voor elk EEG-kanaal en elke band de band-envelope over de HELE nacht,
    één keer per nacht. Wordt hergebruikt voor zowel de whole-night baseline
    (mediaan) als de during-event statistieken (via slicing), zodat er niet
    per event opnieuw gefilterd hoeft te worden.

    Key: (kanaal, band_naam) -> envelope-array (zelfde lengte als signals[kanaal])
    """
    envelopes = {}
    for ch in EEG_CHANNELS:
        if ch not in signals:
            continue
        sig = signals[ch]
        for band_name, band_range in BANDS.items():
            envelopes[(ch, band_name)] = band_envelope(sig, TARGET_SFREQ, band_range)
    return envelopes


def compute_night_baselines(night_envelopes: dict) -> dict:
    """
    Mediaan van elke band-envelope over de HELE nacht -> baseline-referentie
    per (kanaal, band). Dit vervangt de vroegere lokale 90s-baseline vóór elk
    event: alle events in dezelfde nacht worden nu vergeleken t.o.v. hetzelfde,
    stabiele referentiepunt i.p.v. een steeds wisselend lokaal venster.
    """
    return {key: (np.median(env) if len(env) else np.nan) for key, env in night_envelopes.items()}


def compute_ridge_features(sig: np.ndarray, sf: float, start_i: int, end_i: int) -> dict:
    """
    Morlet-tijd-frequentie-analyse + ridge-extractie (via compute_morlet_tf,
    dezelfde methode als ScoringHero): volgt de dominante frequentie op elk
    moment binnen het event, i.p.v. energie in vooraf-vaste banden (zoals de
    band_envelope-features). Geeft frequentie bij onset/piek/einde, de
    drift (Hz/sec) over het event, en de spreiding van de ridge (een ruwe
    maat voor hoe smalbandig/breedbandig het event is).

    compute_morlet_tf werkt via een FFT over het hele meegegeven signaal,
    wat op een kort event-segment tot wrap-around-randeffecten kan leiden
    (het einde "lekt" een beetje naar het begin en andersom). Om dat te
    beperken wordt het event met RIDGE_PAD_SEC seconden padding aan beide
    kanten meegenomen in de berekening, en pas daarna teruggesneden naar
    het eigenlijke event-window.
    """
    empty = {
        "ridge_onset_hz": np.nan,
        "ridge_peak_hz": np.nan,
        "ridge_end_hz": np.nan,
        "ridge_drift": np.nan,
    }

    if end_i <= start_i or end_i > len(sig):
        return empty

    pad_samples = int(RIDGE_PAD_SEC * sf)
    seg_start = max(0, start_i - pad_samples)
    seg_end = min(len(sig), end_i + pad_samples)
    padded_segment = sig[seg_start:seg_end]

    if len(padded_segment) < MIN_FILTER_SAMPLES:
        return empty

    freqs = np.linspace(RIDGE_FREQ_MIN, RIDGE_FREQ_MAX, RIDGE_N_FREQS)
    power_padded = compute_morlet_tf(padded_segment, sf, freqs, n_cycles=RIDGE_N_CYCLES)

    # Terugsnijden naar het eigenlijke event-window (padding eraf)
    crop_start = start_i - seg_start
    crop_end = crop_start + (end_i - start_i)
    power = power_padded[:, crop_start:crop_end]

    if power.shape[1] == 0:
        return empty

    ridge_idx = np.argmax(power, axis=0)
    ridge_freq = freqs[ridge_idx]

    n_t = len(ridge_freq)
    duration = n_t / sf

    onset_freq = ridge_freq[0]
    end_freq = ridge_freq[-1]
    drift = (end_freq - onset_freq) / duration if duration > 0 else np.nan

    peak_time_idx = np.unravel_index(np.argmax(power), power.shape)[1]
    peak_freq = ridge_freq[peak_time_idx]

    return {
        "ridge_onset_hz": onset_freq,
        "ridge_peak_hz": peak_freq,
        "ridge_end_hz": end_freq,
        "ridge_drift": drift,
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    """Deelt twee waarden, met NaN/0-bescherming."""
    if numerator is None or denominator is None:
        return np.nan
    if np.isnan(numerator) or np.isnan(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def extract_event_features(signals: dict, start_sec: float, end_sec: float,
                            night_envelopes: dict, night_baselines: dict,
                            oxy_night_std: float | None) -> dict:
    """
    Berekent features voor één event, gebaseerd op de EEG-signalen.

    Band-ratio's: tijdens-event gemiddelde/piek gedeeld door de MEDIAAN VAN
    DE HELE NACHT (night_baselines) voor dat kanaal+band, in plaats van een
    lokale 90s-baseline vóór het event. Alle events in dezelfde nacht worden
    zo tegen hetzelfde, stabiele referentiepunt afgezet:

        {band}_ratio = tijdens-event gemiddelde amplitude / mediane amplitude over de hele nacht

    De tijdens-event amplitude-waarden worden gesliced uit de al voor de hele
    nacht berekende band-envelope (night_envelopes), i.p.v. opnieuw een
    bandpass-filter op het korte event-segment te draaien — dat voorkomt
    filter-randeffecten op korte segmenten én is sneller.

    oxy_amp_ratio gebruikt dezelfde whole-night-logica: tijdens-event std
    van het OXY_IR_AC-signaal gedeeld door de std over de HELE nacht
    (oxy_night_std, één keer per nacht berekend), i.p.v. een lokale
    90s-baseline.
    """
    feats = {}
    sf = TARGET_SFREQ

    start_i = int(start_sec * sf)
    end_i = int(end_sec * sf)

    for ch in EEG_CHANNELS:
        label = CHANNEL_LABELS[ch]
        if ch not in signals:
            continue
        sig = signals[ch]
        if end_i > len(sig) or start_i >= end_i:
            continue

        for band_name, band_range in BANDS.items():
            env_full = night_envelopes.get((ch, band_name))
            baseline_med = night_baselines.get((ch, band_name), np.nan)

            if env_full is not None and end_i <= len(env_full):
                env_event = env_full[start_i:end_i]
                during_mean = np.mean(env_event) if len(env_event) else np.nan
                during_peak = np.max(env_event) if len(env_event) else np.nan
            else:
                during_mean = during_peak = np.nan

            feats[f"{label}_{band_name}_ratio"] = safe_ratio(during_mean, baseline_med)
            feats[f"{label}_{band_name}_peak_ratio"] = safe_ratio(during_peak, baseline_med)

    # Gemiddelde over kanalen (voor als er maar 1 kanaal beschikbaar is, of ter samenvatting)
    for band_name in BANDS:
        ratios = [feats.get(f"{CHANNEL_LABELS[ch]}_{band_name}_ratio") for ch in EEG_CHANNELS
                  if f"{CHANNEL_LABELS[ch]}_{band_name}_ratio" in feats]
        ratios = [r for r in ratios if r is not None and not (isinstance(r, float) and np.isnan(r))]
        feats[f"mean_{band_name}_ratio"] = np.mean(ratios) if ratios else np.nan

    # Ridge-features (Morlet CWT): dominante frequentie over tijd binnen het event
    if "EEG L" in signals:
        sig = signals["EEG L"]
        ridge_feats = compute_ridge_features(sig, sf, start_i, end_i)
        feats.update(ridge_feats)
    else:
        feats.update({
            "ridge_onset_hz": np.nan,
            "ridge_peak_hz": np.nan,
            "ridge_end_hz": np.nan,
            "ridge_drift": np.nan,
        })

    # Motion features (accelerometer), als proxy voor beweging tijdens het event
    motion_rms = []
    for ch in MOTION_CHANNELS:
        if ch in signals:
            sig = signals[ch]
            if end_i <= len(sig) and start_i < end_i:
                seg = sig[start_i:end_i]
                motion_rms.append(np.sqrt(np.mean(seg ** 2)))
    feats["motion_rms"] = np.mean(motion_rms) if motion_rms else np.nan

    # Pulse-oximetrie amplitude-ratio (cardiovasculaire arousal proxy): tijdens-event std
    # gedeeld door de std over de HELE nacht, indien het OXY-kanaal beschikbaar is.
    if OXY_CHANNEL in signals and oxy_night_std is not None:
        sig = signals[OXY_CHANNEL]
        if end_i <= len(sig) and start_i < end_i:
            event_std = np.std(sig[start_i:end_i])
            feats["oxy_amp_ratio"] = safe_ratio(event_std, oxy_night_std)
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

    # Whole-night band-envelopes en -baselines: één keer per nacht berekenen,
    # daarna hergebruiken voor elk event (i.p.v. per event opnieuw filteren).
    night_envelopes = compute_night_band_envelopes(signals)
    night_baselines = compute_night_baselines(night_envelopes)
    oxy_night_std = np.std(signals[OXY_CHANNEL]) if OXY_CHANNEL in signals else None

    rows = []
    prev_end = None
    for i, ev in events.iterrows():
        feats = extract_event_features(signals, ev["start_sec"], ev["end_sec"],
                                        night_envelopes, night_baselines, oxy_night_std)
        feats.update({
            "subject_id": ids["subject_id"],
            "group": ids["group"],
            "night_id": ids["night_id"],
            "event_idx": i,
            "start_sec": ev["start_sec"],
            "end_sec": ev["end_sec"],
            "duration_sec": ev["duration_sec"],
            "stage_rk": get_stage_at(hypnogram, ev["start_sec"]),
            "sec_prev_event": (ev["start_sec"] - prev_end) if prev_end is not None else np.nan,
        })
        rows.append(feats)
        prev_end = ev["end_sec"]

    df = pd.DataFrame(rows)
    return df[[c for c in FEATURE_COLUMN_ORDER if c in df.columns]]


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

            # Per-nacht featurematrix apart opslaan (standaard CSV: komma-scheiding, punt-decimaal —
            # NIET meer Excel-NL-formaat, want dat corrumpeert bij openen/opslaan met een niet-NL
            # Excel-locale-instelling: decimale komma's worden dan als duizendtal-scheiding gelezen)
            EVENTS_DIR.mkdir(parents=True, exist_ok=True)
            night_out_path = EVENTS_DIR / f"{ids['stem']}_fm.csv"
            df.to_csv(night_out_path, index=False, float_format="%.3f")

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
    feature_matrix.to_csv(out_path, index=False, float_format="%.3f")
    print(f"\nFeaturematrix opgeslagen: {out_path}")
    print(f"Shape: {feature_matrix.shape}")


if __name__ == "__main__":
    main()
