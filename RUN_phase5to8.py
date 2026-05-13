# =============================================================================
# PHASE 9 — Batch candidate event pipeline (Fasen 5 t/m 8)
#
# Verwerkt alle preprocessed nachten (.fif) parallel:
#   - Fase 5: band-envelopes (theta / alpha / sigma / beta)
#   - Fase 6: lokale baseline + ratio
#   - Fase 7: candidate activation mask
#   - Fase 8: event boundaries + DataFrame
#
# Output per nacht:
#   EVENTS_DIR / GROUP / subject_id / candidate_events_{subject_id}_{night_id}.csv
#
# Bestaande CSV-bestanden worden overgeslagen (idempotent).
#
# Gebruik:
#   uv run phase9_batch_events.py            # volledige batch
#   uv run phase9_batch_events.py --limit 3  # eerst testen op 3 nachten
# =============================================================================

import io
import contextlib
import traceback
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# ── Gedeelde constanten (zelfde als notebook) ─────────────────────────────────
EEG_CH = ["EEG L psg-lp", "EEG R psg-lp"]
EMG_CH = ["EEG L psg-emg", "EEG R psg-emg"]
MOV_CH = ["dX", "dY", "dZ"]
ALL_CH = EEG_CH + EMG_CH + MOV_CH

TARGET_SFREQ         = 128
MOVEMENT_THRESHOLD_UV = 1000.0

CH_F7 = "EEG L psg-lp"
CH_F8 = "EEG R psg-lp"

BANDS = {
    "theta": (4.0,  7.0),
    "alpha": (8.0,  12.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}

SMOOTH_SEC           = 0.5
BASELINE_SEC         = 30.0
ACTIVATION_THRESHOLD = 2.0
SPINDLE_THRESHOLD    = 2.0
MERGE_GAP_SEC        = 1.0
MIN_DUR_SEC          = 1.0
MAX_DUR_SEC          = 15.0

# ── Mappen ────────────────────────────────────────────────────────────────────
PREP_DIR   = Path(r"C:\Users\zafar\Documents\THESIS_OUTPUTS\1_preprocessing_EEG")
EVENTS_DIR = Path(r"C:\Users\zafar\Documents\THESIS_OUTPUTS\2_candidate_events")
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

N_JOBS = -1  # -1 = alle beschikbare CPU-cores


# =============================================================================
# Fase 1-hulpfuncties (gekopieerd uit notebook zodat dit script standalone draait)
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


def remove_movement_artifacts(signals: dict, sfreq: int = TARGET_SFREQ) -> tuple:
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
            mask_dilated[max(0, idx - buffer):min(n_samples, idx + buffer)] = True
        stats[ch]     = round(mask_dilated.mean() * 100, 2)
        combined_mask |= mask_dilated
    return combined_mask, stats


# =============================================================================
# Fase 5 — Band-envelopes
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
    from scipy.signal import hilbert
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(hilbert(signal)).astype(np.float32)


def smooth_envelope(envelope: np.ndarray, smooth_sec: float = SMOOTH_SEC,
                    sfreq: float = TARGET_SFREQ) -> np.ndarray:
    window = int(smooth_sec * sfreq)
    kernel = np.ones(window) / window
    return np.convolve(envelope, kernel, mode="same").astype(np.float32)


def compute_band_envelopes(signals: dict, sfreq: float = TARGET_SFREQ) -> dict:
    time_axis = np.arange(len(signals[CH_F7])) / sfreq
    result    = {}
    for ch_label, ch_name in [("F7", CH_F7), ("F8", CH_F8)]:
        if ch_name not in signals:
            continue
        sig  = signals[ch_name]
        rows = {"time": time_axis}
        for band_name, (lo, hi) in BANDS.items():
            filtered       = bandpass_filter(sig, lo, hi, sfreq)
            envelope       = compute_envelope(filtered)
            rows[band_name] = smooth_envelope(envelope, SMOOTH_SEC, sfreq)
        result[ch_label] = pd.DataFrame(rows)
    return result


# =============================================================================
# Fase 6 — Lokale baseline + ratio
# =============================================================================

def compute_rolling_baseline(envelopes: dict, movement_mask: np.ndarray = None,
                              sfreq: float = TARGET_SFREQ,
                              baseline_sec: float = BASELINE_SEC) -> dict:
    window    = int(baseline_sec * sfreq)
    baselines = {}
    for ch_label, df in envelopes.items():
        rows = {"time": df["time"].values}
        for band in BANDS:
            envelope = df[band].copy().astype(np.float64)
            if movement_mask is not None:
                envelope[movement_mask] = np.nan
            series   = pd.Series(envelope)
            baseline = series.rolling(window=window, min_periods=1).median()
            baseline = baseline.ffill().bfill()
            rows[band] = baseline.values.astype(np.float32)
        baselines[ch_label] = pd.DataFrame(rows)
    return baselines


def compute_ratio(envelopes: dict, baselines: dict) -> dict:
    ratios = {}
    for ch_label in envelopes:
        env_df  = envelopes[ch_label]
        base_df = baselines[ch_label]
        rows    = {"time": env_df["time"].values}
        for band in BANDS:
            rows[band] = (env_df[band].values / (base_df[band].values + 1e-6)).astype(np.float32)
        ratios[ch_label] = pd.DataFrame(rows)
    return ratios


# =============================================================================
# Fase 7 — Candidate activation mask
# =============================================================================

def compute_activation_mask(ratios: dict, threshold: float = ACTIVATION_THRESHOLD) -> dict:
    masks = {}
    for ch_label, df in ratios.items():
        masks[ch_label] = (df["alpha"].values > threshold) | (df["beta"].values > threshold)
    return masks


def combine_channels(masks: dict) -> dict:
    F7 = masks["F7"]
    F8 = masks["F8"]
    return {
        "F7_only":   F7 & ~F8,
        "F8_only":   F8 & ~F7,
        "bilateral": F7 & F8,
        "combined":  F7 | F8,
    }


def compute_spindle_score(ratios: dict, envelopes: dict) -> np.ndarray:
    scores = []
    for ch_label in ["F7", "F8"]:
        env = envelopes[ch_label]
        rat = ratios[ch_label]
        c1  = (env["sigma"].values > env["theta"].values).astype(np.float32)
        c2  = (env["sigma"].values > env["alpha"].values).astype(np.float32)
        c3  = (env["sigma"].values > env["beta"].values ).astype(np.float32)
        c4  = (rat["sigma"].values > SPINDLE_THRESHOLD  ).astype(np.float32)
        scores.append((c1 + c2 + c3 + c4) / 4.0)
    return np.mean(scores, axis=0).astype(np.float32)


def run_fase7(ratios: dict, envelopes: dict,
              movement_mask: np.ndarray = None) -> dict:
    activation_masks = compute_activation_mask(ratios)
    channel_masks    = combine_channels(activation_masks)
    spindle_score    = compute_spindle_score(ratios, envelopes)

    candidate_mask = channel_masks["combined"]
    if movement_mask is not None:
        candidate_mask = candidate_mask & ~movement_mask

    return {
        "candidate_mask": candidate_mask,
        "channel_masks":  channel_masks,
        "spindle_score":  spindle_score,
        "time":           ratios["F7"]["time"].values,
    }


# =============================================================================
# Fase 8 — Event boundaries
# =============================================================================

def mask_to_events(candidate_mask: np.ndarray) -> list:
    events   = []
    in_event = False
    start    = 0
    for i, active in enumerate(candidate_mask):
        if active and not in_event:
            start    = i
            in_event = True
        elif not active and in_event:
            in_event = False
            events.append((start, i))
    if in_event:
        events.append((start, len(candidate_mask)))
    return events


def merge_events(events: list, sfreq: float = TARGET_SFREQ) -> list:
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


def apply_duration_filter(events: list, sfreq: float = TARGET_SFREQ) -> list:
    min_samp = int(MIN_DUR_SEC * sfreq)
    max_samp = int(MAX_DUR_SEC * sfreq)
    return [(s, e) for s, e in events if min_samp <= (e - s) <= max_samp]


def events_to_dataframe(events: list, channel_masks: dict,
                         spindle_score: np.ndarray,
                         movement_mask: np.ndarray = None,
                         sfreq: float = TARGET_SFREQ) -> pd.DataFrame:
    rows = []
    for start, end in events:
        dur_sec   = (end - start) / sfreq
        is_short  = 1.0 <= dur_sec <= 3.0
        is_arousal = 3.01 <= dur_sec <= 15.0
        cat = "both" if (is_short and is_arousal) else ("short" if is_short else "arousal")

        bilat_seg         = channel_masks["bilateral"][start:end]
        bilateral_overlap = float(bilat_seg.mean())

        rows.append({
            "start_sample":         start,
            "end_sample":           end,
            "start_sec":            round(start / sfreq, 4),
            "end_sec":              round(end   / sfreq, 4),
            "duration_sec":         round(dur_sec, 4),
            "duration_category":    cat,
            "F7_active":            bool((channel_masks["F7_only"][start:end].any()) or bilat_seg.any()),
            "F8_active":            bool((channel_masks["F8_only"][start:end].any()) or bilat_seg.any()),
            "F7_only":              bool(channel_masks["F7_only"][start:end].mean() > 0.5),
            "F8_only":              bool(channel_masks["F8_only"][start:end].mean() > 0.5),
            "bilateral":            bool(bilateral_overlap > 0.1),
            "bilateral_overlap":    round(bilateral_overlap, 4),
            "spindle_score_mean":   round(float(spindle_score[start:end].mean()), 4),
            "spindle_score_max":    round(float(spindle_score[start:end].max()),  4),
            "artifact_overlap_pct": round(float(movement_mask[start:end].mean() * 100), 2)
                                    if movement_mask is not None else 0.0,
        })
    return pd.DataFrame(rows)


def run_fase8(fase7_output: dict, movement_mask: np.ndarray = None) -> pd.DataFrame:
    candidate_mask = fase7_output["candidate_mask"]
    channel_masks  = fase7_output["channel_masks"]
    spindle_score  = fase7_output["spindle_score"]

    events = mask_to_events(candidate_mask)
    events = merge_events(events)
    events = apply_duration_filter(events)

    return events_to_dataframe(events, channel_masks, spindle_score, movement_mask)


# =============================================================================
# Pad-helpers
# =============================================================================

def _ids_from_fif(fif_path: Path) -> dict:
    """Parse subject_id / night_id / group uit de preprocessed FIF-bestandsnaam."""
    stem = fif_path.stem.replace("_prep_raw", "")  # bnbd_nsr_01272_T0_N1_psg
    return extract_ids(Path(stem))


def _csv_path(ids: dict) -> Path:
    """
    Pad naar de output CSV voor één nacht.
    Structuur: EVENTS_DIR / GROUP / subject_id / candidate_events_{subject_id}_{night_id}.csv
    """
    return (
        EVENTS_DIR
        / ids["group"]
        / ids["subject_id"]
        / f"candidate_events_{ids['subject_id']}_{ids['night_id']}.csv"
    )


def _find_fif_files() -> list:
    """Zoek alle preprocessed .fif bestanden, gesorteerd op pad."""
    return sorted(PREP_DIR.rglob("*_psg_prep_raw.fif"))


# =============================================================================
# Verwerking van één nacht (draait in worker-process)
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
        "group":      ids["group"],
        "subject_id": ids["subject_id"],
        "night_id":   ids["night_id"],
        "fif_path":   str(fif_path),
        "output_csv": str(out_csv),
        "status":     "failed",
        "n_events":   "",
        "error":      "",
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Skip als output al bestaat — lees info uit bestaande CSV ──
    if out_csv.exists():
        try:
            existing      = pd.read_csv(out_csv)
            log["n_events"] = len(existing)
        except Exception:
            log["n_events"] = ""
        log["status"] = "skipped"
        return log

    try:
        with contextlib.redirect_stdout(io.StringIO()):

            # Fase 5: band-envelopes
            raw       = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
            signals   = preprocess_signals(raw)
            envelopes = compute_band_envelopes(signals)

            # Fase 6: lokale baseline + ratio (artefact-bewust)
            movement_mask, _ = remove_movement_artifacts(signals)
            baselines         = compute_rolling_baseline(envelopes, movement_mask)
            ratios            = compute_ratio(envelopes, baselines)

            # Fase 7: candidate activation mask
            fase7 = run_fase7(
                ratios        = ratios,
                envelopes     = envelopes,
                movement_mask = movement_mask,
            )

            # Fase 8: event boundaries -> DataFrame
            df = run_fase8(fase7, movement_mask=movement_mask)

        # ── Voeg identifiers toe als eerste kolommen ──────────────
        df.insert(0, "night_id",   ids["night_id"])
        df.insert(0, "subject_id", ids["subject_id"])

        # ── Opslaan ───────────────────────────────────────────────
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        log["status"]   = "ok"
        log["n_events"] = len(df)

    except Exception:
        log["error"] = traceback.format_exc()

    return log


# =============================================================================
# Batch runner
# =============================================================================

def run_batch_events(limit: int = None, n_jobs: int = N_JOBS) -> pd.DataFrame:
    """
    Verwerk alle preprocessed nachten parallel (Fasen 5-8).

    Parameters
    ----------
    limit  : int, optioneel  - verwerk alleen de eerste N nachten (voor testen)
    n_jobs : int             - parallelle workers; -1 = alle CPU-cores

    Returns
    -------
    pd.DataFrame met verwerkingslog (status, n_events, fout per nacht)
    """
    t_start = datetime.now()
    print("=" * 65)
    print("  CANDIDATE EVENT BATCH  |  Fasen 5 t/m 8")
    print(f"  Start : {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Bestanden verzamelen ──────────────────────────────────────
    fif_files = _find_fif_files()
    if limit:
        fif_files = fif_files[:limit]

    ids_list = [_ids_from_fif(f) for f in fif_files]
    n_skip   = sum(1 for ids in ids_list if _csv_path(ids).exists())

    print(f"\n  .fif bestanden gevonden : {len(fif_files)}")
    print(f"  Al verwerkt (skip)      : {n_skip}")
    print(f"  Te verwerken            : {len(fif_files) - n_skip}")
    print(f"  Workers                 : {'alle cores' if n_jobs == -1 else n_jobs}\n")

    # ── Parallel verwerken met real-time voortgangsbalk ──────────
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

    # ── Log opslaan ───────────────────────────────────────────────
    log_df   = pd.DataFrame(results).sort_values(["group", "subject_id", "night_id"])
    log_path = EVENTS_DIR / "batch_log.csv"
    log_df.to_csv(log_path, index=False)

    # ── Eindoverzicht ─────────────────────────────────────────────
    n_ok   = (log_df["status"] == "ok").sum()
    n_skip = (log_df["status"] == "skipped").sum()
    n_fail = (log_df["status"] == "failed").sum()
    mins   = (datetime.now() - t_start).total_seconds() / 60

    print(f"\n{'=' * 65}")
    print(f"  Succesvol     : {n_ok}")
    print(f"  Overgeslagen  : {n_skip}")
    print(f"  Mislukt       : {n_fail}")
    print(f"  Tijdsduur     : {mins:.1f} minuten")
    print(f"  Log           : {log_path}")
    print(f"  Output map    : {EVENTS_DIR}")
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
    parser = argparse.ArgumentParser(description="Batch candidate event pipeline (Fasen 5-8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Verwerk alleen de eerste N nachten (voor testen)")
    parser.add_argument("--jobs",  type=int, default=-1,
                        help="Aantal parallelle workers (-1 = alle cores)")
    args, _ = parser.parse_known_args()  # parse_known_args negeert Jupyter kernel-argumenten

    run_batch_events(limit=args.limit, n_jobs=args.jobs)
