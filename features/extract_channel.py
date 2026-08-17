"""
=============================================================================
extract_channel_matrix.py

Extraheert kanaalwaardes PER TIJDSTIP rond een specifiek arousal-event, voor
alle kanalen die ook in feature_matrix.py gebruikt worden (EEG L, EEG R, dX,
dY, dZ, OXY_IR_AC). Handig om te controleren waarom een feature (bv.
motion_rms) een bepaalde waarde krijgt -- je ziet de daadwerkelijke tijdreeks
i.p.v. alleen het eindresultaat.

BELANGRIJK: dit script hergebruikt de laad- en preprocessing-functies UIT
feature_matrix.py (get_edf_dir, load_channel, preprocess_eeg, _resample_scipy,
parse_ids, find_events_file, load_events). Daardoor zijn de waardes die je
hier ziet gegarandeerd hetzelfde als wat er in de featurematrix belandt --
er is geen aparte, mogelijk afwijkende implementatie.

Vereist dat feature_matrix.py in dezelfde map staat (of op de PYTHONPATH).

Gebruik:
  # eerste event van een nacht, met 5s padding ervoor/erna, alleen gepreprocest
  python extract_channel_matrix.py --night-dir "<pad naar nacht-map>"

  # specifiek event 3, met 10s padding, INCLUSIEF ruwe (ongefilterde) kanalen
  python extract_channel_matrix.py --night-dir "<pad>" --event-idx 3 --pad-sec 10 --raw

  # de eerste 10 events van een nacht, elk als los CSV-bestand
  python extract_channel_matrix.py --night-dir "<pad>" --limit-events 10

Elke CSV heeft een 'time_sec'-kolom en daarna 1 kolom per gepreprocest kanaal
(EEG L, EEG R, dX, dY, dZ, OXY_IR_AC -- welke dan ook geladen konden worden),
plus (met --raw) een *_raw kolom per kanaal met de ongefilterde EDF-waardes op
hun eigen oorspronkelijke samplerate.

Na het wegschrijven print het script ook meteen een samenvatting: de RMS per
motion-as in dit venster, en de gecombineerde vectormagnitude-RMS (= exact de
motion_rms-waarde die ook in de featurematrix terechtkomt voor dit event) --
zodat je meteen ziet hoe die ene waarde uit de tijdreeks is opgebouwd.
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import feature_matrix as fm

ALL_CHANNELS = fm.EEG_CHANNELS + fm.MOTION_CHANNELS + [fm.OXY_CHANNEL]


def load_all_channels(night_dir: Path, stem: str, include_raw: bool = False):
    """
    Laadt alle kanalen die ook feature_matrix.py gebruikt, met DEZELFDE
    preprocessing-functies (dus 1-op-1 wat er de featureberekening ingaat).

    Geeft twee dicts terug:
      processed[kanaalnaam]     -> 1D array, gepreprocest + geresampled naar TARGET_SFREQ
      raw_out[kanaalnaam]       -> (1D array, oorspronkelijke sfreq), alleen gevuld als include_raw=True
    """
    edf_dir = fm.get_edf_dir(night_dir, stem)
    processed = {}
    raw_out = {}

    for ch in fm.EEG_CHANNELS:
        loaded = fm.load_channel(edf_dir, ch)
        if loaded is None:
            continue
        data, sfreq = loaded
        if include_raw:
            raw_out[ch] = (data, sfreq)
        processed[ch] = fm.preprocess_eeg(data, sfreq)

    for ch in fm.MOTION_CHANNELS:
        loaded = fm.load_channel(edf_dir, ch)
        if loaded is None:
            continue
        data, sfreq = loaded
        if include_raw:
            raw_out[ch] = (data, sfreq)
        # zelfde DC-removal + resample als in load_night_signals (feature_matrix.py)
        d = data - np.median(data)
        processed[ch] = fm._resample_scipy(d, sfreq)

    loaded = fm.load_channel(edf_dir, fm.OXY_CHANNEL)
    if loaded is not None:
        data, sfreq = loaded
        if include_raw:
            raw_out[fm.OXY_CHANNEL] = (data, sfreq)
        processed[fm.OXY_CHANNEL] = fm._resample_scipy(data, sfreq)

    return processed, raw_out


def build_matrix_for_window(processed: dict, raw_out: dict,
                             start_sec: float, end_sec: float, pad_sec: float = 0.0) -> pd.DataFrame:
    """
    Bouwt een DataFrame met 1 rij per sample (op TARGET_SFREQ=128Hz) en 1 kolom
    per gepreprocest kanaal, voor het venster [start_sec - pad_sec, end_sec + pad_sec].

    Ruwe kanalen (indien raw_out gevuld is) worden er als aparte *_raw kolommen
    bijgezet via een tijd-uitlijning (merge_asof), omdat ze op hun eigen,
    mogelijk andere, oorspronkelijke samplerate staan.
    """
    win_start = max(0.0, start_sec - pad_sec)
    win_end = end_sec + pad_sec

    sf = fm.TARGET_SFREQ
    start_i = int(win_start * sf)
    end_i = int(win_end * sf)
    n_samples = max(0, end_i - start_i)

    t = (start_i + np.arange(n_samples)) / sf
    df = pd.DataFrame({"time_sec": t})

    for ch, sig in processed.items():
        if end_i <= len(sig):
            seg = sig[start_i:end_i]
        elif start_i < len(sig):
            seg = sig[start_i:len(sig)]
            seg = np.concatenate([seg, np.full(n_samples - len(seg), np.nan)])
        else:
            seg = np.full(n_samples, np.nan)
        df[ch] = seg

    for ch, (data, sfreq) in raw_out.items():
        r_start_i = int(win_start * sfreq)
        r_end_i = int(win_end * sfreq)
        seg = data[r_start_i:r_end_i]
        if len(seg) == 0:
            continue
        t_raw = (r_start_i + np.arange(len(seg))) / sfreq
        df_raw = pd.DataFrame({"time_sec": t_raw, f"{ch}_raw": seg})
        df = pd.merge_asof(df.sort_values("time_sec"), df_raw.sort_values("time_sec"),
                            on="time_sec", direction="nearest")

    return df


def summarize_motion(df: pd.DataFrame, channels=tuple(fm.MOTION_CHANNELS)) -> None:
    """
    Print de RMS per motion-as in dit venster, plus de gecombineerde
    vectormagnitude-RMS -- dat laatste getal is exact motion_rms zoals het
    in de featurematrix terechtkomt voor dit event (zie extract_event_features
    in feature_matrix.py).
    """
    present = [c for c in channels if c in df.columns]
    if not present:
        print("  (geen motion-kanalen in dit venster)")
        return

    print("  Per-as RMS in dit venster:")
    for c in present:
        vals = df[c].dropna().values
        if len(vals) == 0:
            print(f"    {c}: geen data in dit venster")
            continue
        rms = np.sqrt(np.mean(vals ** 2))
        print(f"    {c}: RMS={rms:.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    magnitude = np.sqrt(sum(df[c].values ** 2 for c in present))
    combined_rms = np.sqrt(np.nanmean(magnitude ** 2))
    print(f"  --> Gecombineerde vectormagnitude-RMS (= motion_rms in de featurematrix): {combined_rms:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--night-dir", type=Path, required=True,
                         help="Pad naar de nacht-map, bv. .../bnbd_nsr_01272_T0_N2")
    parser.add_argument("--event-idx", type=int, default=None,
                         help="Index van het specifieke event binnen deze nacht (0-based)")
    parser.add_argument("--limit-events", type=int, default=None,
                         help="I.p.v. 1 event: de eerste N events exporteren, elk als los bestand")
    parser.add_argument("--pad-sec", type=float, default=5.0,
                         help="Extra seconden voor/na het event meenemen (default: 5s)")
    parser.add_argument("--raw", action="store_true",
                         help="Ook de ruwe, ongefilterde kanaalwaardes meenemen (*_raw kolommen)")
    parser.add_argument("--out-dir", type=Path, default=Path("./channel_matrices"),
                         help="Map om de CSV's in weg te schrijven (default: ./channel_matrices)")
    args = parser.parse_args()

    ids = fm.parse_ids(args.night_dir)
    arch_dir = args.night_dir / "sleepArchitecture"
    events_path = fm.find_events_file(arch_dir, ids["stem"])
    if events_path is None:
        print(f"Geen events-bestand gevonden in {arch_dir}")
        return

    events = fm.load_events(events_path)
    print(f"{len(events)} events gevonden voor {ids['stem']}")

    processed, raw_out = load_all_channels(args.night_dir, ids["stem"], include_raw=args.raw)
    if not processed:
        print("Geen kanalen geladen -- klopt het pad naar de EDF-bestanden (get_edf_dir)?")
        return
    print(f"Geladen (gepreprocest) kanalen: {list(processed.keys())}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.event_idx is not None:
        indices = [args.event_idx]
    elif args.limit_events is not None:
        indices = list(range(min(args.limit_events, len(events))))
    else:
        indices = [0]  # default: eerste event van de nacht

    for i in indices:
        if i >= len(events):
            print(f"  [SKIP] event {i} bestaat niet (deze nacht heeft er maar {len(events)})")
            continue
        ev = events.iloc[i]
        print(f"\n--- Event {i}: start={ev['start_sec']:.2f}s, eind={ev['end_sec']:.2f}s, "
              f"duur={ev['duration_sec']:.2f}s ---")

        df = build_matrix_for_window(processed, raw_out, ev["start_sec"], ev["end_sec"], pad_sec=args.pad_sec)
        summarize_motion(df)

        out_path = args.out_dir / f"{ids['stem']}_event{i}_channels.csv"
        df.to_csv(out_path, index=False, float_format="%.5f")
        print(f"  Weggeschreven naar: {out_path}")


if __name__ == "__main__":
    main()