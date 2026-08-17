"""
=============================================================================
check_edf_channel_scale.py  (dependency-free versie)

Zelfde doel als eerder: print per kanaal in een EDF-bestand de eenheid
(physical_dimension) en het bereik (physical_min/max, digital_min/max),
zodat je weet welke schaalfactor er nodig is voor dX/dY/dZ.

Dit keer ZONDER pyedflib of mne -- puur met de standaardbibliotheek van
Python. Dat kan omdat de EDF-header een simpel, vast-opgebouwd ASCII-blok
is aan het begin van het bestand; die lezen we hier rechtstreeks uit, dus
geen gedoe meer met het compileren van pyedflib.

Gebruik:
  python check_edf_channel_scale.py --edf "<pad naar .edf>"
  python check_edf_channel_scale.py --edf "<pad naar .edf>" --channels dX dY dZ
  python check_edf_channel_scale.py --edf "<pad naar .edf>" --channels dX dY dZ --preview 5
=============================================================================
"""

import argparse
import struct
from pathlib import Path


def _read_field(fh, n):
    return fh.read(n).decode("ascii", errors="replace").strip()


def read_edf_header(edf_path: Path):
    """
    Leest alleen de header van een EDF-bestand (geen signaaldata), volgens de
    vaste EDF-specificatie. Geeft een dict terug met algemene info en een
    lijst met per-kanaal info.
    """
    with open(edf_path, "rb") as fh:
        version = _read_field(fh, 8)
        patient_id = _read_field(fh, 80)
        recording_id = _read_field(fh, 80)
        startdate = _read_field(fh, 8)
        starttime = _read_field(fh, 8)
        header_bytes = int(_read_field(fh, 8))
        _reserved = _read_field(fh, 44)
        n_records = int(_read_field(fh, 8))
        record_duration = float(_read_field(fh, 8))
        n_signals = int(_read_field(fh, 4))

        labels = [_read_field(fh, 16) for _ in range(n_signals)]
        transducer = [_read_field(fh, 80) for _ in range(n_signals)]
        phys_dim = [_read_field(fh, 8) for _ in range(n_signals)]
        phys_min = [float(_read_field(fh, 8)) for _ in range(n_signals)]
        phys_max = [float(_read_field(fh, 8)) for _ in range(n_signals)]
        dig_min = [int(_read_field(fh, 8)) for _ in range(n_signals)]
        dig_max = [int(_read_field(fh, 8)) for _ in range(n_signals)]
        prefilter = [_read_field(fh, 80) for _ in range(n_signals)]
        n_samples_per_record = [int(_read_field(fh, 8)) for _ in range(n_signals)]
        _reserved2 = [_read_field(fh, 32) for _ in range(n_signals)]

        channels = []
        for i in range(n_signals):
            sfreq = n_samples_per_record[i] / record_duration if record_duration > 0 else float("nan")
            channels.append({
                "label": labels[i],
                "unit": phys_dim[i],
                "phys_min": phys_min[i],
                "phys_max": phys_max[i],
                "dig_min": dig_min[i],
                "dig_max": dig_max[i],
                "sfreq": sfreq,
                "n_samples_per_record": n_samples_per_record[i],
                "transducer": transducer[i],
                "prefilter": prefilter[i],
            })

        return {
            "version": version,
            "patient_id": patient_id,
            "recording_id": recording_id,
            "startdate": startdate,
            "starttime": starttime,
            "header_bytes": header_bytes,
            "n_records": n_records,
            "record_duration": record_duration,
            "n_signals": n_signals,
            "channels": channels,
            "_data_offset": header_bytes,  # waar de signaaldata begint in het bestand
        }


def preview_samples(edf_path: Path, header: dict, channel_label: str, n_preview: int):
    """
    Leest de eerste n_preview RUWE (digitale) samples van 1 kanaal, en rekent
    ze om naar physical values met dezelfde formule die readSignal() ook
    gebruikt: physical = dig_value * gain + offset, waarbij
        gain   = (phys_max - phys_min) / (dig_max - dig_min)
        offset = phys_max - dig_max * gain
    """
    ch = next((c for c in header["channels"] if c["label"] == channel_label), None)
    if ch is None:
        return None

    ch_index = header["channels"].index(ch)
    record_size = sum(c["n_samples_per_record"] for c in header["channels"])
    offset_in_record = sum(c["n_samples_per_record"] for c in header["channels"][:ch_index])

    gain = (ch["phys_max"] - ch["phys_min"]) / (ch["dig_max"] - ch["dig_min"])
    offset = ch["phys_max"] - ch["dig_max"] * gain

    n_needed_records = max(1, -(-n_preview // ch["n_samples_per_record"]))  # ceil
    with open(edf_path, "rb") as fh:
        raw_digital = []
        for r in range(n_needed_records):
            fh.seek(header["_data_offset"] + r * record_size * 2 + offset_in_record * 2)
            n_this = min(ch["n_samples_per_record"], n_preview - len(raw_digital))
            data = fh.read(n_this * 2)
            vals = struct.unpack(f"<{n_this}h", data)
            raw_digital.extend(vals)
            if len(raw_digital) >= n_preview:
                break

    raw_digital = raw_digital[:n_preview]
    physical = [v * gain + offset for v in raw_digital]
    return raw_digital, physical


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edf", type=Path, required=True, help="Pad naar een .edf-bestand")
    parser.add_argument("--channels", nargs="*", default=None,
                         help="Specifieke kanaalnamen om te tonen (default: alle kanalen)")
    parser.add_argument("--preview", type=int, default=0,
                         help="Aantal eerste samples per kanaal laten zien (default: 0 = geen preview)")
    args = parser.parse_args()

    if not args.edf.exists():
        print(f"Bestand niet gevonden: {args.edf}")
        return

    header = read_edf_header(args.edf)
    all_labels = [c["label"] for c in header["channels"]]

    if args.channels is None:
        selected = header["channels"]
    else:
        selected = []
        for name in args.channels:
            match = next((c for c in header["channels"] if c["label"] == name), None)
            if match is None:
                print(f"  [WAARSCHUWING] kanaal '{name}' niet gevonden. Beschikbare kanalen: {all_labels}")
                continue
            selected.append(match)

    print(f"\nBestand: {args.edf.name}")
    print(f"Opname-datum/tijd: {header['startdate']} {header['starttime']}  |  "
          f"{header['n_records']} records x {header['record_duration']}s")
    print(f"\n{'Kanaal':<15} {'Eenheid':<10} {'Phys min':>14} {'Phys max':>14} "
          f"{'Dig min':>10} {'Dig max':>10} {'Sfreq':>8}")
    print("-" * 95)

    for ch in selected:
        print(f"{ch['label']:<15} {ch['unit']:<10} {ch['phys_min']:>14.2f} {ch['phys_max']:>14.2f} "
              f"{ch['dig_min']:>10} {ch['dig_max']:>10} {ch['sfreq']:>8.2f}")

        if args.preview > 0:
            result = preview_samples(args.edf, header, ch["label"], args.preview)
            if result is not None:
                raw_digital, physical = result
                print(f"    eerste {args.preview} ruwe (digitale) samples: {raw_digital}")
                print(f"    zelfde samples omgerekend naar physical:      "
                      f"{[round(p, 4) for p in physical]}")

    print("\nHoe te lezen:")
    print("  - 'Eenheid' is wat er in de EDF zelf staat. Staat hier 'uV' voor EEG, en iets")
    print("    als 'mg', 'g', 'raw' of leeg voor dX/dY/dZ -- dat vertelt je meteen of er al")
    print("    een zinnige eenheid bekend is, of dat je zelf moet schalen.")
    print("  - 'Phys min/max' is het bereik van de FYSIEKE waarde (na omrekening vanuit de")
    print("    ruwe digitale sample via gain+offset -- zie de preview-kolommen hierboven).")
    print("    Is dit bereik voor dX/dY/dZ bijvoorbeeld [-2000000, 2000000] met een lege of")
    print("    onduidelijke eenheid? Dan staat de accelerometer in de EDF zelf niet correct")
    print("    gekalibreerd naar g's -- dan moet je de ZMax-specsheet raadplegen voor de")
    print("    juiste schaalfactor, want die zit niet betrouwbaar in het bestand.")
    print("  - Is de eenheid wel bruikbaar (bv. 'mg')? Dan hoef je alleen te delen door 1000")
    print("    om naar g te gaan -- geen aparte lookup nodig.")


if __name__ == "__main__":
    main()