"""
=============================================================================
convert_to_scoringhero.py

Converteert alle candidate_events_*.csv bestanden in EVENTS_DIR naar het
formaat dat ScoringHero verwacht — identiek aan Lucija's human rater bestanden.

Input (algoritme, puntkomma-gescheiden, komma als decimaal):
  subject_id;night_id;start_sec;end_sec;duration_sec;...gate

Output (ScoringHero, komma-gescheiden, punt als decimaal):
  event,start,stop,duration,channel
  Arousal,116.8594,118.5000,1.6406,all
  ...

Gebruik:
  python convert_to_scoringhero.py               # alle events (ook A/B/C)
  python convert_to_scoringhero.py --accepted    # alleen gate=accepted
  python convert_to_scoringhero.py --limit 5     # test op 5 bestanden
=============================================================================
"""

import argparse
from pathlib import Path

import pandas as pd

# ── Configuratie ──────────────────────────────────────────────────────────────
EVENTS_DIR = Path(r"C:\Users\zafar\Documents\THESIS_OUTPUTS\2_candidate_events")
OUTPUT_DIR = Path(r"C:\Users\zafar\Documents\THESIS_OUTPUTS\3_scoringhero_events")

DECIMAL_COLS = ["start_sec", "end_sec", "duration_sec"]


def convert_file(csv_path: Path, accepted_only: bool) -> int:
    """
    Converteert één algoritme CSV naar ScoringHero CSV formaat.
    Geeft het aantal geëxporteerde events terug.
    """
    df = pd.read_csv(csv_path, sep=";")

    # Converteer komma-decimalen naar float
    for col in DECIMAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    # Optioneel: alleen geaccepteerde events
    if accepted_only and "gate" in df.columns:
        df = df[df["gate"] == "accepted"].copy()

    if df.empty:
        return 0

    # Bouw output DataFrame in ScoringHero formaat
    # Identiek aan Lucija's bestanden: event, start, stop, duration, channel
    out = pd.DataFrame({
        "event":    "Arousal",
        "start":    df["start_sec"].values,
        "stop":     df["end_sec"].values,
        "duration": df["duration_sec"].values,
        "channel":  "all",
    })

    # Output pad — zelfde mappenstructuur als input maar onder OUTPUT_DIR
    relative = csv_path.relative_to(EVENTS_DIR)
    out_path = (
        OUTPUT_DIR
        / relative.parent
        / f"scoringhero_{csv_path.stem.replace('candidate_events_', '')}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Komma als kolom-separator, punt als decimaal (zelfde als Lucija's bestanden)
    out.to_csv(out_path, index=False, sep="\t", float_format="%.6f")

    return len(out)


def run(accepted_only: bool, limit: int = None):
    print("=" * 60)
    print("  Convert to ScoringHero format")
    print(f"  Input  : {EVENTS_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Filter : {'alleen accepted' if accepted_only else 'alle events (ook A/B/C)'}")
    print("=" * 60)

    all_files = sorted(EVENTS_DIR.rglob("candidate_events_*.csv"))

    if not all_files:
        print(f"\n  [FOUT] Geen candidate_events_*.csv gevonden in {EVENTS_DIR}")
        return

    if limit:
        all_files = all_files[:limit]

    print(f"\n  Gevonden bestanden : {len(all_files)}\n")

    total_events = 0
    n_ok         = 0
    n_empty      = 0
    n_fail       = 0

    for csv_path in all_files:
        try:
            n = convert_file(csv_path, accepted_only)
            if n > 0:
                print(f"  OK    {csv_path.stem:55s}  ({n} events)")
                total_events += n
                n_ok         += 1
            else:
                print(f"  LEEG  {csv_path.stem:55s}  (0 events na filter)")
                n_empty      += 1
        except Exception as e:
            print(f"  FAIL  {csv_path.stem:55s}  {e}")
            n_fail += 1

    print(f"\n{'=' * 60}")
    print(f"  Geconverteerd  : {n_ok} bestanden")
    print(f"  Leeg (0 events): {n_empty} bestanden")
    print(f"  Mislukt        : {n_fail} bestanden")
    print(f"  Totaal events  : {total_events}")
    print(f"  Output map     : {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converteer candidate_events CSV naar ScoringHero formaat"
    )
    parser.add_argument(
        "--accepted", action="store_true",
        help="Exporteer alleen events met gate=accepted (micro-arousals)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Verwerk alleen de eerste N bestanden (voor testen)"
    )
    args = parser.parse_args()
    run(accepted_only=args.accepted, limit=args.limit)