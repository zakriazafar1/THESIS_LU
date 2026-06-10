"""
Debug script: print de gevonden paden en hypnogram paden voor de eerste 5 nachten.
"""
from pathlib import Path

RAW_ROOT = Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg")

def find_night_dirs(raw_root):
    return sorted([d for d in raw_root.rglob("*_edf") if d.is_dir()])

def parse_ids(edf_dir):
    stem  = edf_dir.name.replace("_edf", "")
    parts = stem.split("_")
    return {
        "subject_id": "_".join(parts[:3]),
        "night_id":   "_".join(parts[3:]),
        "group":      parts[1].upper(),
        "stem":       stem,
    }

all_dirs = find_night_dirs(RAW_ROOT)
print(f"Gevonden: {len(all_dirs)} nacht-mappen\n")

for d in all_dirs[:5]:
    ids      = parse_ids(d)
    night_dir = d.parent
    hyp_path  = night_dir / "sleepArchitecture" / f"{ids['stem']}.csv"

    print(f"edf_dir    : {d}")
    print(f"night_dir  : {night_dir}")
    print(f"hyp_path   : {hyp_path}")
    print(f"hyp exists : {hyp_path.exists()}")

    # Check wat er wel in de nacht-map zit
    if night_dir.exists():
        contents = [x.name for x in night_dir.iterdir()]
        print(f"night_dir contents: {contents}")

    print()
    