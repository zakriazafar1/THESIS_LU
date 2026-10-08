"""
=============================================================================
1.1_clip_nights.py

Herberekent de features van een paar nachten waarvan het laatste deel van de
opname artefact is, en vervangt ALLEEN die nachten in de bestaande
arousal_feature_matrix_ORIGIN.csv. De andere nachten worden niet opnieuw
berekend.

Hoe:
  - Gebruikt de functies uit 1_feature_matrix.py zelf (process_night e.d.),
    zodat de features exact op dezelfde manier berekend worden.
  - Elk RUW EDF-kanaal wordt direct na inladen afgeknipt bij het clip-punt,
    dus VOOR preprocessing. Filters, whole-night band-medianen (baselines)
    en de whole-night OXY-std zien daardoor alleen het schone deel.
    -> ook events VOOR het clip-punt krijgen (licht) andere waarden.
  - Events die na het clip-punt eindigen (of eroverheen lopen) vervallen.
  - Behouden events houden hun ORIGINELE event_idx en meta-kolommen
    (stage_rk, start_sec, sec_prev_event, ...); alleen de featurekolommen
    worden overschreven. Vervallen events laten een gat in event_idx achter,
    net als in 1.2_clean_feature_matrix.py.

Epoch-conventie (zoals in 1_feature_matrix.py / het hypnogram):
  epoch 1 begint op t = 0 s. "Clip bij epoch N" = epoch N en alles daarna
  valt weg, dus de data loopt van 0 tot (N - 1) * 30 s.

Gebruik:
  python 1.1_clip_nights.py --inspect        # alleen paden, aantallen, clip-tijden
  python 1.1_clip_nights.py                  # schrijft ..._ORIGIN_clipped.csv
  python 1.1_clip_nights.py --overwrite      # overschrijft ORIGIN (met backup),
                                             # zodat 1.2 ongewijzigd kan draaien
=============================================================================
"""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATIE
# =============================================================================

HERE = Path(__file__).resolve().parent
FM_SCRIPT = HERE / "1_feature_matrix.py"

FEATURE_DIR = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices\.feature info")
ORIGIN_FILE = FEATURE_DIR / "arousal_feature_matrix_ORIGIN.csv"

EPOCH_SEC = 30.0

# (subject_id, night_id) -> eerste epoch die WEGVALT (1-based, epoch 1 = t=0 s)
CLIP_NIGHTS: dict[tuple[str, str], int] = {
    ("bnbd_nsr_17598", "T0_N1"): 1011,
    ("bnbd_nsr_16379", "T0_N1"): 990,
    ("bnbd_nsr_19611", "T0_N1"): 910,
}

MATCH_TOL_SEC = 0.002   


def clip_end_sec(clip_epoch: int) -> float:
    return (clip_epoch - 1) * EPOCH_SEC


# =============================================================================
# HULPFUNCTIES
# =============================================================================

def load_fm_module(path: Path):
    """Importeert 1_feature_matrix.py (naam begint met een cijfer, dus via importlib)."""
    if not path.exists():
        sys.exit(f"1_feature_matrix.py niet gevonden: {path}")
    spec = importlib.util.spec_from_file_location("feature_matrix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Zelfde inlezer als 1.2: komma/puntkomma-detectie + decimale komma's."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if col in ("subject_id", "group", "night_id") or pd.api.types.is_numeric_dtype(df[col]):
            continue
        converted = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
        if converted.notna().any():
            df[col] = converted
    return df


def find_night_dir(fm, subject_id: str, night_id: str) -> Path | None:
    """Directe padconstructie (snel); valt terug op rglob binnen de groepsmap."""
    stem = f"{subject_id}_{night_id}"
    group_code = subject_id.split("_")[1].lower() if "_" in subject_id else ""
    groups = [g for g in fm.GROUPS if g.lower() == group_code] or list(fm.GROUPS)

    for g in groups:
        cand = fm.RAW_ROOT / g / subject_id / stem
        if (cand / "sleepArchitecture").is_dir():
            return cand
    for g in groups:
        gdir = fm.RAW_ROOT / g
        if gdir.is_dir():
            for hit in gdir.rglob(stem):
                if (hit / "sleepArchitecture").is_dir():
                    return hit
    return None


def make_clipped_loader(original_loader, t_end_sec: float, info: dict):
    """Wrapper rond fm.load_channel die elk ruw kanaal afknipt op t_end_sec."""
    def loader(edf_dir, name):
        loaded = original_loader(edf_dir, name)
        if loaded is None:
            return None
        data, sfreq = loaded
        info.setdefault("rec_len_sec", len(data) / sfreq)
        n_keep = int(round(t_end_sec * sfreq))
        if n_keep < len(data):
            data = data[:n_keep]
        else:
            info["not_clipped"] = True   # opname is korter dan het clip-punt
        return data, sfreq
    return loader


def match_to_origin(orig_night: pd.DataFrame, new: pd.DataFrame) -> list:
    """Geeft per nieuw event de index van het overeenkomende ORIGIN-event terug."""
    matched = []
    for _, r in new.iterrows():
        hit = orig_night.index[
            ((orig_night["start_sec"] - r["start_sec"]).abs() < MATCH_TOL_SEC)
            & ((orig_night["end_sec"] - r["end_sec"]).abs() < MATCH_TOL_SEC)
        ]
        if len(hit) != 1:
            raise RuntimeError(
                f"event {r['start_sec']:.3f}-{r['end_sec']:.3f} s: {len(hit)} matches in ORIGIN (verwacht 1)")
        matched.append(hit[0])
    return matched


# =============================================================================
# HOOFDLOOP
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true", help="Alleen paden/aantallen tonen, niets berekenen of schrijven")
    p.add_argument("--origin", type=Path, default=ORIGIN_FILE, help="pad naar arousal_feature_matrix_ORIGIN.csv")
    p.add_argument("--overwrite", action="store_true",
                   help="ORIGIN overschrijven (oude versie -> ..._ORIGIN_unclipped_backup.csv)")
    args = p.parse_args()

    fm = load_fm_module(FM_SCRIPT)
    if not args.origin.exists():
        sys.exit(f"Featurematrix niet gevonden: {args.origin}")
    df = load_feature_matrix(args.origin)
    print(f"ORIGIN ingelezen: {args.origin}  ({len(df)} events)")

    meta_end = fm.FEATURE_COLUMN_ORDER.index("sec_prev_event") + 1
    feat_cols = [c for c in fm.FEATURE_COLUMN_ORDER[meta_end:] if c in df.columns]

    drop_index = []
    n_done = 0

    for (subject_id, night_id), clip_epoch in CLIP_NIGHTS.items():
        stem = f"{subject_id}_{night_id}"
        t_end = clip_end_sec(clip_epoch)
        print(f"\n=== {stem}  (clip bij epoch {clip_epoch} -> data tot {t_end:.0f} s) ===")

        night_mask = (df["subject_id"] == subject_id) & (df["night_id"] == night_id)
        orig_night = df[night_mask]
        if orig_night.empty:
            print("  [SKIP] nacht niet gevonden in ORIGIN -- klopt het subject_id?")
            continue

        night_dir = find_night_dir(fm, subject_id, night_id)
        if night_dir is None:
            print(f"  [SKIP] nachtmap niet gevonden onder {fm.RAW_ROOT}")
            continue

        n_after = int((orig_night["end_sec"] > t_end).sum())
        print(f"  night_dir             : {night_dir}")
        print(f"  events in ORIGIN      : {len(orig_night)}")
        print(f"  waarvan na clip-punt  : {n_after}  (vervallen)")

        if args.inspect:
            continue

        info = {}
        original_loader = fm.load_channel
        fm.load_channel = make_clipped_loader(original_loader, t_end, info)
        try:
            new = fm.process_night(night_dir, fm.parse_ids(night_dir), inspect=False, start_idx=0)
        finally:
            fm.load_channel = original_loader

        if new is None or new.empty:
            print("  [SKIP] process_night gaf geen events terug")
            continue
        if info.get("not_clipped"):
            print(f"  [LET OP] opname ({info.get('rec_len_sec', np.nan):.0f} s) is korter dan het clip-punt; "
                  f"niets afgeknipt")
        else:
            print(f"  opnameduur            : {info.get('rec_len_sec', np.nan):.0f} s -> {t_end:.0f} s")

        new = new[new["end_sec"] <= t_end].reset_index(drop=True)

        kept_orig = orig_night[orig_night["end_sec"] <= t_end]
        matched = match_to_origin(kept_orig, new)
        if len(matched) != len(kept_orig):
            raise RuntimeError(f"{stem}: {len(kept_orig)} ORIGIN-events voor het clip-punt, "
                               f"maar {len(matched)} herberekend")

        # Overzicht: hoeveel verschuiven de features door de schone baseline?
        old_vals = df.loc[matched, feat_cols].to_numpy(dtype=float)
        new_vals = new[feat_cols].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(new_vals - old_vals) / np.abs(old_vals)
        print(f"  herberekend           : {len(new)} events")
        print(f"  mediane rel. verandering per feature (behouden events):")
        med = pd.Series(np.nanmedian(rel, axis=0), index=feat_cols)
        for c in [c for c in feat_cols if c.startswith("mean_")] + ["motion_rms", "oxy_amp_ratio"]:
            if c in med:
                print(f"    {c:<18} {100 * med[c]:6.1f} %")

        df.loc[matched, feat_cols] = new_vals
        drop_index.extend(orig_night.index[orig_night["end_sec"] > t_end])
        n_done += 1

    if args.inspect:
        print("\nInspectie klaar, niets geschreven.")
        return
    if n_done == 0:
        print("\nGeen nachten bijgewerkt, niets geschreven.")
        return

    out = df.drop(index=drop_index)
    print(f"\nTotaal: {len(df)} -> {len(out)} events ({len(drop_index)} vervallen na clip-punt)")

    if args.overwrite:
        backup = args.origin.with_name(args.origin.stem + "_unclipped_backup.csv")
        if not backup.exists():
            shutil.copy2(args.origin, backup)
            print(f"Backup: {backup}")
        out_path = args.origin
    else:
        out_path = args.origin.with_name(args.origin.stem + "_clipped.csv")

    out.to_csv(out_path, index=False, float_format="%.3f")
    print(f"Opgeslagen: {out_path}")


if __name__ == "__main__":
    main()