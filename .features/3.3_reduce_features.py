"""
=============================================================================
3.3_reduce_features.py

Reduceert de arousal-featurematrix tot de 8 gekozen features, in alle drie
de versies: origineel, getransformeerd (ln) en geschaald (z-score).

Keuze (zie 3.1_corr_vif.py, corr_groups.csv / dendrogram):
  Features zijn gegroepeerd met hiërarchische clustering op 1 - |Spearman rho|
  (complete linkage, snijlijn |rho| = 0.8). Per frequentieband vallen L, R,
  mean en de peak-varianten in één groep; per band is de over beide hemisferen
  gemiddelde ratio behouden (mean_*_ratio):
    - mean i.p.v. L/R: minder ruis, arousal is een globaal fenomeen (L~R r ~0.87)
    - ratio i.p.v. peak_ratio: peak hangt meer samen met duration_sec
  duration_sec, oxy_amp_ratio en motion_rms correleren met geen enkele andere
  feature >= 0.8 en worden allemaal behouden.

Nachtexclusie:
  De transformed- en scaled-matrix zijn al zonder EXCLUDE_NIGHTS (gebeurt in
  2.1). De originele matrix (FILTERED) nog niet -- die exclusie wordt hier
  toegepast, met dezelfde lijst als in 2.1, zodat alle drie de versies
  exact dezelfde events bevatten. Dat wordt gecontroleerd.

Scaling hoeft niet opnieuw: z-scoring gebeurt per feature, dus een subset van
kolommen uit de geschaalde matrix is identiek aan opnieuw schalen.

Stappenplan:
  1. Drie matrices inladen (origineel, getransformeerd, geschaald).
  2. Nachten uitsluiten in de originele matrix.
  3. Controle: zelfde events in dezelfde volgorde in alle drie.
  4. Metadata + de 8 features selecteren en wegschrijven.

Gebruik:
  python 3.3_reduce_features.py
=============================================================================
"""

import argparse
from pathlib import Path

import pandas as pd

# =============================================================================
# CONFIGURATIE
# =============================================================================

BASE = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2"
)

ORIGINAL_PATH = BASE / r"1. feature matrices\.feature info\arousal_feature_matrix_FILTERED.csv"
TRANSFORMED_PATH = BASE / r"2. preprocessing\transformed\arousal_feature_matrix_transformed.csv"
SCALED_PATH = BASE / r"2. preprocessing\scaled\arousal_feature_matrix_scaled.csv"

OUTPUT_DIR = BASE / r"reduced feature matrix"

METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]

# Kolommen die samen één event uniek identificeren 
EVENT_KEY = ["subject_id", "night_id", "event_idx"]

KEEP_FEATURES = [
    "mean_delta_ratio",
    "mean_theta_ratio",
    "mean_alpha_ratio",
    "mean_sigma_ratio",
    "mean_beta_ratio",
    "duration_sec",
    "oxy_amp_ratio",
    "motion_rms",
]

# Moet gelijk zijn aan EXCLUDE_NIGHTS in 2.1_transform_features.py. 
# LEEG = niet nodig.
EXCLUDE_NIGHTS: list[tuple[str, str]] = []


# =============================================================================
# SECTIE 1 — INLADEN
# =============================================================================

def read_csv_robust(path: Path, label: str) -> pd.DataFrame:
    """
    Leest een CSV met automatische separator-detectie, en leest opnieuw in met
    decimal=',' als numerieke kolommen als tekst binnenkomen (Excel-NL).
    """
    if not path.exists():
        raise FileNotFoundError(f"{label}: {path} bestaat niet.")
    df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[1] == 1:
        raise ValueError(f"{label}: {path} lijkt maar 1 kolom te hebben -- separator niet herkend.")

    text_cols = {"subject_id", "group", "night_id"}
    check = [c for c in df.columns if c not in text_cols]
    n_bad = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in check)
    if n_bad > 0:
        df2 = pd.read_csv(path, sep=None, engine="python", decimal=",")
        n_bad2 = sum(not pd.api.types.is_numeric_dtype(df2[c]) for c in check if c in df2.columns)
        if n_bad2 < n_bad:
            print(f"[LET OP] {label}: decimaal-komma gedetecteerd, opnieuw ingelezen met decimal=','.")
            df = df2

    print(f"{label:<12} geladen: {path.name}  shape={df.shape}")
    return df


# =============================================================================
# SECTIE 2 — NACHTEN UITSLUITEN
# =============================================================================

def night_keys(df: pd.DataFrame) -> pd.Series:
    return df["subject_id"].astype(str).str.strip() + "|" + df["night_id"].astype(str).str.strip()


def exclude_nights(df: pd.DataFrame, exclude: list[tuple[str, str]], label: str) -> pd.DataFrame:
    """Verwijdert alle events van de opgegeven (subject_id, night_id)-combinaties."""
    excl = {f"{s}|{n}" for s, n in exclude}
    mask = night_keys(df).isin(excl)
    if mask.any():
        print(f"{label}: {int(mask.sum())} events uit uitgesloten nachten verwijderd "
              f"({int((~mask).sum())} over).")
    return df.loc[~mask].reset_index(drop=True)


# =============================================================================
# SECTIE 3 — CONTROLES
# =============================================================================

def check_columns(df: pd.DataFrame, label: str) -> list[str]:
    """Controleert of alle KEEP_FEATURES aanwezig zijn; geeft de aanwezige metadata terug."""
    missing = [c for c in KEEP_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"{label}: deze features ontbreken: {missing}")
    return [c for c in METADATA_COLS if c in df.columns]


def check_same_events(dfs: dict[str, pd.DataFrame]) -> None:
    """
    Alle versies moeten exact dezelfde events in dezelfde volgorde bevatten,
    anders lopen de rijen van origineel/transformed/scaled uit elkaar.
    """
    key_cols = [c for c in EVENT_KEY if all(c in d.columns for d in dfs.values())]
    if not key_cols:
        print("[LET OP] geen event-sleutelkolommen gevonden, alleen aantal rijen gecontroleerd.")

    ref_label, ref = next(iter(dfs.items()))
    ref_keys = ref[key_cols].astype(str).agg("|".join, axis=1) if key_cols else None
    for label, d in dfs.items():
        if len(d) != len(ref):
            raise ValueError(f"Aantal events verschilt: {ref_label}={len(ref)}, {label}={len(d)}. "
                             "Staat EXCLUDE_NIGHTS gelijk aan die in 2.1, en zijn 2.1/2.2 opnieuw gedraaid?")
        if key_cols:
            keys = d[key_cols].astype(str).agg("|".join, axis=1)
            if not keys.equals(ref_keys):
                n_diff = int((keys != ref_keys).sum())
                raise ValueError(f"{label}: {n_diff} rij(en) wijken af van {ref_label} "
                                 f"op {key_cols} (andere events of andere volgorde).")
    print(f"Controle OK: alle versies bevatten dezelfde {len(ref)} events in dezelfde volgorde.")


def check_excluded_nights_gone(df: pd.DataFrame, label: str) -> None:
    excl = {f"{s}|{n}" for s, n in EXCLUDE_NIGHTS}
    left = set(night_keys(df)) & excl
    if left:
        raise ValueError(f"{label}: uitgesloten nachten zitten er nog in: {sorted(left)} "
                         "-- draai 2.1 en 2.2 opnieuw.")


# =============================================================================
# HOOFDLOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=ORIGINAL_PATH)
    parser.add_argument("--transformed", type=Path, default=TRANSFORMED_PATH)
    parser.add_argument("--scaled", type=Path, default=SCALED_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stap 1: inladen ---
    dfs = {
        "original": read_csv_robust(args.original, "original"),
        "transformed": read_csv_robust(args.transformed, "transformed"),
        "scaled": read_csv_robust(args.scaled, "scaled"),
    }

    # --- Stap 2: nachten uitsluiten in de originele matrix ---
    dfs["original"] = exclude_nights(dfs["original"], EXCLUDE_NIGHTS, "original")
    for label in ("transformed", "scaled"):
        check_excluded_nights_gone(dfs[label], label)

    # --- Stap 3: controles ---
    meta = {label: check_columns(d, label) for label, d in dfs.items()}
    check_same_events(dfs)

    # --- Stap 4: selecteren + wegschrijven ---
    names = {
        "original": "arousal_features_reduced_original.csv",
        "transformed": "arousal_features_reduced_transformed.csv",
        "scaled": "arousal_features_reduced_scaled.csv",
    }
    for label, d in dfs.items():
        reduced = d[meta[label] + KEEP_FEATURES]
        reduced.to_csv(out_dir / names[label], index=False)
        print(f"{label:<12} -> {names[label]}  shape={reduced.shape}")

    # overzicht van behouden en weggelaten features
    all_feats = [c for c in dfs["scaled"].columns if c not in METADATA_COLS]
    pd.DataFrame({
        "feature": all_feats,
        "kept": [c in KEEP_FEATURES for c in all_feats],
    }).to_csv(out_dir / "feature_selection.csv", index=False)

    dropped = [c for c in all_feats if c not in KEEP_FEATURES]
    print(f"\n{len(KEEP_FEATURES)} features behouden: {KEEP_FEATURES}")
    print(f"{len(dropped)} features weggelaten: {dropped}")
    print(f"\nAlles opgeslagen in: {out_dir}")
    print("  - arousal_features_reduced_original.csv     (ruwe waarden, voor beschrijving/interpretatie)")
    print("  - arousal_features_reduced_transformed.csv  (ln-getransformeerd)")
    print("  - arousal_features_reduced_scaled.csv       (z-scores, input voor UMAP/HDBSCAN)")
    print("  - feature_selection.csv                     (welke features behouden/weggelaten)")


if __name__ == "__main__":
    main()