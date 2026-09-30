"""
=============================================================================
2.1_transform_features.py

Log-transformatie van de arousal-featurematrix (arousal_feature_matrix_FILTERED.csv).

Stappenplan:
  1. Visualiseer de verdeling van elke feature (histogram met skewness),
     vóór en na transformatie.
  2. Transformeer met een VASTE regel (geen auto-selectie per feature):
       - alle features -> natuurlijke log (ln), behalve:
       - FORCE_UNTOUCHED_COLS (motion_rms, oxy_amp_ratio) -> ongetransformeerd.
     Rationale: de spectrale features zijn ratio's (strikt positief, sterk
     rechts-scheef, veel waarden < 1). ln maakt ze symmetrisch rond 0
     (halvering en verdubbeling even ver van 0) en haalt de scheefheid
     grotendeels weg. log1p is hier minder geschikt: die drukt het bereik
     < 1 plat, waardoor de skew blijft. motion_rms en oxy_amp_ratio zijn
     nauwelijks scheef; ln zou ze juist links-scheef maken.
     ln vereist strikt positieve waarden: het script stopt met een foutmelding
     als een log-feature een waarde <= 0 bevat.
     Skew voor/na wordt per feature weggeschreven (transform_choices.csv).

  0. Vooraf: events uit EXCLUDE_NIGHTS worden verwijderd (night-level
     baseline/opname-artefact, zie configuratie). Log -> excluded_nights.csv.

  Scaling gebeurt in 2.2_scale_features.py, dat de hier weggeschreven
  arousal_feature_matrix_transformed.csv als input gebruikt.

Gebruik:
  python 2.1_transform_features.py
  python 2.1_transform_features.py --input <pad> --output-dir <map>
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

# =============================================================================
# CONFIGURATIE
# =============================================================================

DEFAULT_INPUT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices\.feature info\arousal_feature_matrix_FILTERED.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\transformed"
)

# Metadata kolommen, dus uitgesloten van distributie-plots en transformatie.
# stage_rk is metadata voor interpretatie/descriptive_note, geen clustering-input.
# duration_sec is WEL een feature.
METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk"
]

N_COLS_GRID = 5  # aantal subplots per rij in de histogram-grid

# Nachten uitgesloten wegens een night-level baseline/opname-artefact:
# 90-100% van de events in deze nachten had extreem lage spectrale ratio's
# (z < -3 op alle banden), gevonden met 3.1.1_scatter_divergent.py.
# Andere nachten van dezelfde subjects waren wel in orde.
EXCLUDE_NIGHTS: list[tuple[str, str]] = [
    ("bnbd_nsr_17598", "T0_N1"),
    ("bnbd_nsr_16379", "T0_N1"),
    ("bnbd_nsr_19611", "T0_N1"),
]

# Features die NIET getransformeerd worden (nauwelijks scheef; ln zou ze links-scheef maken).
FORCE_UNTOUCHED_COLS: list[str] = ["motion_rms", "oxy_amp_ratio"]

# =============================================================================
# STAP 1 - INLADEN
# =============================================================================

def load_feature_matrix(path: Path) -> pd.DataFrame:
    """
    Leest de featurematrix in. Detecteert het scheidingsteken automatisch
    (sep=None + engine="python"), en checkt daarna of numerieke kolommen als
    tekst zijn binnengekomen (Excel-NL-scenario: puntkomma als scheiding EN
    komma als decimaalteken) -- zo ja, dan opnieuw inlezen met decimal=",".
    """
    df = pd.read_csv(path, sep=None, engine="python")

    if df.shape[1] == 1:
        raise ValueError(
            f"{path} lijkt maar 1 kolom te hebben ({df.columns[0]!r}) -- "
            "het scheidingsteken kon niet automatisch herkend worden, of het "
            "bestand is beschadigd. Open het bestand in een teksteditor om te "
            "checken wat er precies staat."
        )

    non_numeric_id_cols = {"subject_id", "group", "night_id"}
    check_cols = [c for c in df.columns if c not in non_numeric_id_cols]
    n_non_numeric = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in check_cols)

    if n_non_numeric > 0:
        df_comma_decimal = pd.read_csv(path, sep=None, engine="python", decimal=",")
        n_non_numeric_comma = sum(
            not pd.api.types.is_numeric_dtype(df_comma_decimal[c])
            for c in check_cols if c in df_comma_decimal.columns
        )
        if n_non_numeric_comma < n_non_numeric:
            print(f"[LET OP] {n_non_numeric} kolom(men) kwamen als tekst binnen -- "
                  f"decimaal-komma gedetecteerd (Excel-NL-formaat), opnieuw ingelezen met decimal=','.")
            df = df_comma_decimal

    print(f"Featurematrix geladen: {path}")
    print(f"Shape: {df.shape}")
    return df


def exclude_nights(df: pd.DataFrame, exclude: list[tuple[str, str]],
                   log_path: Path) -> pd.DataFrame:
    """
    Verwijdert alle events van de opgegeven (subject_id, night_id)-combinaties.
    Filtert op de combinatie, want night_id (T0_N1 etc.) komt bij elk subject terug.
    Schrijft weg hoeveel events per nacht verwijderd zijn.
    """
    if not exclude:
        return df

    keys = df["subject_id"].astype(str).str.strip() + "|" + df["night_id"].astype(str).str.strip()
    excl_keys = {f"{s}|{n}" for s, n in exclude}

    not_found = sorted(excl_keys - set(keys))
    if not_found:
        print(f"[LET OP] deze nachten staan niet in de data: {not_found}")

    mask = keys.isin(excl_keys)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (df[mask].groupby(["subject_id", "night_id"]).size()
       .rename("n_events_removed").to_csv(log_path))
    print(f"{int(mask.sum())} events uit {len(excl_keys) - len(not_found)} nacht(en) verwijderd "
          f"({int((~mask).sum())} van {len(df)} events over). Log: {log_path}")
    return df.loc[~mask].reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Alle kolommen behalve de metadata-kolommen -> dit zijn de clustering-features."""
    return [c for c in df.columns if c not in METADATA_COLS]


# =============================================================================
# STAP 2 - DISTRIBUTIES VISUALISEREN (voor/na transformatie)
# =============================================================================

def _skew_or_nan(vals: pd.Series) -> float:
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    return skew(vals) if len(vals) >= 3 else np.nan


def plot_distributions(df: pd.DataFrame, feature_cols: list[str], out_path: Path) -> None:
    """Grid van histogrammen (1 per feature), met skewness in de titel."""
    n_feats = len(feature_cols)
    n_cols = N_COLS_GRID
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals):
            ax.hist(vals, bins=40, color="steelblue", edgecolor="white")
            s = skew(vals) if len(vals) >= 3 else np.nan
            ax.set_title(f"{col}\nskew={s:.2f}" if not np.isnan(s) else col, fontsize=9)
        else:
            ax.set_title(f"{col}\n(geen data)", fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(n_feats, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Distributie-grid opgeslagen: {out_path}")


# =============================================================================
# STAP 3 - TRANSFORMATIE (vaste regel: ln, behalve FORCE_UNTOUCHED_COLS)
# =============================================================================

def classify_transform_columns(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Verdeelt de features in log_cols (-> ln) en untouched_cols (FORCE_UNTOUCHED_COLS)."""
    untouched = [c for c in feature_cols if c in FORCE_UNTOUCHED_COLS]
    log_cols = [c for c in feature_cols if c not in FORCE_UNTOUCHED_COLS]
    missing = [c for c in FORCE_UNTOUCHED_COLS if c not in feature_cols]
    if missing:
        print(f"[LET OP] FORCE_UNTOUCHED_COLS bevat kolommen die niet in de matrix staan: {missing}")
    return log_cols, untouched


def check_strictly_positive(df: pd.DataFrame, log_cols: list[str]) -> None:
    """ln is alleen gedefinieerd voor x > 0 -- stop met een duidelijke fout als dat niet klopt."""
    problems = []
    for c in log_cols:
        vals = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        n_nonpos = int((vals <= 0).sum())
        if n_nonpos:
            problems.append(f"  {c}: {n_nonpos} waarde(n) <= 0 (min = {vals.min():.4g})")
    if problems:
        raise ValueError(
            "ln-transformatie niet mogelijk, deze features bevatten waarden <= 0:\n"
            + "\n".join(problems)
            + "\nVoeg ze toe aan FORCE_UNTOUCHED_COLS of kies een andere transformatie."
        )


def apply_log_transform(df: pd.DataFrame, log_cols: list[str], untouched_cols: list[str]
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Past ln toe op log_cols; untouched_cols blijven origineel. Geeft df + overzicht terug."""
    out = df.copy()
    choices = []
    for c in log_cols + untouched_cols:
        raw = df[c].replace([np.inf, -np.inf], np.nan)
        if c in log_cols:
            out[c] = np.log(raw)
            transform = "ln"
        else:
            out[c] = raw
            transform = "geen (FORCE_UNTOUCHED)"
        sb, sa = _skew_or_nan(raw), _skew_or_nan(out[c])
        choices.append({
            "feature": c,
            "transform": transform,
            "skew_before": round(sb, 3) if pd.notna(sb) else np.nan,
            "skew_after": round(sa, 3) if pd.notna(sa) else np.nan,
        })
    return out, pd.DataFrame(choices)


# =============================================================================
# HOOFDLOOP
# =============================================================================

def run(df: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Stap 1: distributies vóór transformatie ---
    plot_distributions(df, feature_cols, output_dir / "feature_distributions.png")

    # --- Stap 2: transformatie ---
    log_cols, untouched_cols = classify_transform_columns(feature_cols)
    check_strictly_positive(df, log_cols)
    df_t, transform_choices = apply_log_transform(df, log_cols, untouched_cols)

    df_t.to_csv(output_dir / "arousal_feature_matrix_transformed.csv", index=False)
    transform_choices.to_csv(output_dir / "transform_choices.csv", index=False)
    plot_distributions(df_t, feature_cols, output_dir / "feature_distributions_transformed.png")

    print(f"\nVaste regel -- {len(log_cols)} features ln-getransformeerd, "
          f"{len(untouched_cols)} ongemoeid ({untouched_cols}).")
    print(transform_choices.to_string(index=False))
    print(f"\nOpgeslagen in {output_dir}:"
          "\n  - arousal_feature_matrix_transformed.csv\n  - transform_choices.csv"
          "\n  - excluded_nights.csv"
          "\n  - feature_distributions.png\n  - feature_distributions_transformed.png")

    return df_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Pad naar arousal_feature_matrix_FILTERED.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Map voor de output")
    args = parser.parse_args()

    df = load_feature_matrix(args.input)
    df = exclude_nights(df, EXCLUDE_NIGHTS, args.output_dir / "excluded_nights.csv")
    feature_cols = get_feature_columns(df)
    print(f"\n{len(feature_cols)} features (metadata-kolommen uitgesloten): {feature_cols}")

    run(df, feature_cols, args.output_dir)


if __name__ == "__main__":
    main()