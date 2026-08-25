"""
=============================================================================
3.1_corr_vif.py

Correlatie- en multicollineariteit-check op de geschaalde arousal-featurematrix
(output van 2.2_scale_features.py), als check tussen scaling en UMAP/HDBSCAN.

Waarom Spearman i.p.v. Pearson:
  Ook na RobustScaler blijft een deel van de features behoorlijk scheef
  (skew tot ~2.0 voor de sigma/alpha-ratio's, zie distribution_stats_after_scaling.csv).
  RobustScaler verandert de vorm van de verdeling niet (alleen centrum/schaal),
  dus de resterende scheefheid en staart-outliers kunnen een Pearson-r
  vertekenen. Spearman (rank-based) is ongevoelig voor de exacte schaal/vorm
  en vangt elke monotone relatie, niet alleen lineaire -- precies wat nodig is
  om redundante features te vinden vóór UMAP.

Waarom VIF op de ranks:
  VIF (Variance Inflation Factor) vangt multicollineariteit over meerdere
  features tegelijk (in tegenstelling tot pairwise correlatie), maar de
  standaard VIF-formule (uit OLS) is zelf Pearson-based. Door eerst te
  rank-transformeren (scipy.stats.rankdata) en dan VIF op de ranks te draaien,
  krijg je een robuustere versie die aansluit bij de Spearman-aanpak hierboven.

Stappenplan:
  1. Geschaalde featurematrix inladen (arousal_feature_matrix_scaled.csv).
  2. Spearman-correlatiematrix berekenen + wegschrijven.
  3. Correlatie-heatmap plotten.
  4. Sterk gecorreleerde paren (|rho| > CORR_THRESHOLD) oplijsten.
  5. VIF berekenen op rank-getransformeerde features + wegschrijven.
  6. Features met VIF > VIF_THRESHOLD flaggen.

Gebruik:
  python 3.1_corr_vif.py
      -> leest arousal_feature_matrix_scaled.csv uit INPUT_DIR, schrijft
         alle output naar OUTPUT_DIR (zie configuratie hieronder).
  python 3.1_corr_vif.py --input pad/naar/andere_scaled.csv
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =============================================================================
# CONFIGURATIE
# =============================================================================

# Map waar 2.2_scale_features.py de geschaalde featurematrix neerzet.
INPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled"
)

# Map waar de output van dit script (corr matrix, heatmap, VIF-tabel) naartoe gaat.
OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\3. feature selection\corr_vif" 
)

DEFAULT_INPUT = INPUT_DIR / "arousal_feature_matrix_scaled.csv"

# Zelfde metadata-kolommen als in 2.2_scale_features.py -- deze worden
# uitgesloten van de correlatie/VIF-check.
METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]

CORR_THRESHOLD = 0.80   # |rho| boven deze grens wordt als "sterk gecorreleerd" gelogd
VIF_THRESHOLD = 5.0     # gangbare vuistregel-grens voor multicollineariteit


# =============================================================================
# SECTIE 1 — INLADEN
# =============================================================================

def load_scaled_matrix(path: Path) -> pd.DataFrame:
    """
    Leest de geschaalde featurematrix in. Zelfde separator/decimal-detectie
    als in 2.2_scale_features.py, voor het geval het bestand tussendoor met
    een NL-Excel-locale is geopend en opgeslagen.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} bestaat niet -- run eerst 2.2_scale_features.py, of geef het juiste "
            "pad mee met --input."
        )
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

    print(f"Geschaalde featurematrix geladen: {path}")
    print(f"Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Alle kolommen behalve de metadata-kolommen -> dit zijn de te checken features."""
    return [c for c in df.columns if c not in METADATA_COLS]


# =============================================================================
# SECTIE 2 — SPEARMAN-CORRELATIEMATRIX
# =============================================================================

def compute_spearman_corr(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Spearman-correlatiematrix op de features (pairwise, met NaN/inf als missing)."""
    sub = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    corr = sub.corr(method="spearman")
    return corr


def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    """Heatmap van de Spearman-correlatiematrix, geannoteerd."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.45 * n + 2))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        annot=n <= 20,          # alleen annoteren als het leesbaar blijft
        fmt=".2f",
        annot_kws={"size": 6},
        cbar_kws={"label": "Spearman rho"},
        ax=ax,
    )
    ax.set_title("Spearman-correlatiematrix (geschaalde features)", fontsize=12)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Correlatie-heatmap opgeslagen: {out_path}")


def list_high_corr_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Alle feature-paren met |rho| > threshold, gesorteerd van hoog naar laag."""
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = corr.iloc[i, j]
            if pd.notna(rho) and abs(rho) > threshold:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "spearman_rho": round(float(rho), 4),
                })
    out = pd.DataFrame(pairs)
    if not out.empty:
        out = out.reindex(
            out["spearman_rho"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
    return out


# =============================================================================
# SECTIE 3 — VIF OP RANK-GETRANSFORMEERDE FEATURES
# =============================================================================

def compute_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    VIF per feature, berekend op rank-getransformeerde waarden (scipy.stats.rankdata)
    i.p.v. de ruwe geschaalde waarden -- dit maakt de VIF-check consistent met de
    Spearman-aanpak (rank-based, ongevoelig voor resterende scheefheid).

    Rijen met een NaN/inf in minstens één feature worden voor de VIF-berekening
    uitgesloten (VIF/OLS kan niet met missing values omgaan); dit is alleen
    relevant voor deze check en raakt de featurematrix zelf niet.
    """
    sub = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    n_before = len(sub)
    sub = sub.dropna()
    n_dropped = n_before - len(sub)
    if n_dropped > 0:
        print(f"[LET OP] {n_dropped} rij(en) met NaN/inf uitgesloten voor VIF-berekening "
              f"({len(sub)} van {n_before} rijen gebruikt).")

    ranked = sub.apply(lambda col: rankdata(col), axis=0)
    ranked = pd.DataFrame(ranked, columns=feature_cols)

    rows = []
    for i, col in enumerate(feature_cols):
        vif_val = variance_inflation_factor(ranked.to_numpy(), i)
        rows.append({"feature": col, "VIF": round(float(vif_val), 3)})

    vif_df = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_df


# =============================================================================
# HOOFDLOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help="Pad naar arousal_feature_matrix_scaled.csv")
    args = parser.parse_args()

    df = load_scaled_matrix(args.input)
    feature_cols = get_feature_columns(df)
    print(f"\n{len(feature_cols)} features (metadata-kolommen uitgesloten): {feature_cols}")

    # --- Stap 2+3: Spearman-correlatiematrix + heatmap ---
    corr = compute_spearman_corr(df, feature_cols)
    corr.to_csv(OUTPUT_DIR / "spearman_corr_matrix.csv")
    plot_corr_heatmap(corr, OUTPUT_DIR / "spearman_corr_heatmap.png")

    # --- Stap 4: sterk gecorreleerde paren ---
    high_corr = list_high_corr_pairs(corr, CORR_THRESHOLD)
    high_corr.to_csv(OUTPUT_DIR / "high_corr_pairs.csv", index=False)
    print(f"\nStap 4: {len(high_corr)} paar/paren met |rho| > {CORR_THRESHOLD}:")
    print(high_corr.to_string(index=False) if not high_corr.empty else "  (geen)")

    # --- Stap 5: VIF op ranks ---
    vif_df = compute_vif(df, feature_cols)
    vif_df.to_csv(OUTPUT_DIR / "vif_ranked.csv", index=False)
    print(f"\nStap 5: VIF berekend op rank-getransformeerde features.")
    print(vif_df.to_string(index=False))

    # --- Stap 6: flaggen ---
    flagged = vif_df[vif_df["VIF"] > VIF_THRESHOLD]
    print(f"\nStap 6: {len(flagged)} feature(s) met VIF > {VIF_THRESHOLD}:")
    print(flagged.to_string(index=False) if not flagged.empty else "  (geen)")

    print(f"\nAlles opgeslagen in: {OUTPUT_DIR}")
    print("  - spearman_corr_matrix.csv")
    print("  - spearman_corr_heatmap.png")
    print("  - high_corr_pairs.csv")
    print("  - vif_ranked.csv  <- features met hoge VIF zijn kandidaten om te droppen")


if __name__ == "__main__":
    main()