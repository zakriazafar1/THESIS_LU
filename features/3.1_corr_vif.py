"""
=============================================================================
3.1_corr_vif.py

Correlatie- en multicollineariteit-check op de geschaalde arousal-featurematrix
(output van 2.2_scale_features.py), als check tussen scaling en UMAP/HDBSCAN.

Hoofdanalyse: Pearson op de geschaalde matrix
  De geschaalde matrix (na transformatie + scaling) is precies wat UMAP/HDBSCAN
  te zien krijgt. Pearson op die matrix vertelt dus welke features in de
  afstandsberekening dubbel tellen. Pearson is ongevoelig voor lineaire
  herschaling, dus Pearson op 'scaled' == Pearson op 'transformed'.

Controle: Spearman op dezelfde matrix
  Spearman is rank-based en daarmee ongevoelig voor elke monotone transformatie
  (log, Box-Cox, Yeo-Johnson, scaling). Spearman op 'scaled' == Spearman op
  'raw', dus we hoeven de ruwe matrix niet apart in te laden.
  Paren waar Pearson en Spearman sterk verschillen wijzen op outliers of
  niet-lineaire (maar monotone) samenhang -> even visueel checken.

VIF (Pearson-based, passend bij de hoofdanalyse)
  VIF vangt multicollineariteit over meerdere features tegelijk. Berekend op de
  geschaalde waarden MET een constante (intercept) in het model; zonder
  constante geeft statsmodels' variance_inflation_factor een ongecentreerde,
  vertekende VIF. Na StandardScaler zijn de features al mean-gecentreerd, maar
  na het droppen van rijen met NaN geldt dat niet meer exact -- de constante
  houdt de VIF dan correct.

Stappenplan:
  1. Geschaalde featurematrix inladen (arousal_feature_matrix_scaled.csv).
  2. Pearson-correlatiematrix (hoofd) + Spearman-correlatiematrix (controle).
  3. Heatmaps van beide.
  4. Sterk gecorreleerde paren (|r| > CORR_THRESHOLD, Pearson), met Spearman erbij.
  5. Paren waar |r - rho| > DIFF_THRESHOLD (Pearson vs Spearman wijken af).
  6. VIF op de geschaalde features (met intercept) + flaggen > VIF_THRESHOLD.

Gebruik:
  python 3.1_corr_vif.py
  python 3.1_corr_vif.py --input pad/naar/andere_scaled.csv
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# =============================================================================
# CONFIGURATIE
# =============================================================================

INPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled"
)

OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\3. feature selection\corr_vif"
)

DEFAULT_INPUT = INPUT_DIR / "arousal_feature_matrix_scaled.csv"

METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]

CORR_THRESHOLD = 0.80   # |r| boven deze grens = "sterk gecorreleerd"
DIFF_THRESHOLD = 0.15   # |r - rho| boven deze grens = Pearson en Spearman wijken af
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
            "bestand is beschadigd."
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
                  f"decimaal-komma gedetecteerd, opnieuw ingelezen met decimal=','.")
            df = df_comma_decimal

    print(f"Geschaalde featurematrix geladen: {path}")
    print(f"Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Alle kolommen behalve de metadata-kolommen -> dit zijn de te checken features."""
    return [c for c in df.columns if c not in METADATA_COLS]


# =============================================================================
# SECTIE 2 — CORRELATIEMATRICES
# =============================================================================

def compute_corr(df: pd.DataFrame, feature_cols: list[str], method: str) -> pd.DataFrame:
    """Correlatiematrix ('pearson' of 'spearman'), pairwise, met NaN/inf als missing."""
    sub = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return sub.corr(method=method)


def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path, label: str) -> None:
    """Heatmap van een correlatiematrix, geannoteerd als het leesbaar blijft."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.45 * n + 2))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        annot=n <= 20,
        fmt=".2f",
        annot_kws={"size": 6},
        cbar_kws={"label": label},
        ax=ax,
    )
    ax.set_title(f"{label} (geschaalde features)", fontsize=12)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap opgeslagen: {out_path}")


def upper_triangle_pairs(pearson: pd.DataFrame, spearman: pd.DataFrame) -> pd.DataFrame:
    """Alle unieke feature-paren met Pearson r, Spearman rho en hun verschil."""
    cols = pearson.columns
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = pearson.iloc[i, j]
            rho = spearman.iloc[i, j]
            rows.append({
                "feature_1": cols[i],
                "feature_2": cols[j],
                "pearson_r": r,
                "spearman_rho": rho,
                "diff_r_minus_rho": r - rho if pd.notna(r) and pd.notna(rho) else np.nan,
            })
    return pd.DataFrame(rows)


def list_high_corr_pairs(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Paren met |Pearson r| > threshold, gesorteerd van hoog naar laag."""
    out = pairs[pairs["pearson_r"].abs() > threshold].copy()
    out = out.reindex(out["pearson_r"].abs().sort_values(ascending=False).index)
    return out.round(4).reset_index(drop=True)


def list_divergent_pairs(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Paren waar Pearson en Spearman meer dan `threshold` van elkaar verschillen."""
    out = pairs[pairs["diff_r_minus_rho"].abs() > threshold].copy()
    out = out.reindex(out["diff_r_minus_rho"].abs().sort_values(ascending=False).index)
    return out.round(4).reset_index(drop=True)


# =============================================================================
# SECTIE 3 — VIF OP GESCHAALDE FEATURES (MET INTERCEPT)
# =============================================================================

def compute_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    VIF per feature op de geschaalde waarden, met een constante in het model.
    Rijen met NaN/inf in minstens één feature worden alleen voor deze check
    uitgesloten (OLS kan niet met missing values omgaan).
    """
    sub = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    n_before = len(sub)
    sub = sub.dropna()
    n_dropped = n_before - len(sub)
    if n_dropped > 0:
        print(f"[LET OP] {n_dropped} rij(en) met NaN/inf uitgesloten voor VIF-berekening "
              f"({len(sub)} van {n_before} rijen gebruikt).")

    X = add_constant(sub, has_constant="add").to_numpy()
    # kolom 0 is de constante -> features beginnen bij index 1
    rows = []
    for i, col in enumerate(feature_cols, start=1):
        vif_val = variance_inflation_factor(X, i)
        rows.append({"feature": col, "VIF": round(float(vif_val), 3)})

    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


# =============================================================================
# HOOFDLOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Pad naar arousal_feature_matrix_scaled.csv")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scaled_matrix(args.input)
    feature_cols = get_feature_columns(df)
    print(f"\n{len(feature_cols)} features (metadata-kolommen uitgesloten): {feature_cols}")

    # --- Stap 2+3: correlatiematrices + heatmaps ---
    pearson = compute_corr(df, feature_cols, "pearson")
    spearman = compute_corr(df, feature_cols, "spearman")
    pearson.to_csv(OUTPUT_DIR / "pearson_corr_matrix.csv")
    spearman.to_csv(OUTPUT_DIR / "spearman_corr_matrix.csv")
    plot_corr_heatmap(pearson, OUTPUT_DIR / "pearson_corr_heatmap.png", "Pearson r")
    plot_corr_heatmap(spearman, OUTPUT_DIR / "spearman_corr_heatmap.png", "Spearman rho")

    pairs = upper_triangle_pairs(pearson, spearman)

    # --- Stap 4: sterk gecorreleerde paren (Pearson) ---
    high_corr = list_high_corr_pairs(pairs, CORR_THRESHOLD)
    high_corr.to_csv(OUTPUT_DIR / "high_corr_pairs.csv", index=False)
    print(f"\nStap 4: {len(high_corr)} paar/paren met |r| > {CORR_THRESHOLD} (Pearson):")
    print(high_corr.to_string(index=False) if not high_corr.empty else "  (geen)")

    # --- Stap 5: Pearson vs Spearman wijken af ---
    divergent = list_divergent_pairs(pairs, DIFF_THRESHOLD)
    divergent.to_csv(OUTPUT_DIR / "pearson_vs_spearman_divergent.csv", index=False)
    print(f"\nStap 5: {len(divergent)} paar/paren met |r - rho| > {DIFF_THRESHOLD}:")
    print(divergent.to_string(index=False) if not divergent.empty else "  (geen)")

    # --- Stap 6: VIF + flaggen ---
    vif_df = compute_vif(df, feature_cols)
    vif_df.to_csv(OUTPUT_DIR / "vif.csv", index=False)
    print(f"\nStap 6: VIF op geschaalde features (met intercept):")
    print(vif_df.to_string(index=False))

    flagged = vif_df[vif_df["VIF"] > VIF_THRESHOLD]
    print(f"\n{len(flagged)} feature(s) met VIF > {VIF_THRESHOLD}:")
    print(flagged.to_string(index=False) if not flagged.empty else "  (geen)")

    print(f"\nAlles opgeslagen in: {OUTPUT_DIR}")
    print("  - pearson_corr_matrix.csv / pearson_corr_heatmap.png   <- hoofdanalyse")
    print("  - spearman_corr_matrix.csv / spearman_corr_heatmap.png <- controle")
    print("  - high_corr_pairs.csv                <- redundante paren (Pearson, met rho erbij)")
    print("  - pearson_vs_spearman_divergent.csv  <- paren om visueel te checken (scatterplot)")
    print("  - vif.csv                            <- hoge VIF = kandidaat om te droppen")


if __name__ == "__main__":
    main()