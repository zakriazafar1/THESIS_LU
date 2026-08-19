"""
=============================================================================
3_scale_features.py

Standaardisatie (RobustScaler) van de getransformeerde arousal-featurematrix
(output van 2_transform_features.py. 

Waarom RobustScaler i.p.v. StandardScaler:
  RobustScaler schaalt op basis van mediaan en IQR (25e-75e percentiel) i.p.v.
  mean/std. Ook na de log1p/sqrt-transformatie in stap 2 kunnen er nog wat
  resterende extremen in de staarten zitten (er is in dit stappenplan geen
  aparte outlier-winsorize/remove-stap toegepast) -- RobustScaler is daar
  minder gevoelig voor dan StandardScaler, en dat is hier voor ALLE features
  consistent toegepast (dus niet per-feature gekozen o.b.v. skew, zoals in de
  eerdere versie van stap 5 in 2_transform_features.py).

Stappenplan:
  1. Featurematrix inladen (arousal_feature_matrix_transformed.csv).
  2. Distributie vóór scaling visualiseren + skew/kurtosis wegschrijven.
  3. RobustScaler fitten en toepassen op alle features (median=0, IQR=1 na
     scaling). Per feature wordt de gebruikte mediaan en IQR weggeschreven
     (scaler_summary.csv), zodat de scaling reproduceerbaar/na te rekenen is.
  4. Distributie NA scaling visualiseren + skew/kurtosis wegschrijven.
  5. Geschaalde featurematrix wegschrijven.

Gebruik:
  python 3_scale_features.py
      -> leest arousal_feature_matrix_transformed.csv uit OUTPUT_DIR (zie
         onder), schrijft alles terug naar dezelfde map.
  python 3_scale_features.py --input pad/naar/andere_transformed.csv
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import RobustScaler

# =============================================================================
# CONFIGURATIE
# =============================================================================

# Zelfde map als waar 2_transform_features.py zijn output neerzet.
OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing"
)

DEFAULT_INPUT = OUTPUT_DIR / "arousal_feature_matrix_transformed.csv"

# Zelfde metadata-kolommen als in 2_transform_features.py -- deze worden
# uitgesloten van scaling, en gewoon meegekopieerd naar de output.
METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]

N_COLS_GRID = 5  # aantal subplots per rij in de histogram-grid


# =============================================================================
# SECTIE 1 — INLADEN
# =============================================================================

def load_transformed_matrix(path: Path) -> pd.DataFrame:
    """Leest de getransformeerde featurematrix in (standaard-CSV, door 2_transform_features.py weggeschreven)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} bestaat niet -- run eerst 2_transform_features.py, of geef het juiste "
            "pad mee met --input."
        )
    df = pd.read_csv(path)
    print(f"Getransformeerde featurematrix geladen: {path}")
    print(f"Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Alle kolommen behalve de metadata-kolommen -> dit zijn de te scalen features."""
    return [c for c in df.columns if c not in METADATA_COLS]


# =============================================================================
# SECTIE 2 — DISTRIBUTIE-STATS EN PLOTS (voor en na, zelfde functies)
# =============================================================================

def compute_distribution_stats(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Skewness en kurtosis per feature (op de niet-NaN, eindige waarden)."""
    rows = []
    for col in feature_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) < 3:
            rows.append({"feature": col, "skew": np.nan, "kurtosis": np.nan,
                          "min": np.nan, "max": np.nan})
            continue
        rows.append({
            "feature": col,
            "skew": round(skew(vals), 3),
            "kurtosis": round(kurtosis(vals), 3),
            "min": round(vals.min(), 3),
            "max": round(vals.max(), 3),
        })
    stats = pd.DataFrame(rows)
    return stats.sort_values("skew", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def plot_distributions(df: pd.DataFrame, feature_cols: list[str], out_path: Path, title_suffix: str = "") -> None:
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

    fig.suptitle(f"Feature distributions{title_suffix}", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Distributie-grid opgeslagen: {out_path}")


# =============================================================================
# SECTIE 3 — ROBUSTSCALER (STAP 3)
# =============================================================================

def scale_with_robust_scaler(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Fit een sklearn RobustScaler (default: median + IQR 25-75) op alle
    features tegelijk, en past 'm toe. NaN/inf worden vóór het fitten naar
    NaN gezet zodat ze de mediaan/IQR-berekening niet verstoren; sklearn's
    RobustScaler kan zelf niet met NaN omgaan, dus we scalen kolom-voor-kolom
    zodat missende waarden per feature NaN blijven (i.p.v. de hele rij te
    moeten droppen).
    """
    out = df.copy()
    rows = []
    for col in feature_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        mask = vals.notna()

        scaler = RobustScaler()  # default: quantile_range=(25.0, 75.0), with_centering/scaling=True
        scaled_vals = scaler.fit_transform(vals[mask].to_numpy().reshape(-1, 1)).ravel()

        out.loc[mask, col] = scaled_vals
        rows.append({
            "feature": col,
            "median_": round(float(scaler.center_[0]), 4),
            "iqr_": round(float(scaler.scale_[0]), 4),
            "n_used": int(mask.sum()),
            "n_missing_skipped": int((~mask).sum()),
        })

    scaler_summary = pd.DataFrame(rows)
    # Eén scaler-object teruggeven is niet zinvol bij per-kolom fitten, dus
    # geven we None terug -- alle info staat al in scaler_summary (median_/iqr_
    # per feature is genoeg om de scaling elders te reproduceren).
    return out, scaler_summary, None


# =============================================================================
# HOOFDLOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help="Pad naar arousal_feature_matrix_transformed.csv")
    args = parser.parse_args()

    df = load_transformed_matrix(args.input)
    feature_cols = get_feature_columns(df)
    print(f"\n{len(feature_cols)} features (metadata-kolommen uitgesloten): {feature_cols}")

    # --- Stap 2: vóór scaling ---
    stats_before = compute_distribution_stats(df, feature_cols)
    stats_before.to_csv(OUTPUT_DIR / "distribution_stats_before_scaling.csv", index=False)
    plot_distributions(df, feature_cols, OUTPUT_DIR / "feature_distributions_before_scaling.png",
                        title_suffix=" -- vóór RobustScaler")

    # --- Stap 3: RobustScaler ---
    df_scaled, scaler_summary, _ = scale_with_robust_scaler(df, feature_cols)
    scaler_summary.to_csv(OUTPUT_DIR / "scaler_summary.csv", index=False)
    print(f"\nStap 3: RobustScaler toegepast op alle {len(feature_cols)} features "
          f"(median -> 0, IQR -> 1).")
    print(scaler_summary.to_string(index=False))

    # --- Stap 4: na scaling ---
    stats_after = compute_distribution_stats(df_scaled, feature_cols)
    stats_after.to_csv(OUTPUT_DIR / "distribution_stats_after_scaling.csv", index=False)
    plot_distributions(df_scaled, feature_cols, OUTPUT_DIR / "feature_distributions_after_scaling.png",
                        title_suffix=" -- na RobustScaler")

    # --- Stap 5: wegschrijven ---
    df_scaled.to_csv(OUTPUT_DIR / "arousal_feature_matrix_scaled.csv", index=False)

    print(f"\nAlles opgeslagen in: {OUTPUT_DIR}")
    print("  - distribution_stats_before_scaling.csv")
    print("  - feature_distributions_before_scaling.png")
    print("  - scaler_summary.csv")
    print("  - distribution_stats_after_scaling.csv")
    print("  - feature_distributions_after_scaling.png")
    print("  - arousal_feature_matrix_scaled.csv  <- klaar voor clustering")


if __name__ == "__main__":
    main()