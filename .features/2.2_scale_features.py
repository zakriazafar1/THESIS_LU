"""
=============================================================================
2.2_scale_features.py

Standaardisatie (z-score, StandardScaler) van de getransformeerde
arousal-featurematrix (output van 2.1_transform_features.py).

Waarom schalen:
  De features hebben na 2.1 nog verschillende eenheden en spreidingen
  (ln-seconden, ln-ratio's, ongetransformeerde motion_rms/oxy_amp_ratio).
  PCA en HDBSCAN zijn variantie- resp. afstandsgebaseerd: zonder schalen zou
  een feature met een grote spreiding domineren puur door zijn schaal.
  Na z-scoring heeft elke feature gemiddelde 0 en SD 1, zodat PCA op de
  correlatiematrix werkt en elke feature even zwaar meetelt in afstanden.

Waarom StandardScaler (en niet meer RobustScaler):
  Na de ln-transformatie in 2.1 is de scheefheid grotendeels weg
  (|skew| < ~1.4), dus mean/SD zijn representatief. z-scores zijn de
  standaard vóór PCA en eenvoudig te rapporteren.

Scaling gebeurt over ALLE events tegelijk (geen train/test-split, dus geen
leakage-probleem). Per-nacht-normalisatie zit al in de spectrale features,
dus er wordt NIET per proefpersoon/nacht geschaald.

Stappenplan:
  1. Getransformeerde featurematrix inladen (arousal_feature_matrix_transformed.csv).
  2. StandardScaler per feature fitten en toepassen (mean -> 0, SD -> 1).
     Per feature worden mean en SD weggeschreven (scaler_params.csv), zodat
     geschaalde waarden later terug te rekenen zijn: x = z * sd + mean.
  3. Controle (mean ~ 0, SD ~ 1) per feature -> scaling_check.csv,
     + histogram-grid na scaling.
  4. Geschaalde featurematrix wegschrijven.

Gebruik:
  python 2.2_scale_features.py
  python 2.2_scale_features.py --input <pad> --output-dir <map>
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATIE
# =============================================================================

# Map waar 2.1_transform_features.py de getransformeerde featurematrix neerzet.
INPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\transformed"
)

# Map waar de output van dit script naartoe gaat.
OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled"
)

DEFAULT_INPUT = INPUT_DIR / "arousal_feature_matrix_transformed.csv"

# Zelfde metadata-kolommen als in 2.1_transform_features.py -- worden niet geschaald.
METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]

N_COLS_GRID = 5  # aantal subplots per rij in de histogram-grid

# =============================================================================
# STAP 1 - INLADEN
# =============================================================================

def load_transformed_matrix(path: Path) -> pd.DataFrame:
    """
    Leest de getransformeerde featurematrix in. Detecteert het scheidingsteken
    automatisch, en leest opnieuw in met decimal="," als numerieke kolommen
    als tekst binnenkomen (Excel-NL-scenario). Zelfde logica als in 2.1.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} bestaat niet -- run eerst 2.1_transform_features.py, of geef het juiste "
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

    print(f"Getransformeerde featurematrix geladen: {path}")
    print(f"Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Alle kolommen behalve de metadata-kolommen -> dit zijn de te schalen features."""
    return [c for c in df.columns if c not in METADATA_COLS]


# =============================================================================
# STAP 2 - STANDARDSCALER
# =============================================================================

def scale_with_standard_scaler(df: pd.DataFrame, feature_cols: list[str]
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    z-score per feature: (x - mean) / SD. Per kolom gefit zodat eventuele
    NaN's per feature NaN blijven (sklearn's StandardScaler negeert NaN bij
    het fitten, maar per kolom houden we ook n_used per feature bij).
    Let op: sklearn gebruikt de populatie-SD (ddof=0).
    """
    out = df.copy()
    rows = []
    for col in feature_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        mask = vals.notna()

        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(vals[mask].to_numpy().reshape(-1, 1)).ravel()

        out[col] = np.nan
        out.loc[mask, col] = scaled_vals
        rows.append({
            "feature": col,
            "mean": float(scaler.mean_[0]),
            "sd": float(scaler.scale_[0]),
            "n_used": int(mask.sum()),
            "n_missing_skipped": int((~mask).sum()),
        })
    return out, pd.DataFrame(rows)


def check_scaling(df_scaled: pd.DataFrame, feature_cols: list[str], tol: float = 1e-6) -> pd.DataFrame:
    """
    Controle: na z-scoring moet elke feature mean ~ 0 en SD ~ 1 hebben.
    Geeft per feature de mean en SD NA scaling terug (ddof=0, zoals sklearn).
    """
    means = df_scaled[feature_cols].mean()
    sds = df_scaled[feature_cols].std(ddof=0)
    check = pd.DataFrame({
        "feature": feature_cols,
        "mean_after": means.round(6).to_numpy(),
        "sd_after": sds.round(6).to_numpy(),
    })
    check["mean_after"] = check["mean_after"].replace(-0.0, 0.0)
    check["ok"] = (means.abs() <= tol).to_numpy() & ((sds - 1).abs() <= tol).to_numpy()

    bad = check.loc[~check["ok"], "feature"].tolist()
    if bad:
        print(f"[LET OP] mean/SD wijkt af van 0/1 voor: {bad}")
    else:
        print(f"Controle OK: alle {len(feature_cols)} features hebben mean = 0 en SD = 1.")
    return check


# =============================================================================
# STAP 3 - PLOT
# =============================================================================

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
# HOOFDLOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Pad naar arousal_feature_matrix_transformed.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Map voor de output")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_transformed_matrix(args.input)
    feature_cols = get_feature_columns(df)
    print(f"\n{len(feature_cols)} features (metadata-kolommen uitgesloten): {feature_cols}")

    # --- Stap 2: StandardScaler ---
    df_scaled, scaler_params = scale_with_standard_scaler(df, feature_cols)
    scaler_params.to_csv(output_dir / "scaler_params.csv", index=False)
    print(f"\nStandardScaler toegepast op alle {len(feature_cols)} features (mean -> 0, SD -> 1).")
    print(scaler_params.round(4).to_string(index=False))

    # --- Stap 3: controle + plot ---
    scaling_check = check_scaling(df_scaled, feature_cols)
    scaling_check.to_csv(output_dir / "scaling_check.csv", index=False)
    plot_distributions(df_scaled, feature_cols, output_dir / "feature_distributions_scaled.png",
                       title_suffix=" -- na StandardScaler (z-scores)")

    # --- Stap 4: wegschrijven ---
    df_scaled.to_csv(output_dir / "arousal_feature_matrix_scaled.csv", index=False)

    print(f"\nOpgeslagen in: {output_dir}")
    print("  - scaler_params.csv          (mean/SD vóór scaling, gebruikt voor de z-scores)")
    print("  - scaling_check.csv          (mean/SD NA scaling, moet 0/1 zijn)")
    print("  - feature_distributions_scaled.png")
    print("  - arousal_feature_matrix_scaled.csv  <- input voor feature-reductie / PCA")


if __name__ == "__main__":
    main()
    