"""
=============================================================================
pca_reduction.py   (stap 2 van 2)

Tweede helft van het vorige pca_features.py, losgetrokken: alleen de PCA /
dimensionality reduction. Draai dit NA correlation_selection.py, op de
reduced_features_events.csv die dat script wegschrijft (default hieronder)
— maar het werkt op elke featurematrix-csv met dezelfde ID-kolommen, dus je
kan er ook de volledige (niet-gededupliceerde) featurematrix in gooien als
je dat liever wil.

Stappen:
  1. Missing-check + mediaan-imputatie (nodig, want sklearn's PCA kan niet
     met NaN's overweg). Als correlation_selection.py al gedraaid heeft
     zou dit in principe niets meer hoeven te doen.
  2. Standaardisatie (mean=0, std=1) — nodig want features staan op heel
     verschillende schalen: ratio's rond 1, motion_rms, ridge_drift in
     Hz/sec, duration_sec in seconden, etc. Zonder dit domineren features
     met toevallig een grotere schaal de eerste componenten.
  3. PCA: scree plot (verklaarde variantie per component + cumulatief),
     loadings per component (welke originele features wegen het zwaarst
     mee, handig om een component te kunnen duiden), en de
     PCA-getransformeerde featurematrix zelf.

BELANGRIJK — nog te verifiëren aannames:
  - INPUT_MATRIX default = output van correlation_selection.py. Pas aan
    (of geef --input mee) als je een ander bestand wil gebruiken.
  - subject_id / group / night_id / event_idx / start_sec / end_sec worden
    NOOIT als PCA-feature meegenomen (identifiers/positie). Alle andere
    kolommen in het input-bestand worden als feature behandeld — als je
    dus de volledige (niet-gededupliceerde) featurematrix invoert, doet
    PCA impliciet ook de "correlatie-reductie" (want componenten zijn per
    definitie orthogonaal), maar dan verlies je de directe interpreteer-
    baarheid die de gededupliceerde features wel geven.
  - Draai eerst met --inspect om te zien welke features meegaan en hoeveel
    missing data er nog is, voordat je de volledige run doet.

Gebruik:
  python correlation_selection.py                           # stap 1 eerst
  python pca_reduction.py --inspect                         # dan hier eerst checken
  python pca_reduction.py                                   # volledige PCA-run
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")  # geen display nodig, alleen wegschrijven naar png
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATIE
# =============================================================================

FEATURE_SELECTION_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\3. feature selection\correlation"
)
# Default input = output van correlation_selection.py
INPUT_MATRIX = FEATURE_SELECTION_DIR / "reduced_features_events.csv"
OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\3. feature selection\pca"
)

# Kolommen die NOOIT als PCA-feature meedoen (identifiers / positie)
ID_COLS = ["subject_id", "group", "night_id", "event_idx", "start_sec", "end_sec"]

PCA_VARIANCE_TARGET = 0.90    # hoeveel cumulatieve verklaarde variantie we willen dekken

pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)


# =============================================================================
# SECTIE 1 — INLADEN
# =============================================================================

def load_feature_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Featurematrix niet gevonden op {path}. Heb je correlation_selection.py "
            f"al gedraaid, of moet je --input naar een ander bestand wijzen?"
        )
    # sep=None + engine="python" detecteert automatisch komma of puntkomma
    # (Excel schrijft bij een NL-locale soms puntkomma-gescheiden CSV's terug
    # als het bestand tussentijds geopend/opgeslagen is).
    df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[1] == 1:
        raise ValueError(
            f"Alleen 1 kolom ingelezen uit {path} — de separator-detectie is "
            f"waarschijnlijk misgegaan of het bestand is corrupt. Kolomnaam: "
            f"{df.columns[0]!r}. Check het bestand handmatig (bv. in een teksteditor, "
            f"niet Excel) voordat je verder gaat."
        )
    df.columns = [c.strip() for c in df.columns]
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS]


# =============================================================================
# SECTIE 2 — PCA
# =============================================================================

def run_pca(df: pd.DataFrame, feature_cols: list[str]):
    """
    Imputeert missing values (mediaan per feature) en standaardiseert
    (mean=0, std=1), en draait PCA. Geeft (pca_model, components_df,
    loadings_df) terug.
    """
    X_raw = df[feature_cols].values

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    components_df = pd.DataFrame(
        X_pca, index=df.index,
        columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
    )
    loadings_df = pd.DataFrame(
        pca.components_.T, index=feature_cols,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
    )
    return pca, components_df, loadings_df


def n_components_for_variance(pca: PCA, target: float) -> int:
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumvar, target) + 1)


def plot_scree(pca: PCA, out_path: Path):
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
            alpha=0.6, label="per component")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Verklaarde variantie (per component)")
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cumvar) + 1), cumvar, color="black", marker="o", markersize=3,
              label="cumulatief")
    ax2.axhline(PCA_VARIANCE_TARGET, color="grey", linestyle="--", linewidth=1)
    ax2.set_ylabel("Cumulatieve verklaarde variantie")
    ax1.set_title("Scree plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loadings_top(loadings_df: pd.DataFrame, out_path: Path, n_pcs: int = 3, top_n: int = 12):
    n_pcs = min(n_pcs, loadings_df.shape[1])
    fig, axes = plt.subplots(1, n_pcs, figsize=(5 * n_pcs, 6), sharey=False)
    if n_pcs == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        pc = f"PC{i+1}"
        top = loadings_df[pc].reindex(loadings_df[pc].abs().sort_values(ascending=False).index)[:top_n]
        ax.barh(top.index[::-1], top.values[::-1])
        ax.set_title(pc)
        ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# SECTIE 3 — HOOFDLOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                         help="Alleen tonen welke features meegaan + missing-check, geen PCA/bestanden wegschrijven")
    parser.add_argument("--input", type=Path, default=INPUT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--variance-target", type=float, default=PCA_VARIANCE_TARGET)
    args = parser.parse_args()

    df = load_feature_matrix(args.input)
    feature_cols = get_feature_cols(df)
    print(f"Featurematrix geladen: {df.shape[0]} events, {len(feature_cols)} features")
    print(f"Features: {feature_cols}")

    missing_frac = df[feature_cols].isna().mean().sort_values(ascending=False)
    n_missing = (missing_frac > 0).sum()
    print(f"\n{n_missing} features met missing data (worden mediaan-geïmputeerd):")
    if n_missing:
        print(missing_frac[missing_frac > 0])

    if args.inspect:
        print("\nInspectie klaar. Check of dit de features zijn die je verwacht "
              "(anders --input aanpassen), voordat je de volledige PCA-run doet.")
        return

    pca, components_df, loadings_df = run_pca(df, feature_cols)
    n_pcs_target = n_components_for_variance(pca, args.variance_target)
    print(f"\n--- PCA ---")
    print(f"{n_pcs_target} componenten nodig voor >= {args.variance_target:.0%} verklaarde variantie "
          f"(van {len(feature_cols)} features)")
    print("Verklaarde variantie per component (eerste 10):")
    print(np.round(pca.explained_variance_ratio_[:10], 3))

    # --- Output wegschrijven ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loadings_df.to_csv(args.output_dir / "pca_loadings.csv")

    pca_out = pd.concat([df[[c for c in ID_COLS if c in df.columns]].reset_index(drop=True),
                          components_df.reset_index(drop=True)], axis=1)
    pca_out.to_csv(args.output_dir / "pca_transformed_events.csv", index=False, float_format="%.4f")

    plot_scree(pca, args.output_dir / "pca_scree.png")
    plot_loadings_top(loadings_df, args.output_dir / "pca_top_loadings.png")

    print(f"\nAlle output weggeschreven naar: {args.output_dir}")
    print("  - pca_transformed_events.csv -> PCA-componenten, te gebruiken als clustering-input")
    print("  - pca_loadings.csv / pca_scree.png / pca_top_loadings.png")


if __name__ == "__main__":
    main()