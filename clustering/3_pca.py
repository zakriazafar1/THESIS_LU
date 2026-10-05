"""
=============================================================================
3_pca.py

PCA op de gereduceerde, geschaalde arousal-featurematrix
(output van 3.3_reduce_features.py: arousal_features_reduced_scaled.csv).

Doel: de correlaties tussen de features (vooral het spectrale blok
theta/alpha/sigma) opheffen vóór HDBSCAN.
  - PCA draait het assenstelsel zodat de componenten ongecorreleerd zijn.
    Een draaiing verandert geen afstanden; het "dubbel tellen" van het
    spectrale blok zit daarna in PC1 (grootste variantie).
  - Met --whiten krijgt elke behouden PC variantie 1, zodat elke onafhankelijke
    dimensie even zwaar meetelt in de afstanden (= Mahalanobis-afstand in de
    gereduceerde ruimte). Alleen verantwoord als de ruis-PC's met hele kleine
    eigenwaarde eerst zijn weggelaten (--n-components), anders worden die
    enorm opgeblazen.

Wat het script geeft:
  1. Alle PC's (ook de niet-behouden): eigenwaarde, % verklaarde variantie,
     cumulatief %  -> explained_variance.csv
  2. Loadings van alle PC's:
       - pca_loadings.csv          (eigenvectoren, gewicht per feature)
       - pca_loading_correlations.csv (correlatie feature <-> PC, = loading *
         sqrt(eigenwaarde); makkelijker te interpreteren: "hoe sterk hangt deze
         feature samen met deze PC")
     + heatmap van de correlaties.
  3. Screeplot: eigenwaarden (met Kaiser-lijn = 1) en cumulatieve verklaarde
     variantie (met 90%/95%-lijnen), het gekozen aantal PC's gemarkeerd.
  4. PC-scores van de behouden componenten (+ metadata), met of zonder
     whitening, als CSV -> input voor HDBSCAN

Gebruik:
  python 3_pca.py                         # 5 PC's, zonder én met whitening
  python 3_pca.py --whiten no             # alleen zonder whitening
  python 3_pca.py --whiten yes            # alleen met whitening
  python 3_pca.py --n-components 4
  python 3_pca.py --n-components 0.95     # zoveel PC's als nodig voor 95%
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# =============================================================================
# CONFIGURATIE
# =============================================================================

BASE = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2"
)
DEFAULT_INPUT = BASE / r"3. feature selection\reduced\arousal_features_reduced_scaled.csv"
OUTPUT_DIR = BASE / r"4. clustering\pca"

METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]

N_COMPONENTS_DEFAULT = "5"   # int = aantal PC's, float < 1 = doel cumulatieve variantie
RANDOM_STATE = 42

BAR_COLOR = "#4C78A8"
REF_COLOR = "#8C8C8C"
KEEP_COLOR = "#C0392B"


# =============================================================================
# SECTIE 1 — INLADEN
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    """CSV met automatische separator-detectie en fallback naar decimal=','."""
    if not path.exists():
        raise FileNotFoundError(f"{path} bestaat niet -- run eerst 3.3_reduce_features.py.")
    df = pd.read_csv(path, sep=None, engine="python")
    text_cols = {"subject_id", "group", "night_id"}
    check = [c for c in df.columns if c not in text_cols]
    n_bad = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in check)
    if n_bad > 0:
        df2 = pd.read_csv(path, sep=None, engine="python", decimal=",")
        n_bad2 = sum(not pd.api.types.is_numeric_dtype(df2[c]) for c in check if c in df2.columns)
        if n_bad2 < n_bad:
            print("[LET OP] decimaal-komma gedetecteerd, opnieuw ingelezen met decimal=','.")
            df = df2
    print(f"Geladen: {path}  shape={df.shape}")
    return df


def parse_n_components(value: str, n_features: int) -> int | float:
    """'5' -> 5 PC's; '0.95' -> doel cumulatieve variantie."""
    v = float(value)
    if 0 < v < 1:
        return v
    k = int(v)
    if not 1 <= k <= n_features:
        raise ValueError(f"--n-components moet tussen 1 en {n_features} liggen (of 0-1 als fractie).")
    return k


# =============================================================================
# SECTIE 2 — PCA OP ALLE COMPONENTEN (eigenwaarden + loadings)
# =============================================================================

def fit_full_pca(X: pd.DataFrame) -> tuple[PCA, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    PCA met alle componenten. Geeft het model + tabellen voor verklaarde variantie,
    loadings (eigenvectoren) en loading-correlaties.
    Tekens van PC's zijn willekeurig; ze worden zo gezet dat de grootste
    absolute loading per PC positief is (alleen voor leesbaarheid).
    """
    pca = PCA(random_state=RANDOM_STATE).fit(X)

    # tekens vastzetten
    for i in range(pca.n_components_):
        if pca.components_[i, np.argmax(np.abs(pca.components_[i]))] < 0:
            pca.components_[i] *= -1

    pcs = [f"PC{i + 1}" for i in range(pca.n_components_)]
    ev = pd.DataFrame({
        "component": pcs,
        "eigenvalue": pca.explained_variance_,
        "pct_variance": 100 * pca.explained_variance_ratio_,
        "cum_pct_variance": 100 * np.cumsum(pca.explained_variance_ratio_),
    }).round(4)

    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=pcs).round(4)
    # correlatie feature <-> PC = eigenvector * sqrt(eigenwaarde) / SD(feature)
    sd = X.std(ddof=1).to_numpy()[:, None]
    corr = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_)[None, :] / sd,
                        index=X.columns, columns=pcs).round(4)
    return pca, ev, loadings, corr


def resolve_k(ev: pd.DataFrame, n_components: int | float) -> int:
    """Aantal te behouden PC's; bij een fractie: kleinste k die die variantie haalt."""
    if isinstance(n_components, float):
        return int(np.searchsorted(ev["cum_pct_variance"].to_numpy(), 100 * n_components - 1e-9) + 1)
    return n_components


# =============================================================================
# SECTIE 3 — PLOTS
# =============================================================================

def plot_scree(ev: pd.DataFrame, k: int, out_path: Path) -> None:
    """
    Twee panelen (geen dubbele y-as):
      links  eigenwaarde per PC + Kaiser-lijn (eigenwaarde = 1)
      rechts cumulatieve verklaarde variantie + 90%/95%-lijnen
    Het gekozen aantal PC's is in beide gemarkeerd.
    """
    x = np.arange(1, len(ev) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    colors = [BAR_COLOR if i < k else "#C9D6E6" for i in range(len(ev))]
    ax1.bar(x, ev["eigenvalue"], color=colors, width=0.6, zorder=2)
    ax1.plot(x, ev["eigenvalue"], color="#2F4B6E", lw=1.5, marker="o", ms=5, zorder=3)
    ax1.axhline(1, color=REF_COLOR, ls="--", lw=1, zorder=1)
    ax1.text(len(ev) + 0.4, 1, "Kaiser (= 1)", va="bottom", ha="right", fontsize=8, color="#555555")
    for xi, val, pct in zip(x, ev["eigenvalue"], ev["pct_variance"]):
        ax1.text(xi, val, f"{pct:.0f}%", ha="center", va="bottom", fontsize=7, color="#333333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ev["component"], fontsize=8)
    ax1.set_ylabel("Eigenwaarde")
    ax1.set_title("Screeplot", fontsize=11)

    ax2.plot(x, ev["cum_pct_variance"], color="#2F4B6E", lw=2, marker="o", ms=6, zorder=3)
    for lvl in (90, 95):
        ax2.axhline(lvl, color=REF_COLOR, ls=":", lw=1, zorder=1)
        ax2.text(0.6, lvl, f"{lvl}%", va="bottom", fontsize=8, color="#555555")
    ax2.scatter([k], [ev["cum_pct_variance"].iloc[k - 1]], s=90, facecolor="none",
                edgecolor=KEEP_COLOR, lw=2, zorder=4)
    ax2.annotate(f"{k} PC's: {ev['cum_pct_variance'].iloc[k - 1]:.1f}%",
                 (k, ev["cum_pct_variance"].iloc[k - 1]), textcoords="offset points",
                 xytext=(8, -16), fontsize=9, color="#333333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ev["component"], fontsize=8)
    ax2.set_ylim(0, 102)
    ax2.set_ylabel("Cumulatieve verklaarde variantie (%)")
    ax2.set_title("Cumulatieve verklaarde variantie", fontsize=11)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#E5E5E5", lw=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(k + 0.5, color=KEEP_COLOR, ls="--", lw=1, alpha=0.7, zorder=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Screeplot opgeslagen: {out_path}")


def plot_loadings(corr: pd.DataFrame, ev: pd.DataFrame, k: int, out_path: Path) -> None:
    """Heatmap van de feature-PC-correlaties voor alle PC's; behouden PC's gemarkeerd."""
    labels = [f"{c}\n{p:.0f}%" for c, p in zip(ev["component"], ev["pct_variance"])]
    fig, ax = plt.subplots(figsize=(0.95 * corr.shape[1] + 3, 0.5 * corr.shape[0] + 2))
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "correlatie feature – PC"}, ax=ax)
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.tick_params(axis="y", labelsize=8)
    ax.axvline(k, color=KEEP_COLOR, lw=2)
    ax.set_title(f"PCA-loadings (correlatie met PC) -- links van de rode lijn: {k} behouden PC's",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Loadings-heatmap opgeslagen: {out_path}")


# =============================================================================
# SECTIE 4 — SCORES (met / zonder whitening)
# =============================================================================

def compute_scores(X: pd.DataFrame, full: PCA, k: int, whiten: bool) -> pd.DataFrame:
    """
    PC-scores van de eerste k componenten. Gebruikt dezelfde (teken-gecorrigeerde)
    eigenvectoren als de volledige PCA, zodat scores en loadings bij elkaar horen.
    whiten=True: scores gedeeld door sqrt(eigenwaarde) -> variantie 1 per PC.
    """
    centered = X.to_numpy() - full.mean_
    scores = centered @ full.components_[:k].T
    if whiten:
        scores = scores / np.sqrt(full.explained_variance_[:k])
    return pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(k)], index=X.index)


# =============================================================================
# HOOFDLOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-components", default=N_COMPONENTS_DEFAULT,
                        help="Aantal PC's (bijv. 5) of doel-fractie variantie (bijv. 0.95)")
    parser.add_argument("--whiten", choices=["no", "yes", "both"], default="both",
                        help="Scores zonder whitening, met, of allebei (default)")
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_robust(args.input)
    meta = [c for c in METADATA_COLS if c in df.columns]
    features = [c for c in df.columns if c not in METADATA_COLS]
    X = df[features].replace([np.inf, -np.inf], np.nan)
    n_nan = int(X.isna().any(axis=1).sum())
    if n_nan:
        print(f"[LET OP] {n_nan} rij(en) met NaN/inf uitgesloten voor PCA.")
        keep = ~X.isna().any(axis=1)
        X, df = X[keep], df[keep]
    print(f"{len(features)} features: {features}")

    # --- Stap 1+2: volledige PCA ---
    full, ev, loadings, corr = fit_full_pca(X)
    k = resolve_k(ev, parse_n_components(args.n_components, len(features)))

    ev.to_csv(out_dir / "explained_variance.csv", index=False)
    loadings.to_csv(out_dir / "pca_loadings.csv")
    corr.to_csv(out_dir / "pca_loading_correlations.csv")

    print("\nAlle componenten:")
    print(ev.to_string(index=False))
    n_kaiser = int((ev["eigenvalue"] > 1).sum())
    print(f"\nKaiser-criterium (eigenwaarde > 1): {n_kaiser} PC('s).  "
          f"Gekozen: {k} PC's = {ev['cum_pct_variance'].iloc[k - 1]:.1f}% variantie.")
    print("\nLoadings als correlatie feature <-> PC (alle PC's):")
    print(corr.round(2).to_string())

    # --- Stap 3: plots ---
    plot_scree(ev, k, out_dir / "pca_screeplot.png")
    plot_loadings(corr, ev, k, out_dir / "pca_loadings_heatmap.png")

    # --- Stap 4: scores ---
    modes = {"no": [False], "yes": [True], "both": [False, True]}[args.whiten]
    print()
    for w in modes:
        scores = compute_scores(X, full, k, w)
        out = pd.concat([df[meta].reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        name = f"pca_scores_{k}pc_{'whitened' if w else 'unwhitened'}.csv"
        out.to_csv(out_dir / name, index=False)
        var = scores.var(ddof=1).round(2).to_dict()
        print(f"Scores {'MET' if w else 'ZONDER'} whitening -> {name}")
        print(f"  variantie per PC: {var}")

    print(f"\nAlles opgeslagen in: {out_dir}")
    print("  - explained_variance.csv               <- eigenwaarde / % / cumulatief % per PC")
    print("  - pca_loadings.csv                     <- eigenvectoren")
    print("  - pca_loading_correlations.csv         <- correlatie feature <-> PC (interpretatie)")
    print("  - pca_screeplot.png / pca_loadings_heatmap.png")
    print("  - pca_scores_*pc_(un)whitened.csv      <- input voor HDBSCAN")


if __name__ == "__main__":
    main()