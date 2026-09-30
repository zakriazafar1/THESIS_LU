"""
=============================================================================
2_pca_features.py

Correlatie- en PCA-analyse van de arousal-featurematrix uit 1_feature_matrix.py.

Wat dit script doet:
  1. Laadt de featurematrix (standaard: arousal_feature_matrix_ORIGIN.csv)
  2. Selecteert alleen de echte features (geen ID's, tijden of stage)
  3. Transformeert: log1p + schalen  (overslaan met --already-scaled als je
     een bestand inlaadt dat al getransformeerd en geschaald is)
  4. Spearman-correlatiematrix + lijst van sterk gecorreleerde paren
  5. PCA: scree-plot, cumulatieve verklaarde variantie, loadings, PC1 vs PC2

Waarom PCA op de getransformeerde + geschaalde features (en niet op ruw):
  PCA zoekt richtingen met maximale VARIANTIE. Op ruwe data domineren dan
  de features met de grootste getallen/eenheden (motion_rms in g,
  sec_prev_event in seconden, scheve peak_ratio's met uitschieters), niet de
  features die inhoudelijk het meest samenhangen. Na log1p + schalen draagt
  elke feature ongeveer even zwaar bij, en meet PCA echt de samenhang.

  Let op: RobustScaler (mediaan/IQR) geeft GEEN eenheidsvariantie. Features
  met een brede staart houden dan meer gewicht. Met SCALER = "standard" is PCA
  exact equivalent aan PCA op de correlatiematrix — dat is de gangbare keuze
  als je vraag "hoe hangen de features samen" is. Beide opties zitten erin.

Gebruik:
  python 2_pca_features.py                                   # ORIGIN, log1p + StandardScaler
  python 2_pca_features.py --scaler robust                   # log1p + RobustScaler
  python 2_pca_features.py --input pad/naar/scaled.csv --already-scaled
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

# =============================================================================
# CONFIGURATIE
# =============================================================================

FM_DIR    = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices")
INPUT_CSV = FM_DIR / "arousal_feature_matrix_ORIGIN.csv"
OUT_DIR   = FM_DIR.parent / "2. PCA"

BANDS = ["delta", "theta", "alpha", "sigma", "beta"]
CHANNEL_LABELS = ["L", "R"]

# mean_{band}_ratio is het gemiddelde van L en R -> puur afgeleid, dus
# redundant. Standaard NIET meenemen, anders blaas je PC1 kunstmatig op.
INCLUDE_MEAN_COLS = False
INCLUDE_PEAK_RATIOS = True

# Niet-EEG features. stage_rk is categorisch -> niet in de PCA, wel als kleur.
EXTRA_FEATURES = ["duration_sec", "sec_prev_event", "motion_rms", "oxy_amp_ratio"]

META_COLS = ["subject_id", "group", "night_id", "stage_rk", "event_idx",
             "start_sec", "end_sec"]

MAX_NAN_FRAC = 0.30     # features met meer NaN dan dit worden eruit gegooid
HIGH_CORR = 0.80        # drempel voor "sterk gecorreleerd" paar (|rho|)
N_LOADING_PCS = 6       # aantal PC's in de loadings-heatmap

RK_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "N4", 5: "REM"}


# =============================================================================
# FEATURES SELECTEREN EN TRANSFORMEREN
# =============================================================================

def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for band in BANDS:
        for lab in CHANNEL_LABELS:
            cols.append(f"{lab}_{band}_ratio")
            if INCLUDE_PEAK_RATIOS:
                cols.append(f"{lab}_{band}_peak_ratio")
        if INCLUDE_MEAN_COLS:
            cols.append(f"mean_{band}_ratio")
    cols += EXTRA_FEATURES
    return [c for c in cols if c in df.columns]


def select_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = feature_columns(df)
    X = df[cols].apply(pd.to_numeric, errors="coerce")

    nan_frac = X.isna().mean()
    dropped = nan_frac[nan_frac > MAX_NAN_FRAC]
    if len(dropped):
        print("Features verwijderd (te veel NaN):")
        for c, f in dropped.items():
            print(f"  - {c}: {f:.0%} NaN")
    X = X.drop(columns=dropped.index)

    # sec_prev_event is NaN voor het eerste event van elke nacht -> die rijen vallen weg
    keep = X.notna().all(axis=1)
    print(f"Rijen met NaN verwijderd: {(~keep).sum()} van {len(X)}")
    meta = df.loc[keep, [c for c in META_COLS if c in df.columns]].reset_index(drop=True)
    return X.loc[keep].reset_index(drop=True), meta


def transform(X: pd.DataFrame, scaler_name: str) -> pd.DataFrame:
    # log1p vereist waarden > -1; negatieve sec_prev_event (overlappende events) -> 0
    X_log = np.log1p(X.clip(lower=0))
    scaler = StandardScaler() if scaler_name == "standard" else RobustScaler()
    return pd.DataFrame(scaler.fit_transform(X_log), columns=X.columns)


# =============================================================================
# CORRELATIE
# =============================================================================

def correlation_analysis(X: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # Spearman is rang-gebaseerd -> identiek op ruwe of log1p-data, robuust voor uitschieters
    corr = X.corr(method="spearman")
    corr.to_csv(out_dir / "spearman_correlation.csv", float_format="%.3f")

    fig, ax = plt.subplots(figsize=(0.45 * len(corr) + 3, 0.45 * len(corr) + 2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    if len(corr) <= 30:
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(corr.iat[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rho")
    ax.set_title("Spearman-correlatie tussen features")
    fig.tight_layout()
    fig.savefig(out_dir / "spearman_correlation_heatmap.png", dpi=200)
    plt.close(fig)

    # Sterk gecorreleerde paren (bovendriehoek)
    iu = np.triu_indices(len(corr), k=1)
    pairs = pd.DataFrame({
        "feature_a": corr.columns[iu[0]],
        "feature_b": corr.columns[iu[1]],
        "rho": corr.values[iu],
    })
    pairs["abs_rho"] = pairs["rho"].abs()
    pairs = pairs.sort_values("abs_rho", ascending=False).drop(columns="abs_rho")
    pairs.to_csv(out_dir / "correlation_pairs_sorted.csv", index=False, float_format="%.3f")

    high = pairs[pairs["rho"].abs() >= HIGH_CORR]
    print(f"\nSterk gecorreleerde paren (|rho| >= {HIGH_CORR}): {len(high)}")
    for _, r in high.head(25).iterrows():
        print(f"  {r.feature_a:<22} ~ {r.feature_b:<22} rho = {r.rho:+.2f}")
    return corr


# =============================================================================
# PCA
# =============================================================================

def pca_analysis(Xs: pd.DataFrame, meta: pd.DataFrame, out_dir: Path) -> None:
    pca = PCA()
    scores = pca.fit_transform(Xs.values)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    pc_names = [f"PC{i+1}" for i in range(len(evr))]

    n80 = int(np.searchsorted(cum, 0.80) + 1)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n_kaiser = int((pca.explained_variance_ > 1).sum())
    print(f"\nPCA op {Xs.shape[1]} features, {Xs.shape[0]} events")
    for i in range(min(8, len(evr))):
        print(f"  {pc_names[i]}: {evr[i]:6.1%}  (cumulatief {cum[i]:6.1%})")
    print(f"  PC's nodig voor 80% variantie: {n80}, voor 90%: {n90}")
    print(f"  Kaiser-criterium (eigenwaarde > 1, alleen zinvol bij StandardScaler): {n_kaiser}")

    pd.DataFrame({"PC": pc_names, "eigenvalue": pca.explained_variance_,
                  "explained_var": evr, "cumulative": cum}) \
        .to_csv(out_dir / "pca_explained_variance.csv", index=False, float_format="%.4f")

    # Scree-plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(evr) + 1)
    ax.bar(x, evr * 100, color="steelblue", label="per PC")
    ax.plot(x, cum * 100, "o-", color="darkorange", label="cumulatief")
    ax.axhline(80, ls="--", c="grey", lw=0.8)
    ax.axhline(90, ls=":", c="grey", lw=0.8)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Verklaarde variantie (%)")
    ax.set_xticks(x)
    ax.legend()
    ax.set_title("Scree-plot")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scree.png", dpi=200)
    plt.close(fig)

    # Loadings. Geschaald met sqrt(eigenwaarde): bij StandardScaler zijn dit
    # de correlaties tussen feature en PC -> direct interpreteerbaar.
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_),
                            index=Xs.columns, columns=pc_names)
    loadings.to_csv(out_dir / "pca_loadings.csv", float_format="%.3f")

    k = min(N_LOADING_PCS, len(pc_names))
    L = loadings.iloc[:, :k]
    lim = np.abs(L.values).max()
    fig, ax = plt.subplots(figsize=(1.1 * k + 3, 0.35 * len(L) + 2))
    im = ax.imshow(L.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"{p}\n({evr[i]:.0%})" for i, p in enumerate(L.columns)], fontsize=8)
    ax.set_yticks(range(len(L)))
    ax.set_yticklabels(L.index, fontsize=8)
    for i in range(len(L)):
        for j in range(k):
            ax.text(j, i, f"{L.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="loading")
    ax.set_title("PCA-loadings")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_loadings_heatmap.png", dpi=200)
    plt.close(fig)

    print("\nTop-5 features per PC (|loading|):")
    for p in pc_names[:min(4, k)]:
        top = loadings[p].abs().sort_values(ascending=False).head(5).index
        print(f"  {p}: " + ", ".join(f"{f} ({loadings.at[f, p]:+.2f})" for f in top))

    # Scores + PC1 vs PC2
    scores_df = pd.concat([meta, pd.DataFrame(scores[:, :k], columns=pc_names[:k])], axis=1)
    scores_df.to_csv(out_dir / "pca_scores.csv", index=False, float_format="%.4f")

    fig, ax = plt.subplots(figsize=(7, 6))
    if "stage_rk" in meta.columns and meta["stage_rk"].notna().any():
        for stage, sub in scores_df.groupby("stage_rk"):
            ax.scatter(sub["PC1"], sub["PC2"], s=6, alpha=0.5,
                       label=RK_STAGE_NAMES.get(int(stage), str(stage)))
        ax.legend(title="R&K stage", markerscale=3)
    else:
        ax.scatter(scores_df["PC1"], scores_df["PC2"], s=6, alpha=0.5)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.axhline(0, c="grey", lw=0.5)
    ax.axvline(0, c="grey", lw=0.5)
    ax.set_title("Events in PC1-PC2 ruimte")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_pc1_pc2_scatter.png", dpi=200)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--scaler", choices=["standard", "robust"], default="standard")
    parser.add_argument("--already-scaled", action="store_true",
                        help="Input is al log1p-getransformeerd en geschaald: niets meer doen")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"Ingeladen: {args.input}  shape={df.shape}")

    X, meta = select_and_clean(df)
    print(f"Features in analyse ({X.shape[1]}): {', '.join(X.columns)}")

    correlation_analysis(X, args.out)

    Xs = X if args.already_scaled else transform(X, args.scaler)
    pca_analysis(Xs, meta, args.out)

    print(f"\nOutput opgeslagen in: {args.out}")


if __name__ == "__main__":
    main()