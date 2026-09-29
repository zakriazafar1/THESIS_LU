"""
=============================================================================
3.1_corr_vif.py

Correlatie-check op de geschaalde arousal-featurematrix
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

Stappenplan:
  1. Geschaalde featurematrix inladen (arousal_feature_matrix_scaled.csv).
  2. Pearson-correlatiematrix (hoofd) + Spearman-correlatiematrix (controle).
  3. Heatmaps van beide.
  4. Sterk gecorreleerde paren (|r| > CORR_THRESHOLD, Pearson), met Spearman erbij.
  5. Paren waar |r - rho| > DIFF_THRESHOLD (Pearson vs Spearman wijken af).
  6. CORRELATIEGROEPEN: features die onderling allemaal |rho| >= GROUP_THRESHOLD
     hebben (hiërarchische clustering op 1 - |rho|, complete linkage). Uit elke
     groep kies je er één. Per feature: gemiddelde |rho| met de rest van de groep
     (hoe "centraal") en % missend -> helpt bij de keuze.
     Complete linkage garandeert dat ELK paar binnen een groep >= de drempel
     correleert (geen ketens A~B~C waarbij A en C nauwelijks samenhangen).

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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

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

GROUP_METHOD = "spearman"   # correlatie waarop de groepen gebaseerd zijn
GROUP_THRESHOLD = 0.80      # binnen een groep: elk paar |rho| >= deze waarde


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
# SECTIE 3 — CORRELATIEGROEPEN
# =============================================================================

def find_corr_groups(corr: pd.DataFrame, threshold: float) -> pd.Series:
    """
    Hiërarchische clustering op afstand 1 - |r|, complete linkage, gesneden op
    1 - threshold. Resultaat: groepsnummer per feature. Binnen een groep heeft
    elk paar |r| >= threshold.
    """
    dist = 1 - corr.abs().fillna(0).to_numpy()
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2                      # numeriek symmetrisch maken
    Z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(Z, t=1 - threshold, criterion="distance")
    return pd.Series(labels, index=corr.columns, name="group"), Z


def summarize_groups(corr: pd.DataFrame, groups: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """
    Eén rij per feature: groep, groepsgrootte, gemiddelde en minimale |rho| met
    de andere groepsleden en % missend. Groepen genummerd 1..k op grootte.
    'suggested' = het meest centrale lid (hoogste gemiddelde |rho|) -- alleen een
    startpunt, kies inhoudelijk.
    """
    # hernummer: grootste groep = 1
    sizes = groups.value_counts()
    order = sorted(sizes.index, key=lambda g: (-sizes[g], groups[groups == g].index[0]))
    renum = {old: new for new, old in enumerate(order, start=1)}
    groups = groups.map(renum)

    rows = []
    for g, members in groups.groupby(groups):
        feats = list(members.index)
        for f in feats:
            others = [o for o in feats if o != f]
            a = corr.loc[f, others].abs() if others else pd.Series(dtype=float)
            rows.append({
                "group": g,
                "group_size": len(feats),
                "feature": f,
                f"mean_abs_{GROUP_METHOD}_in_group": round(a.mean(), 3) if others else np.nan,
                f"min_abs_{GROUP_METHOD}_in_group": round(a.min(), 3) if others else np.nan,
                "pct_missing": round(100 * df[f].isna().mean(), 2),
            })
    out = pd.DataFrame(rows)
    key = f"mean_abs_{GROUP_METHOD}_in_group"
    out = out.sort_values(["group", key], ascending=[True, False], na_position="last")
    out["suggested"] = False
    multi = out["group_size"] > 1
    out.loc[out[multi].groupby("group")[key].idxmax(), "suggested"] = True
    return out.reset_index(drop=True)


def print_groups(summary: pd.DataFrame, threshold: float) -> None:
    key = f"mean_abs_{GROUP_METHOD}_in_group"
    multi = summary[summary["group_size"] > 1]
    single = summary[summary["group_size"] == 1]["feature"].tolist()
    n_groups = multi["group"].nunique()
    print(f"\nStap 6: {n_groups} correlatiegroep(en) (elk paar |{GROUP_METHOD}| >= {threshold}); "
          "kies uit elke groep één feature:")
    for g, sub in multi.groupby("group"):
        print(f"\n  Groep {g} ({len(sub)} features, min |rho| in groep = "
              f"{sub[f'min_abs_{GROUP_METHOD}_in_group'].min():.2f}):")
        for _, r in sub.iterrows():
            flag = "  <- meest centraal" if r["suggested"] else ""
            print(f"    {r['feature']:<22} gem |rho| = {r[key]:.2f}   "
                  f"missend = {r['pct_missing']:.1f}%{flag}")
    print(f"\n  Losse features (geen partner >= {threshold}), gewoon houden: {single if single else '(geen)'}")


def plot_dendrogram(Z, labels: list[str], threshold: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(labels)), 5))
    dendrogram(Z, labels=labels, color_threshold=1 - threshold, leaf_rotation=90,
               leaf_font_size=7, ax=ax)
    ax.axhline(1 - threshold, color="crimson", ls="--", lw=1,
               label=f"snijlijn: |{GROUP_METHOD}| = {threshold}")
    ax.set_ylabel(f"1 - |{GROUP_METHOD}| (complete linkage)")
    ax.set_title("Correlatiegroepen van features")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Dendrogram opgeslagen: {out_path}")


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

    # --- Stap 6: correlatiegroepen ---
    group_corr = spearman if GROUP_METHOD == "spearman" else pearson
    groups, Z = find_corr_groups(group_corr, GROUP_THRESHOLD)
    summary = summarize_groups(group_corr, groups, df)
    summary.to_csv(OUTPUT_DIR / "corr_groups.csv", index=False)
    print_groups(summary, GROUP_THRESHOLD)
    plot_dendrogram(Z, list(group_corr.columns), GROUP_THRESHOLD,
                    OUTPUT_DIR / "corr_groups_dendrogram.png")

    print(f"\nAlles opgeslagen in: {OUTPUT_DIR}")
    print("  - pearson_corr_matrix.csv / pearson_corr_heatmap.png   <- hoofdanalyse")
    print("  - spearman_corr_matrix.csv / spearman_corr_heatmap.png <- controle")
    print("  - high_corr_pairs.csv                <- redundante paren (Pearson, met rho erbij)")
    print("  - pearson_vs_spearman_divergent.csv  <- paren om visueel te checken (scatterplot)")
    print("  - corr_groups.csv / corr_groups_dendrogram.png <- kies 1 feature per groep")


if __name__ == "__main__":
    main()