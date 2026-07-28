"""
=============================================================================
correlation_selection.py   (stap 1 van 2)

Eerste helft van het vorige pca_features.py, losgetrokken: alleen de
correlatie-check en feature-selectie. Doel: van de volledige featurematrix
naar een REDUCED FEATURE SET zonder sterk onderling gecorreleerde features,
in originele/interpreteerbare eenheden (dus GEEN PCA hier — dat staat nu in
het losse script pca_reduction.py, dat als input de output van dit script
kan gebruiken, of desgewenst de volledige featurematrix).

Stappen:
  1. Missing/variantie-filter: features met te veel missing data of
     (bijna) geen spreiding gaan er eerst uit.
  2. Correlatie-check: Spearman-correlatiematrix + heatmap, overzicht van
     sterk gecorreleerde paren, en een hiërarchische clustering van de
     features zelf (op correlatie-afstand) om groepjes "zeggen hetzelfde"
     features te vinden. Per groep wordt de feature met de hoogste
     standaarddeviatie als representant gekozen.

BELANGRIJK — nog te verifiëren aannames:
  - Input is het output-bestand van build_feature_matrix.py
    (arousal_feature_matrix.csv). Check of INPUT_MATRIX nog klopt.
  - Output-map is als nieuwe submap "3. feature selection" naast
    "2. feature matrices ridge" gezet, puur als aanname op de bestaande
    naamgeving-conventie. Pas OUTPUT_DIR aan indien gewenst.
  - subject_id / group / night_id / event_idx / start_sec / end_sec worden
    NOOIT als clustering-feature meegenomen (identifiers/positie, geen
    signaal-eigenschap). duration_sec is een gewone, altijd meegenomen
    feature. stage_rk en sec_prev_event zijn kandidaat-context-features:
    die WORDEN standaard meegenomen, maar zijn met --drop-context uit te
    sluiten als je liever puur op signaal-morfologie clustert.
  - Draai eerst met --inspect om de missing/variantie/correlatie-tabellen
    te zien voordat je de volledige run (bestanden wegschrijven) doet.

Gebruik:
  python correlation_selection.py --inspect        # eerst checken
  python correlation_selection.py                  # volledige run
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")  # geen display nodig, alleen wegschrijven naar png
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATIE
# =============================================================================

FEATURE_MATRIX_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. feature matrices ridge"
)
INPUT_MATRIX = FEATURE_MATRIX_DIR / "arousal_feature_matrix.csv"
OUTPUT_DIR = FEATURE_MATRIX_DIR.parent / "3. feature selection\correlation"

# Kolommen die NOOIT als clustering-feature meedoen (identifiers / positie)
ID_COLS = ["subject_id", "group", "night_id", "event_idx", "start_sec", "end_sec"]

# Kandidaat context-features: inhoudelijk mogelijk relevant, maar geen
# signaal-morfologie. Los te herkennen in de output zodat je bewust kan
# kiezen of je ze meeneemt (met --drop-context laat je ze weg).
# duration_sec staat er NIET in: die wordt standaard altijd als volwaardige
# feature gebruikt, ook met --drop-context.
CONTEXT_FEATURE_COLS = ["stage_rk", "sec_prev_event"]

# Diagnostiek-thresholds
MAX_MISSING_FRAC = 0.30      # features met meer missing dan dit -> eruit
MIN_STD = 1e-8                # (bijna) constante features -> eruit

# Correlatie-instellingen
CORR_METHOD = "spearman"      # robuuster dan pearson voor ratio-achtige features die niet normaal verdeeld zijn
CORR_THRESHOLD = 0.85         # boven deze |correlatie| -> features worden als "redundant" gezien

pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)


# =============================================================================
# SECTIE 1 — INLADEN EN FEATURE-KOLOMMEN BEPALEN
# =============================================================================

def load_feature_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Featurematrix niet gevonden op {path}. Klopt INPUT_MATRIX nog, "
            f"of moet build_feature_matrix.py eerst (opnieuw) draaien?"
        )
    # sep=None + engine="python" detecteert automatisch komma of puntkomma
    # (Excel schrijft bij een NL-locale soms puntkomma-gescheiden CSV's terug
    # als het bestand tussentijds geopend/opgeslagen is). Zonder deze fallback
    # leest pandas het hele bestand als 1 kolom, met alle waarden als string.
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


def get_candidate_feature_cols(df: pd.DataFrame) -> list[str]:
    """Alle kolommen behalve de identifiers zijn in principe kandidaat-features."""
    return [c for c in df.columns if c not in ID_COLS]


# =============================================================================
# SECTIE 2 — MISSINGNESS EN VARIANTIE-FILTER
# =============================================================================

def summarize_missingness(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    frac_missing = df[feature_cols].isna().mean().sort_values(ascending=False)
    return frac_missing.rename("frac_missing").to_frame()


def summarize_variance(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    std = df[feature_cols].std(numeric_only=True)
    mean = df[feature_cols].mean(numeric_only=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = (std / mean.abs()).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({"std": std, "mean": mean, "cv": cv})
    return out.sort_values("std")


def filter_missing_and_constant(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], dict]:
    """Verwijdert features met te veel missing data of (bijna) geen spreiding.
    Geeft de overgebleven feature-lijst terug plus een log van wat/waarom eruit ging."""
    dropped = {}

    missing = summarize_missingness(df, feature_cols)
    too_missing = missing[missing["frac_missing"] > MAX_MISSING_FRAC].index.tolist()
    for f in too_missing:
        dropped[f] = f"missing={missing.loc[f, 'frac_missing']:.2f} > {MAX_MISSING_FRAC}"

    remaining = [c for c in feature_cols if c not in too_missing]

    variance = summarize_variance(df, remaining)
    near_constant = variance[variance["std"].fillna(0) < MIN_STD].index.tolist()
    for f in near_constant:
        dropped[f] = f"std={variance.loc[f, 'std']:.2e} < {MIN_STD}"

    remaining = [c for c in remaining if c not in near_constant]
    return remaining, dropped


# =============================================================================
# SECTIE 3 — CORRELATIE-CHECK
# =============================================================================

def compute_correlation_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return df[feature_cols].corr(method=CORR_METHOD)


def list_correlated_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Alle paren met |correlatie| > threshold, één keer per paar, gesorteerd op sterkte."""
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > threshold:
                pairs.append((cols[i], cols[j], r))
    pairs_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "corr"])
    return pairs_df.reindex(pairs_df["corr"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def cluster_correlated_features(corr: pd.DataFrame, threshold: float) -> tuple[pd.Series, np.ndarray]:
    """
    Hiërarchische clustering van features op correlatie-afstand (1 - |r|),
    zodat groepjes onderling sterk gecorreleerde features samen 1 cluster
    vormen. Een NaN in de correlatiematrix (kan bij een feature met bijna
    geen variatie in een subset) wordt behandeld als "geen relatie" (afstand 1).

    Geeft (cluster_assignment, Z) terug: een Series feature-naam -> cluster-id,
    en de linkage-matrix (voor het dendrogram).
    """
    dist = 1 - corr.abs()
    dist = dist.fillna(1.0)
    dist_vals = dist.values.copy()
    np.fill_diagonal(dist_vals, 0.0)
    # symmetrie afdwingen (floating point rounding kan het net iets scheef trekken,
    # squareform is daar streng in)
    dist_vals = (dist_vals + dist_vals.T) / 2
    condensed = squareform(dist_vals, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1 - threshold, criterion="distance")
    return pd.Series(cluster_ids, index=corr.columns, name="cluster_id"), Z


def pick_cluster_representatives(cluster_assignment: pd.Series,
                                  variance_table: pd.DataFrame) -> pd.DataFrame:
    """
    Kiest per correlatie-cluster één representatieve feature: degene met de
    hoogste standaarddeviatie (meest informatieve spreiding) binnen de groep.
    Geeft een overzichtstabel terug: cluster_id, alle leden, gekozen representant.
    """
    rows = []
    for cid, members in cluster_assignment.groupby(cluster_assignment).groups.items():
        members = list(members)
        if len(members) == 1:
            rep = members[0]
        else:
            stds = variance_table.loc[members, "std"]
            rep = stds.idxmax()
        rows.append({"cluster_id": cid, "n_members": len(members),
                      "members": ", ".join(members), "representative": rep})
    return pd.DataFrame(rows).sort_values("n_members", ascending=False).reset_index(drop=True)


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(corr.columns)), max(6, 0.4 * len(corr.columns))))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title(f"{CORR_METHOD.capitalize()} correlatie tussen features")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlatie")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dendrogram(Z, labels, out_path: Path, threshold: float):
    fig, ax = plt.subplots(figsize=(max(8, 0.3 * len(labels)), 6))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=7,
               color_threshold=1 - threshold, ax=ax)
    ax.axhline(1 - threshold, color="grey", linestyle="--", linewidth=1,
               label=f"cluster-afkap (|r| = {threshold})")
    ax.set_ylabel("afstand (1 - |correlatie|)")
    ax.set_title("Hiërarchische clustering van features op correlatie")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# SECTIE 4 — HOOFDLOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                         help="Alleen diagnostiek printen (missing/variantie/correlatie-paren), niets wegschrijven")
    parser.add_argument("--input", type=Path, default=INPUT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD)
    parser.add_argument("--drop-context", action="store_true",
                         help="Laat stage_rk/sec_prev_event weg, gebruik alleen signaal-features + duration_sec")
    args = parser.parse_args()

    df = load_feature_matrix(args.input)
    print(f"Featurematrix geladen: {df.shape[0]} events, {df.shape[1]} kolommen")

    feature_cols = get_candidate_feature_cols(df)
    if args.drop_context:
        feature_cols = [c for c in feature_cols if c not in CONTEXT_FEATURE_COLS]
        print(f"--drop-context: {CONTEXT_FEATURE_COLS} weggelaten")

    # --- Stap 1: missing + variantie filter ---
    missing = summarize_missingness(df, feature_cols)
    print("\n--- Missingness per feature (top 10) ---")
    print(missing.head(10))

    variance = summarize_variance(df, feature_cols)
    print("\n--- Variantie per feature (10 laagste std) ---")
    print(variance.head(10))

    clean_cols, dropped_log = filter_missing_and_constant(df, feature_cols)
    print(f"\n{len(dropped_log)} features verwijderd (te veel missing / (bijna) constant):")
    for f, reason in dropped_log.items():
        print(f"  - {f}: {reason}")
    print(f"{len(clean_cols)} features over na deze filter.")

    # --- Stap 2: correlatie-check ---
    corr = compute_correlation_matrix(df, clean_cols)
    corr_pairs = list_correlated_pairs(corr, args.corr_threshold)
    print(f"\n--- Sterk gecorreleerde paren (|{CORR_METHOD}| > {args.corr_threshold}) ---")
    print(corr_pairs if len(corr_pairs) else "(geen)")

    cluster_assignment, Z = cluster_correlated_features(corr, args.corr_threshold)
    variance_clean = summarize_variance(df, clean_cols)
    reps_table = pick_cluster_representatives(cluster_assignment, variance_clean)
    print(f"\n--- Feature-clusters (correlatie-afkap |r| = {args.corr_threshold}) ---")
    print(reps_table)

    reduced_feature_cols = reps_table["representative"].tolist()
    print(f"\nReduced feature set ({len(reduced_feature_cols)} features, "
          f"1 per correlatie-cluster): {reduced_feature_cols}")

    if args.inspect:
        print("\nInspectie klaar. Pas CORR_THRESHOLD / MAX_MISSING_FRAC / MIN_STD aan "
              "als de filtering niet klopt, of check CONTEXT_FEATURE_COLS, voordat je "
              "de volledige run (bestanden wegschrijven) doet.")
        return

    # --- Output wegschrijven ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing.to_csv(args.output_dir / "missingness_per_feature.csv")
    variance.to_csv(args.output_dir / "variance_per_feature.csv")
    corr.to_csv(args.output_dir / "correlation_matrix.csv")
    corr_pairs.to_csv(args.output_dir / "correlated_pairs.csv", index=False)
    reps_table.to_csv(args.output_dir / "feature_clusters.csv", index=False)

    # id-kolommen + reduced (originele, niet-getransformeerde) features
    reduced_out = pd.concat([df[[c for c in ID_COLS if c in df.columns]].reset_index(drop=True),
                              df[reduced_feature_cols].reset_index(drop=True)], axis=1)
    reduced_out.to_csv(args.output_dir / "reduced_features_events.csv", index=False, float_format="%.4f")

    plot_correlation_heatmap(corr, args.output_dir / "correlation_heatmap.png")
    plot_dendrogram(Z, list(corr.columns), args.output_dir / "feature_dendrogram.png", args.corr_threshold)

    print(f"\nAlle output weggeschreven naar: {args.output_dir}")
    print("  - reduced_features_events.csv  -> originele features, 1 per correlatie-cluster")
    print("    (gebruik dit bestand als --input voor pca_reduction.py, stap 2)")
    print("  - correlation_heatmap.png / feature_dendrogram.png")


if __name__ == "__main__":
    main()