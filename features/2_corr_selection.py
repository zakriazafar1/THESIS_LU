"""
=============================================================================
2_corr_selection.py

Doel: van de volledige featurematrix naar een REDUCED FEATURE SET zonder sterk
onderling gecorreleerde features, in originele/interpreteerbare eenheden
(dus GEEN PCA hier — dat staat nu in het losse script pca.py, dat als input
de output van dit script kan gebruiken, of desgewenst de volledige featurematrix).

Stappen:
  1. Missing/variantie-filter: features met te veel missing data of
     (bijna) geen spreiding gaan er eerst uit.
  2. Correlatie-check: Spearman-correlatiematrix + heatmap, overzicht van
     sterk gecorreleerde paren, en een hiërarchische clustering van de
     features zelf (op correlatie-afstand) om groepjes "zeggen hetzelfde"
     features te vinden. Per groep wordt de feature met de hoogste
     standaarddeviatie als representant gekozen.
  3. Pooled vs. within-subject correlatie: dezelfde correlatie-paren opnieuw
     berekend NA het aftrekken van het subject-gemiddelde per feature. Een
     paar dat pooled sterk correleert maar within-subject niet (of andersom)
     wijst op een Simpson's-paradox-achtig pooling-effect (bv. iemand met
     over de hele linie hogere band-power in elke band, wat de gepoolde
     correlatie tussen banden opblaast zonder dat het iets zegt over hoe
     een individueel event zich onderscheidt).
  4. VIF (variance inflation factor) per feature: vangt multivariate
     redundantie die een paarsgewijze correlatiematrix mist (3+ features die
     samen bijna volledig voorspelbaar zijn uit elkaar, zonder dat een los
     paar boven de threshold komt).
  5. Scatterplots van de sterkst gecorreleerde paren, want eenzelfde
     correlatiecoëfficiënt kan bij heel verschillende onderliggende
     relaties horen (Anscombe's quartet) — niet blind op het getal varen.

BELANGRIJK:
  - Input is het output-bestand van 1_feature_matrix.py
    (arousal_feature_matrix.csv). Check of INPUT_MATRIX nog klopt.
  - Output-map is als nieuwe submap "2. feature selection" naast
    "1. feature matrices" gezet, puur als aanname op de bestaande
    naamgeving-conventie. Pas OUTPUT_DIR aan indien gewenst.
  - subject_id / group / night_id / event_idx / start_sec / end_sec worden
    NOOIT als clustering-feature meegenomen (identifiers/positie, geen
    signaal-eigenschap). duration_sec is een gewone, altijd meegenomen
    feature. stage_rk en sec_prev_event zijn kandidaat-context-features:
    die WORDEN standaard meegenomen, maar zijn met --drop-context uit te
    sluiten als je liever puur op signaal-morfologie clustert.
  - De within-subject vergelijking (stap 3) en VIF (stap 4) hebben genoeg
    subjects/features nodig om zinvol te zijn. Bij te weinig subjects
    (< 2 met >1 event) of een singuliere correlatiematrix print het script
    een waarschuwing i.p.v. te crashen.
  - Draai eerst met --inspect om de missing/variantie/correlatie/VIF-tabellen
    te zien voordat je de volledige run (bestanden + plots wegschrijven) doet.

Gebruik:
  python 2_corr_selection.py --inspect        # eerst checken
  python 2_corr_selection.py                  # volledige run
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
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices"
)
INPUT_MATRIX = FEATURE_MATRIX_DIR / "arousal_feature_matrix.csv"
OUTPUT_DIR = FEATURE_MATRIX_DIR.parent / "2. feature selection\correlation"

# Kolommen die NOOIT als clustering-feature meedoen (identifiers / positie)
ID_COLS = ["subject_id", "group", "night_id", "event_idx", "start_sec", "end_sec"]
SUBJECT_COL = "subject_id"

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
POOLED_VS_WITHIN_DIFF_FLAG = 0.15   # |pooled - within| boven dit -> gemarkeerd als mogelijk pooling-effect

# VIF-instelling
VIF_FLAG_THRESHOLD = 5.0      # gangbare vuistregel: VIF > 5 a 10 wijst op problematische multicollineariteit

# Scatterplots van de N sterkst gecorreleerde paren
N_SCATTER_PAIRS = 12

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
# SECTIE 3 — CORRELATIE-CHECK (POOLED)
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


# =============================================================================
# SECTIE 4 — POOLED VS. WITHIN-SUBJECT CORRELATIE
# =============================================================================

def compute_within_subject_correlation(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame | None:
    """
    Trekt per subject het gemiddelde van elke feature af (within-subject
    centering) voordat er gecorreleerd wordt. Dit voorkomt dat between-subject
    verschillen (iemand met over de hele linie hogere/lagere band-power in
    elke band) de correlatiematrix domineren -- een gepoolde correlatie kan
    zo'n gedeeld "hoog/laag"-niveau tussen subjects aanzien voor een relatie
    tussen de features zelf op event-niveau (Simpson's paradox).

    Geeft None terug als SUBJECT_COL ontbreekt of er te weinig subjects met
    >1 event zijn om hier iets zinnigs over te zeggen.
    """
    if SUBJECT_COL not in df.columns:
        print(f"  [WAARSCHUWING] kolom '{SUBJECT_COL}' niet gevonden -- within-subject "
              f"vergelijking wordt overgeslagen.")
        return None

    events_per_subject = df.groupby(SUBJECT_COL).size()
    n_usable_subjects = (events_per_subject > 1).sum()
    if n_usable_subjects < 2:
        print(f"  [WAARSCHUWING] maar {n_usable_subjects} subject(en) met >1 event -- "
              f"within-subject centering is hier niet zinvol (te weinig within-subject "
              f"variatie om op te correleren), wordt overgeslagen.")
        return None

    group_means = df.groupby(SUBJECT_COL)[feature_cols].transform("mean")
    centered = df[feature_cols] - group_means
    return centered.corr(method=CORR_METHOD)


def compare_pooled_vs_within(corr_pairs_pooled: pd.DataFrame,
                              within_corr: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Zet voor elk paar boven de threshold in de gepoolde matrix de
    within-subject-correlatie ernaast. Een groot verschil (zie
    POOLED_VS_WITHIN_DIFF_FLAG) betekent dat de gepoolde correlatie deels of
    vooral een between-subject-effect is, en dus voorzichtig geinterpreteerd
    moet worden als het gaat om redundantie tussen events.
    """
    if within_corr is None or len(corr_pairs_pooled) == 0:
        return None
    rows = []
    for _, r in corr_pairs_pooled.iterrows():
        a, b, pooled_r = r["feature_a"], r["feature_b"], r["corr"]
        within_r = within_corr.loc[a, b] if (a in within_corr.index and b in within_corr.columns) else np.nan
        diff = abs(pooled_r - within_r) if pd.notna(within_r) else np.nan
        rows.append({"feature_a": a, "feature_b": b, "pooled_corr": pooled_r,
                      "within_subject_corr": within_r, "abs_diff": diff,
                      "mogelijk_pooling_effect": (diff > POOLED_VS_WITHIN_DIFF_FLAG) if pd.notna(diff) else False})
    out = pd.DataFrame(rows)
    return out.sort_values("abs_diff", ascending=False, na_position="last").reset_index(drop=True)


# =============================================================================
# SECTIE 5 — VIF (MULTIVARIATE REDUNDANTIE)
# =============================================================================

def compute_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    VIF per feature = 1 / (1 - R^2), waarbij R^2 komt uit het regresseren van
    die feature op alle overige features. Vangt multivariate redundantie die
    een paarsgewijze correlatiematrix mist: 3+ features die SAMEN bijna
    volledig voorspelbaar zijn uit elkaar, zonder dat een los paar boven de
    correlatie-threshold hoeft te komen. Vuistregel: VIF > 5 a 10 wijst op
    problematische multicollineariteit.

    Berekend via VIF_i = diag(inverse(pearson-correlatiematrix))_i -- een
    bekende kortere weg die equivalent is aan losse regressies per feature.
    Dit is bewust Pearson (VIF is een lineair-regressie-concept), ook al
    gebruiken we Spearman voor de hoofd-correlatiematrix hierboven.

    Missing values worden alleen voor DEZE berekening per feature met de
    mediaan geimputeerd (VIF kan niet met NaN's rekenen) -- dat raakt verder
    niets anders in het script/de output.
    """
    imputed = df[feature_cols].apply(lambda s: s.fillna(s.median()))
    # kolommen die na imputatie alsnog constant zijn (bv. altijd NaN geweest)
    # geven een singuliere matrix -> zouden er via filter_missing_and_constant
    # al uit moeten zijn, maar voor de zekerheid hier ook nog een check
    zero_var = imputed.columns[imputed.std() < MIN_STD].tolist()
    usable = [c for c in feature_cols if c not in zero_var]
    if zero_var:
        print(f"  [WAARSCHUWING] {zero_var} hebben geen spreiding na imputatie, "
              f"uitgesloten van VIF-berekening.")

    pearson_corr = imputed[usable].corr(method="pearson").values
    try:
        inv_corr = np.linalg.inv(pearson_corr)
        singular = False
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(pearson_corr)
        singular = True

    if singular:
        print("  [WAARSCHUWING] Pearson-correlatiematrix is singulier (perfecte multicollineariteit) "
              "-- pseudo-inverse gebruikt, VIF-waarden hieronder zijn dan indicatief/een ondergrens.")

    vif = np.diag(inv_corr)
    out = pd.DataFrame({"VIF": vif}, index=usable).sort_values("VIF", ascending=False)
    out["boven_threshold"] = out["VIF"] > VIF_FLAG_THRESHOLD
    return out


# =============================================================================
# SECTIE 6 — PLOTS
# =============================================================================

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


def plot_top_pair_scatterplots(df: pd.DataFrame, corr_pairs: pd.DataFrame, out_path: Path,
                                n: int = N_SCATTER_PAIRS):
    """
    Scatterplot per sterk gecorreleerd paar (de n sterkste), zodat je zelf
    ziet wat voor relatie er achter het getal zit -- eenzelfde correlatie-
    coefficient kan bij heel verschillende vormen horen (lineair, gebogen,
    1 uitbijter die alles trekt, twee subgroepen die toevallig op een lijn
    liggen). Puur visuele check, geen extra statistiek.
    """
    n = min(n, len(corr_pairs))
    if n == 0:
        return
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for i in range(n):
        row = corr_pairs.iloc[i]
        a, b, r = row["feature_a"], row["feature_b"], row["corr"]
        ax = axes[i]
        ax.scatter(df[a], df[b], s=8, alpha=0.4, edgecolors="none")
        ax.set_xlabel(a, fontsize=7)
        ax.set_ylabel(b, fontsize=7)
        ax.set_title(f"r = {r:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# SECTIE 7 — HOOFDLOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                         help="Alleen diagnostiek printen (missing/variantie/correlatie/VIF), niets wegschrijven")
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

    # --- Stap 2: correlatie-check (pooled) ---
    corr = compute_correlation_matrix(df, clean_cols)
    corr_pairs = list_correlated_pairs(corr, args.corr_threshold)
    print(f"\n--- Sterk gecorreleerde paren, pooled (|{CORR_METHOD}| > {args.corr_threshold}) ---")
    print(corr_pairs if len(corr_pairs) else "(geen)")

    cluster_assignment, Z = cluster_correlated_features(corr, args.corr_threshold)
    variance_clean = summarize_variance(df, clean_cols)
    reps_table = pick_cluster_representatives(cluster_assignment, variance_clean)
    print(f"\n--- Feature-clusters (correlatie-afkap |r| = {args.corr_threshold}) ---")
    print(reps_table)

    reduced_feature_cols = reps_table["representative"].tolist()
    print(f"\nReduced feature set ({len(reduced_feature_cols)} features, "
          f"1 per correlatie-cluster): {reduced_feature_cols}")

    # --- Stap 3: pooled vs. within-subject correlatie ---
    print(f"\n--- Pooled vs. within-subject correlatie ---")
    within_corr = compute_within_subject_correlation(df, clean_cols)
    comparison = compare_pooled_vs_within(corr_pairs, within_corr)
    if comparison is not None:
        flagged = comparison[comparison["mogelijk_pooling_effect"]]
        print(f"{len(flagged)} / {len(comparison)} paren met |pooled - within| > "
              f"{POOLED_VS_WITHIN_DIFF_FLAG} (mogelijk (deels) between-subject effect):")
        print(flagged.head(15) if len(flagged) else "(geen)")

    # --- Stap 4: VIF ---
    print(f"\n--- VIF per feature (top 10, hoe hoger hoe meer multivariate redundantie) ---")
    vif_table = compute_vif(df, clean_cols)
    print(vif_table.head(10))
    n_flagged_vif = int(vif_table["boven_threshold"].sum())
    print(f"{n_flagged_vif} / {len(vif_table)} features met VIF > {VIF_FLAG_THRESHOLD}")

    if args.inspect:
        print("\nInspectie klaar. Pas CORR_THRESHOLD / MAX_MISSING_FRAC / MIN_STD / "
              "VIF_FLAG_THRESHOLD aan als de filtering niet klopt, of check "
              "CONTEXT_FEATURE_COLS, voordat je de volledige run (bestanden + plots "
              "wegschrijven) doet.")
        return

    # --- Output wegschrijven ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing.to_csv(args.output_dir / "missingness_per_feature.csv")
    variance.to_csv(args.output_dir / "variance_per_feature.csv")
    corr.to_csv(args.output_dir / "correlation_matrix_pooled.csv")
    corr_pairs.to_csv(args.output_dir / "correlated_pairs_pooled.csv", index=False)
    reps_table.to_csv(args.output_dir / "feature_clusters.csv", index=False)
    vif_table.to_csv(args.output_dir / "vif_per_feature.csv")

    if within_corr is not None:
        within_corr.to_csv(args.output_dir / "correlation_matrix_within_subject.csv")
    if comparison is not None:
        comparison.to_csv(args.output_dir / "pooled_vs_within_subject_comparison.csv", index=False)

    # id-kolommen + reduced (originele, niet-getransformeerde) features
    reduced_out = pd.concat([df[[c for c in ID_COLS if c in df.columns]].reset_index(drop=True),
                              df[reduced_feature_cols].reset_index(drop=True)], axis=1)
    reduced_out.to_csv(args.output_dir / "reduced_features_events.csv", index=False, float_format="%.4f")

    plot_correlation_heatmap(corr, args.output_dir / "correlation_heatmap_pooled.png")
    plot_dendrogram(Z, list(corr.columns), args.output_dir / "feature_dendrogram.png", args.corr_threshold)
    plot_top_pair_scatterplots(df, corr_pairs, args.output_dir / "top_correlated_pairs_scatter.png")
    if within_corr is not None:
        plot_correlation_heatmap(within_corr, args.output_dir / "correlation_heatmap_within_subject.png")

    print(f"\nAlle output weggeschreven naar: {args.output_dir}")
    print("  - reduced_features_events.csv          -> originele features, 1 per correlatie-cluster")
    print("    (gebruik dit bestand als --input voor pca_reduction.py, stap 2)")
    print("  - pooled_vs_within_subject_comparison.csv -> check op Simpson's-paradox-achtige pooling")
    print("  - vif_per_feature.csv                   -> multivariate redundantie (3+ features samen)")
    print("  - correlation_heatmap_pooled.png / _within_subject.png / feature_dendrogram.png /")
    print("    top_correlated_pairs_scatter.png")


if __name__ == "__main__":
    main()