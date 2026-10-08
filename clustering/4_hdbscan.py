"""
=============================================================================
4_hdbscan.py

Input:
  pca_scores_5pc_whitened.csv   (default; elke PC weegt even zwaar)
  pca_scores_5pc_unwhitened.csv (via --input; PC1 = spectrale intensiteit domineert)
  arousal_features_reduced_scaled.csv (via --input; de 8 geschaalde features zelf)
  Features worden automatisch herkend: zijn er kolommen die met "PC" beginnen,
  dan zijn dat de features; anders alle numerieke kolommen die geen metadata zijn.

Parameters (de twee die ertoe doen):
  min_cluster_size  -- de kleinste groep die je nog een "subtype" wilt noemen.
  min_samples       -- hoe conservatief: hoe groter, hoe meer punten als ruis (-1)
                       worden gelabeld en hoe alleen echt dichte gebieden cluster
                       worden. Default in HDBSCAN = min_cluster_size.
  cluster_selection_method
                    -- 'eom' (default): kiest de stabielste, vaak grotere clusters.
                       'leaf': kiest de kleinste, fijnste clusters uit de boom.
                       Bij een continuüm geeft 'eom' vaak 1 grote cluster + ruis,
                       'leaf' versnippert juist. Beide proberen is informatief.

Scoring (label-vrij):
  relative_validity_ van HDBSCAN (snelle DBCV-benadering), hoger = beter.
  Plus sanity-checks (aantal clusters, ruisfractie): configuraties met 1 reuzen-
  cluster of 90% ruis kunnen op relative_validity toch goed scoren.
  Let op: die sanity-checks sturen naar "er zijn clusters". Het script rapporteert
  daarom ook hoeveel configuraties ze NIET haalden -- als bijna niets slaagt, is
  dat zelf een aanwijzing dat er geen duidelijke clusterstructuur is.

Stabiliteit (alleen voor de top-N configuraties):
  Herhaald opnieuw clusteren op een willekeurige 80% van de SUBJECTS (niet van
  de events, want events uit dezelfde nacht/persoon lijken op elkaar). Per
  herhaling: Adjusted Rand Index (ARI) tussen de labels op de volledige data
  en op de subset, over de gedeelde events (ruis telt als eigen label).
  ARI ~1 = dezelfde indeling, ~0 = niet beter dan toeval.

Eindresultaat:
  Voor de beste configuratie (of één die je kiest met --final) worden labels,
  membership-probabilities en outlier-scores per event weggeschreven, plus per
  cluster de grootte en hoeveel verschillende subjects erin zitten (een "cluster"
  die uit een paar personen komt is waarschijnlijk geen arousal-subtype).

Gebruik:
  python 4_hdbscan.py                       # default grid op whitened scores
  python 4_hdbscan.py --quick               # klein grid, snelle check
  python 4_hdbscan.py --input ...\\pca_scores_5pc_unwhitened.csv --tag unwhitened
  python 4_hdbscan.py --final 130 25 eom    # vaste eindconfiguratie

Output (in OUTPUT_DIR/<tag>):
  hdbscan_grid.csv              -- elke configuratie
  hdbscan_ranked.csv            -- gerangschikt (valide, dan relative_validity)
  hdbscan_stability.csv         -- ARI-stabiliteit van de top-N
  hdbscan_final_labels.csv      -- metadata + PC's + label/probability/outlier_score
  hdbscan_final_cluster_summary.csv -- per cluster: n, %, n_subjects, grootste subject-aandeel
  fig_clusters_main.png/.pdf    -- 2 scatterplots (PC1 x PC2 + best scheidend PC-paar),
                                   voor de resultatensectie
  fig_clusters_pairs.png/.pdf   -- alle PC-paren (corner plot) + verdeling per PC,
                                   voor de appendix

Plots:
  Ruis (-1) in lichtgrijs onderop, clusters in vaste kleurvolgorde met elk een 
  eigen markervorm. Legenda met n en % per cluster. De asnamen gebruiken 
  PC_LABELS (interpretatie per PC) -- controleer die tegen pca_loadings_heatmap.png 
  uit 3_pca.py als je de PCA opnieuw draait.
  Alleen de plots opnieuw maken (zonder grid/stabiliteit opnieuw te draaien):
      python 4_hdbscan.py --plot-only
  (leest hdbscan_final_labels.csv uit dezelfde <tag>-map)
=============================================================================
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score

try:
    import hdbscan
except ImportError:
    sys.exit("hdbscan is niet geïnstalleerd:  pip install hdbscan")

# =============================================================================
# CONFIGURATIE
# =============================================================================

BASE = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2"
)
DEFAULT_INPUT = BASE / r"4. clustering\3. pca\pca_scores_5pc_whitened.csv"
OUTPUT_DIR = BASE / r"4. clustering\4. hdbscan"

# Full grid.
DEFAULT_MIN_CLUSTER_SIZE = [65, 130, 260, 650]         
DEFAULT_MIN_SAMPLES = [5, 10, 25, 50, 100]
DEFAULT_SELECTION = ["eom", "leaf"]

# --quick grid.
QUICK_MIN_CLUSTER_SIZE = [130, 260]
QUICK_MIN_SAMPLES = [10, 50]
QUICK_SELECTION = ["eom"]

# Sanity-checks voor "valide" configuraties.
MIN_CLUSTERS = 1
MAX_CLUSTERS = 5
MAX_NOISE_FRACTION = 0.5

# Stabiliteit.
TOP_N_STABILITY = 5
N_RESAMPLES = 10
SUBJECT_FRACTION = 0.80
RANDOM_STATE = 2554542

# Metadata-kolommen (nooit als feature gebruikt).
METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event", "stage_rk",
    "cluster", "membership_probability", "outlier_score",
]

# Plots.
# Interpretatie per PC voor de asnamen (op basis van de loadings uit 3_pca.py).
PC_LABELS = {
    "PC1": "PC1: overall spectral activation",
    "PC2": "PC2: event duration",
    "PC3": "PC3: head movement", 
    "PC4": "PC4: contrast PPG vs. duration",
    "PC5": "PC5: contrast slow vs. fast frequencies",
    # asnamen als HDBSCAN op de geschaalde features zelf draait
    "mean_delta_ratio": "Delta ratio (z)",
    "mean_theta_ratio": "Theta ratio (z)",
    "mean_alpha_ratio": "Alpha ratio (z)",
    "mean_sigma_ratio": "Sigma ratio (z)",
    "mean_beta_ratio": "Beta ratio (z)",
    "duration_sec": "Duration (z)",
    "oxy_amp_ratio": "PPG amplitude ratio (z)",
    "motion_rms": "Head movement RMS (z)",
}
# Vaste plotparen bij features (i.p.v. PC's) als er < 2 clusters zijn:
# spectrale intensiteit x duur, en traag (delta) x snel (beta).
FEATURE_PLOT_PAIRS = [("mean_theta_ratio", "duration_sec"),
                      ("mean_delta_ratio", "mean_beta_ratio")]
# Vaste categorische volgorde (gevalideerd palet); elke cluster ook een eigen marker.
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4",
                  "#008300", "#eda100", "#e34948"]
CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
NOISE_COLOR = "#C4C4C4"
TEXT_COLOR = "#222222"
MUTED_COLOR = "#666666"
POINT_SIZE = 5
POINT_ALPHA = 0.45


# =============================================================================
# SECTIE 1 — INLADEN
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    """CSV met automatische separator-detectie en fallback naar decimal=','."""
    if not path.exists():
        raise FileNotFoundError(f"{path} bestaat niet -- run eerst 3_pca.py.")
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
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """PC-kolommen als die er zijn, anders alle numerieke niet-metadata-kolommen."""
    pc_cols = [c for c in df.columns if c.upper().startswith("PC")]
    if pc_cols:
        return pc_cols
    return [c for c in df.columns
            if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])]


def load_scores(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = read_csv_robust(path)
    pc_cols = get_feature_cols(df)
    if not pc_cols:
        raise ValueError(f"Geen feature-kolommen gevonden in {path}.")
    if df[pc_cols].isna().any().any():
        raise ValueError("NaN in de PC-scores -- los dat eerst op in 3.4.")
    if "subject_id" not in df.columns:
        raise ValueError("Kolom 'subject_id' ontbreekt (nodig voor de stabiliteitscheck).")
    print(f"Geladen: {path}")
    kind = "PC's" if pc_cols[0].upper().startswith("PC") else "features"
    print(f"{len(df)} events x {len(pc_cols)} {kind}: {pc_cols}")
    return df, pc_cols


# =============================================================================
# SECTIE 2 — HDBSCAN
# =============================================================================

def fit_hdbscan(X: np.ndarray, mcs: int, ms: int, method: str,
                allow_single: bool, validity: bool = True):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        cluster_selection_method=method,
        allow_single_cluster=allow_single,
        gen_min_span_tree=validity,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X)
    return clusterer, labels


def describe_labels(labels: np.ndarray) -> dict:
    clusters = [l for l in np.unique(labels) if l != -1]
    sizes = sorted([int((labels == l).sum()) for l in clusters], reverse=True)
    n = len(labels)
    return {
        "n_clusters": len(clusters),
        "noise_fraction": float(np.mean(labels == -1)),
        "largest_cluster_fraction": sizes[0] / n if sizes else 0.0,
        "cluster_sizes": ",".join(map(str, sizes)),
    }


def run_grid(X, mcs_list, ms_list, methods, allow_single) -> pd.DataFrame:
    configs = list(itertools.product(mcs_list, ms_list, methods))
    rows = []
    t_start = time.time()
    for i, (mcs, ms, method) in enumerate(configs, start=1):
        t0 = time.time()
        clusterer, labels = fit_hdbscan(X, mcs, ms, method, allow_single)
        try:
            rel_val = float(clusterer.relative_validity_)
        except Exception:
            rel_val = np.nan
        info = describe_labels(labels)
        rows.append({"min_cluster_size": mcs, "min_samples": ms, "selection": method,
                     "relative_validity": rel_val, **info})
        print(f"[{i}/{len(configs)}] mcs={mcs:<4} ms={ms:<4} {method:<4} -> "
              f"{info['n_clusters']} cluster(s), ruis {info['noise_fraction']:.1%}, "
              f"grootste {info['largest_cluster_fraction']:.1%}, "
              f"rel_val {rel_val:.3f}  ({time.time() - t0:.1f}s)")
    print(f"\nGrid klaar in {time.time() - t_start:.1f}s ({len(rows)} configuraties)")
    return pd.DataFrame(rows)


def rank_grid(grid: pd.DataFrame) -> pd.DataFrame:
    grid = grid.copy()
    grid["valid"] = (
        (grid["n_clusters"] >= MIN_CLUSTERS)
        & (grid["n_clusters"] <= MAX_CLUSTERS)
        & (grid["noise_fraction"] <= MAX_NOISE_FRACTION)
    )
    return grid.sort_values(["valid", "relative_validity"], ascending=[False, False]
                            ).reset_index(drop=True)


# =============================================================================
# SECTIE 3 — STABILITEIT (subject-gesplitst)
# =============================================================================

def stability(X: np.ndarray, subjects: np.ndarray, mcs: int, ms: int, method: str,
              allow_single: bool, n_resamples: int, frac: float, seed: int) -> dict:
    """
    Fit op alle data, dan herhaald op een random `frac` van de subjects.
    ARI tussen beide labelingen op de gedeelde events (ruis = eigen label).
    """
    _, full_labels = fit_hdbscan(X, mcs, ms, method, allow_single, validity=False)
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    n_pick = max(2, int(round(frac * len(uniq))))

    aris, n_clusters = [], []
    for _ in range(n_resamples):
        picked = rng.choice(uniq, size=n_pick, replace=False)
        mask = np.isin(subjects, picked)
        _, sub_labels = fit_hdbscan(X[mask], mcs, ms, method, allow_single, validity=False)
        aris.append(adjusted_rand_score(full_labels[mask], sub_labels))
        n_clusters.append(len(set(sub_labels)) - (1 if -1 in sub_labels else 0))

    return {
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris, ddof=1)) if len(aris) > 1 else np.nan,
        "ari_min": float(np.min(aris)),
        "n_clusters_subsets_mean": float(np.mean(n_clusters)),
        "n_clusters_subsets_range": f"{min(n_clusters)}-{max(n_clusters)}",
    }


# =============================================================================
# SECTIE 4 — EINDCONFIGURATIE
# =============================================================================

def final_fit(df: pd.DataFrame, pc_cols: list[str], X: np.ndarray,
              mcs: int, ms: int, method: str, allow_single: bool,
              out_dir: Path) -> None:
    clusterer, labels = fit_hdbscan(X, mcs, ms, method, allow_single)
    out = df.copy()
    out["cluster"] = labels
    out["membership_probability"] = clusterer.probabilities_
    out["outlier_score"] = clusterer.outlier_scores_
    out.to_csv(out_dir / "hdbscan_final_labels.csv", index=False)

    n = len(out)
    rows = []
    for c, sub in out.groupby("cluster"):
        subj_counts = sub["subject_id"].value_counts()
        rows.append({
            "cluster": c,
            "n_events": len(sub),
            "pct_events": round(100 * len(sub) / n, 2),
            "n_subjects": int(sub["subject_id"].nunique()),
            "n_nights": int(sub.groupby(["subject_id", "night_id"]).ngroups)
                        if "night_id" in sub.columns else np.nan,
            "largest_subject_share_pct": round(100 * subj_counts.iloc[0] / len(sub), 1),
            "largest_subject": subj_counts.index[0],
            "mean_membership_probability": round(sub["membership_probability"].mean(), 3),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "hdbscan_final_cluster_summary.csv", index=False)

    try:
        rel_val = float(clusterer.relative_validity_)
    except Exception:
        rel_val = np.nan
    print(f"\nEindconfiguratie: min_cluster_size={mcs}, min_samples={ms}, {method} "
          f"(relative_validity {rel_val:.3f})")
    print(f"(cluster -1 = ruis; totaal {out['subject_id'].nunique()} subjects)")
    print(summary.to_string(index=False))

    config_txt = f"min_cluster_size = {mcs}, min_samples = {ms}, {method}"
    make_plots(out, pc_cols, out_dir, config_txt)


# =============================================================================
# SECTIE 5 — PLOTS
# =============================================================================

def _style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#999999")
    ax.tick_params(colors=MUTED_COLOR, labelsize=8)
    ax.grid(color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _cluster_ids(labels: pd.Series) -> list[int]:
    """Clusters gesorteerd op grootte (grootste krijgt kleur 1), zonder ruis."""
    counts = labels[labels != -1].value_counts()
    return list(counts.index)


def _style_map(labels: pd.Series) -> dict:
    ids = _cluster_ids(labels)
    if len(ids) > len(CLUSTER_COLORS):
        print(f"[LET OP] {len(ids)} clusters, maar maar {len(CLUSTER_COLORS)} kleuren -- "
              "kleinste clusters krijgen hergebruikte kleuren.")
    return {c: (CLUSTER_COLORS[i % len(CLUSTER_COLORS)], CLUSTER_MARKERS[i % len(CLUSTER_MARKERS)])
            for i, c in enumerate(ids)}


def _legend_handles(labels: pd.Series, styles: dict) -> list:
    n = len(labels)
    handles = []
    for c in sorted(styles):                       # legenda op clusternummer
        col, mk = styles[c]
        k = int((labels == c).sum())
        handles.append(Line2D([], [], ls="", marker=mk, markersize=7, color=col,
                              label=f"Cluster {c}  (n = {k:,}, {100 * k / n:.1f}%)"))
    k_noise = int((labels == -1).sum())
    if k_noise:
        handles.append(Line2D([], [], ls="", marker="o", markersize=6, color=NOISE_COLOR,
                              label=f"Noise  (n = {k_noise:,}, {100 * k_noise / n:.1f}%)"))
    return handles


def _scatter_pair(ax, df: pd.DataFrame, x: str, y: str, styles: dict,
                  annotate: bool = True, size: float = POINT_SIZE) -> None:
    """Ruis onderop, dan clusters van groot naar klein (kleine clusters liggen bovenop)."""
    noise = df[df["cluster"] == -1]
    if len(noise):
        ax.scatter(noise[x], noise[y], s=size * 0.7, c=NOISE_COLOR, alpha=0.35,
                   linewidths=0, rasterized=True, zorder=1)
    for c, (col, mk) in styles.items():
        sub = df[df["cluster"] == c]
        ax.scatter(sub[x], sub[y], s=size, c=col, marker=mk, alpha=POINT_ALPHA,
                   linewidths=0, rasterized=True, zorder=2)
        if annotate and len(sub):
            ax.annotate(f"{c}", (sub[x].median(), sub[y].median()), fontsize=9,
                        fontweight="bold", color=TEXT_COLOR, ha="center", va="center", zorder=4,
                        bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=col, lw=1.5))


def _discriminating_pair(df: pd.DataFrame, pc_cols: list[str]) -> tuple[str, str]:
    """
    Het PC-paar waarop de clusters het meest verschillen: per PC de verhouding
    tussen-cluster-variantie / binnen-cluster-variantie (F), de twee hoogste.
    Bij < 2 clusters: PC1 x PC5 (spectrale intensiteit x spectrale vorm).
    """
    sub = df[df["cluster"] != -1]
    if sub["cluster"].nunique() < 2:
        if not pc_cols[0].upper().startswith("PC") and all(c in pc_cols for c in FEATURE_PLOT_PAIRS[1]):
            return FEATURE_PLOT_PAIRS[1]
        return (pc_cols[0], pc_cols[4]) if len(pc_cols) >= 5 else (pc_cols[0], pc_cols[-1])
    grand = sub[pc_cols].mean()
    between = sub.groupby("cluster")[pc_cols].apply(
        lambda g: len(g) * (g.mean() - grand) ** 2).sum()
    within = sub.groupby("cluster")[pc_cols].apply(lambda g: ((g - g.mean()) ** 2).sum()).sum()
    f = (between / within).sort_values(ascending=False)
    pair = sorted(f.index[:2], key=pc_cols.index)
    print("Scheiding per PC (tussen/binnen-cluster-variantie): "
          + ", ".join(f"{k} {v:.2f}" for k, v in f.items()))
    return pair[0], pair[1]


def _axis_label(pc: str) -> str:
    return PC_LABELS.get(pc, pc)


def plot_main(df: pd.DataFrame, pc_cols: list[str], out_dir: Path, config_txt: str) -> None:
    """Figuur voor de resultatensectie: PC1 x PC2 en het meest scheidende PC-paar."""
    styles = _style_map(df["cluster"])
    first = FEATURE_PLOT_PAIRS[0]
    if not pc_cols[0].upper().startswith("PC") and all(c in pc_cols for c in first):
        pairs = [first]
    else:
        pairs = [(pc_cols[0], pc_cols[1])]
    disc = _discriminating_pair(df, pc_cols)
    if set(disc) != set(pairs[0]):
        pairs.append(disc)
    elif len(pc_cols) >= 4:
        pairs.append((pc_cols[2], pc_cols[3]))

    fig, axes = plt.subplots(1, len(pairs), figsize=(6.2 * len(pairs), 5.6))
    axes = np.atleast_1d(axes)
    for i, (ax, (x, y)) in enumerate(zip(axes, pairs)):
        _scatter_pair(ax, df, x, y, styles)
        ax.set_xlabel(_axis_label(x), fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel(_axis_label(y), fontsize=10, color=TEXT_COLOR)
        ax.text(-0.12, 1.03, "AB"[i], transform=ax.transAxes, fontsize=14,
                fontweight="bold", color=TEXT_COLOR)
        _style_axes(ax)

    fig.legend(handles=_legend_handles(df["cluster"], styles), loc="lower center",
               ncol=min(4, len(styles) + 1), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.text(0.99, 0.99, f"HDBSCAN: {config_txt}", ha="right", va="top",
             fontsize=8, color=MUTED_COLOR)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_clusters_main.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figuur opgeslagen: {out_dir / 'fig_clusters_main.png'} (+ .pdf)")


def plot_pairs(df: pd.DataFrame, pc_cols: list[str], out_dir: Path, config_txt: str) -> None:
    """
    Corner plot: onder de diagonaal elk PC-paar als scatter, op de diagonaal de
    verdeling per cluster (dichtheid, zodat kleine clusters zichtbaar blijven).
    """
    styles = _style_map(df["cluster"])
    k = len(pc_cols)
    fig, axes = plt.subplots(k, k, figsize=(2.6 * k, 2.6 * k))
    for i, yi in enumerate(pc_cols):
        for j, xj in enumerate(pc_cols):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                lo, hi = np.percentile(df[xj], [0.5, 99.5])
                bins = np.linspace(lo, hi, 40)
                noise = df.loc[df["cluster"] == -1, xj]
                if len(noise):
                    ax.hist(noise, bins=bins, density=True, histtype="stepfilled",
                            color=NOISE_COLOR, alpha=0.5)
                for c, (col, _) in styles.items():
                    ax.hist(df.loc[df["cluster"] == c, xj], bins=bins, density=True,
                            histtype="step", color=col, lw=1.8)
                ax.set_yticks([])
            else:
                _scatter_pair(ax, df, xj, yi, styles, annotate=False, size=2.5)
            _style_axes(ax)
            if i == k - 1:
                ax.set_xlabel(xj, fontsize=10, color=TEXT_COLOR)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(yi, fontsize=10, color=TEXT_COLOR)
            elif j > 0:
                ax.set_yticklabels([])

    # legenda + PC-interpretatie in de lege rechterbovenhoek
    fig.legend(handles=_legend_handles(df["cluster"], styles), loc="upper right",
               bbox_to_anchor=(0.98, 0.98), frameon=False, fontsize=10)
    pc_txt = "\n".join(_axis_label(pc) for pc in pc_cols)
    fig.text(0.98, 0.70, pc_txt, ha="right", va="top", fontsize=9, color=MUTED_COLOR)
    fig.text(0.98, 0.58, f"HDBSCAN: {config_txt}", ha="right", va="top",
             fontsize=8, color=MUTED_COLOR)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_clusters_pairs.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figuur opgeslagen: {out_dir / 'fig_clusters_pairs.png'} (+ .pdf)")


def make_plots(df: pd.DataFrame, pc_cols: list[str], out_dir: Path, config_txt: str) -> None:
    n_clusters = df.loc[df["cluster"] != -1, "cluster"].nunique()
    if n_clusters == 0:
        print("[LET OP] Geen clusters gevonden (alles ruis) -- plots tonen alleen ruis.")
    plot_main(df, pc_cols, out_dir, config_txt)
    plot_pairs(df, pc_cols, out_dir, config_txt)


# =============================================================================
# HOOFDLOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tag", default="whitened",
                        help="Submap-naam voor deze run (bijv. whitened / unwhitened)")
    parser.add_argument("--min-cluster-size", type=int, nargs="+", default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, nargs="+", default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--selection", nargs="+", choices=["eom", "leaf"], default=DEFAULT_SELECTION)
    parser.add_argument("--allow-single-cluster", action="store_true",
                        help="Sta toe dat HDBSCAN alles als één cluster ziet (anders "
                             "forceert het een splitsing of labelt het alles als ruis)")
    parser.add_argument("--quick", action="store_true", help="Klein grid voor een snelle check")
    parser.add_argument("--top-n", type=int, default=TOP_N_STABILITY,
                        help="Aantal top-configuraties voor de stabiliteitscheck")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--plot-only", action="store_true",
                        help="Alleen de plots opnieuw maken uit hdbscan_final_labels.csv "
                             "in de <tag>-map (geen grid/stabiliteit)")
    parser.add_argument("--final", nargs=3, metavar=("MCS", "MS", "METHOD"),
                        help="Vaste eindconfiguratie, bijv. --final 130 25 eom "
                             "(default: de best gerangschikte)")
    args = parser.parse_args()

    if args.quick:
        args.min_cluster_size = QUICK_MIN_CLUSTER_SIZE
        args.min_samples = QUICK_MIN_SAMPLES
        args.selection = QUICK_SELECTION
        args.n_resamples = min(args.n_resamples, 3)
        print("--quick: klein grid.\n")

    out_dir = args.output_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        labels_path = out_dir / "hdbscan_final_labels.csv"
        df_lab = read_csv_robust(labels_path)
        pc_cols = get_feature_cols(df_lab)
        make_plots(df_lab, pc_cols, out_dir, "see hdbscan_final_cluster_summary.csv")
        return

    df, pc_cols = load_scores(args.input)
    X = df[pc_cols].to_numpy()
    subjects = df["subject_id"].astype(str).to_numpy()

    n_cfg = len(args.min_cluster_size) * len(args.min_samples) * len(args.selection)
    print(f"Grid: {len(args.min_cluster_size)} x {len(args.min_samples)} x "
          f"{len(args.selection)} = {n_cfg} configuraties "
          f"(allow_single_cluster={args.allow_single_cluster})\n")

    # --- Stap 1: grid ---
    grid = run_grid(X, args.min_cluster_size, args.min_samples, args.selection,
                    args.allow_single_cluster)
    grid.to_csv(out_dir / "hdbscan_grid.csv", index=False)
    ranked = rank_grid(grid)
    ranked.to_csv(out_dir / "hdbscan_ranked.csv", index=False)

    n_valid = int(ranked["valid"].sum())
    print(f"\n{n_valid} van {len(ranked)} configuraties valide "
          f"({MIN_CLUSTERS}-{MAX_CLUSTERS} clusters, <= {MAX_NOISE_FRACTION:.0%} ruis).")
    print(f"  1 cluster:            {int((ranked['n_clusters'] == 1).sum())}")
    print(f"  0 clusters (alles ruis): {int((ranked['n_clusters'] == 0).sum())}")
    print(f"  > {MAX_NOISE_FRACTION:.0%} ruis:          {int((ranked['noise_fraction'] > MAX_NOISE_FRACTION).sum())}")
    if n_valid == 0:
        print("[LET OP] Geen enkele configuratie valide -- aanwijzing dat er geen duidelijke "
              "clusterstructuur is (of dat het grid aangepast moet worden).")

    show = ["min_cluster_size", "min_samples", "selection", "valid", "relative_validity",
            "n_clusters", "noise_fraction", "largest_cluster_fraction", "cluster_sizes"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\nTop 10:")
        print(ranked[show].head(10).to_string(index=False))

    # --- Stap 2: stabiliteit van de top-N ---
    top = ranked.head(args.top_n)
    print(f"\nStabiliteit top {len(top)} ({args.n_resamples}x {SUBJECT_FRACTION:.0%} van de subjects):")
    stab_rows = []
    for _, r in top.iterrows():
        s = stability(X, subjects, int(r["min_cluster_size"]), int(r["min_samples"]),
                      r["selection"], args.allow_single_cluster, args.n_resamples,
                      SUBJECT_FRACTION, RANDOM_STATE)
        stab_rows.append({"min_cluster_size": int(r["min_cluster_size"]),
                          "min_samples": int(r["min_samples"]), "selection": r["selection"],
                          "valid": bool(r["valid"]), "n_clusters_full": int(r["n_clusters"]), **s})
        print(f"  mcs={int(r['min_cluster_size']):<4} ms={int(r['min_samples']):<4} {r['selection']:<4} "
              f"-> ARI {s['ari_mean']:.2f} ± {s['ari_std']:.2f} (min {s['ari_min']:.2f}), "
              f"clusters in subsets {s['n_clusters_subsets_range']}")
    pd.DataFrame(stab_rows).to_csv(out_dir / "hdbscan_stability.csv", index=False)

    # --- Stap 3: eindconfiguratie ---
    if args.final:
        mcs, ms, method = int(args.final[0]), int(args.final[1]), args.final[2]
        if method not in ("eom", "leaf"):
            raise ValueError("--final METHOD moet 'eom' of 'leaf' zijn.")
    else:
        best = ranked.iloc[0]
        mcs, ms, method = int(best["min_cluster_size"]), int(best["min_samples"]), best["selection"]
    final_fit(df, pc_cols, X, mcs, ms, method, args.allow_single_cluster, out_dir)

    print(f"\nAlles opgeslagen in: {out_dir}")
    print("  - hdbscan_grid.csv / hdbscan_ranked.csv")
    print("  - hdbscan_stability.csv                 <- ARI over subject-subsets")
    print("  - hdbscan_final_labels.csv              <- label per event (-1 = ruis)")
    print("  - hdbscan_final_cluster_summary.csv     <- grootte + subject-spreiding per cluster")
    print("  - fig_clusters_main.png/.pdf            <- resultatensectie")
    print("  - fig_clusters_pairs.png/.pdf           <- appendix: alle PC-paren")
    print("\nVolgende stap: beschrijf de clusters met de ORIGINELE features "
          "(arousal_features_reduced_original.csv) -- relative_validity en ARI zijn een "
          "rangschikking, geen bewijs van een betekenisvol subtype.")


if __name__ == "__main__":
    main()