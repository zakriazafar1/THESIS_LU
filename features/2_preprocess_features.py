"""
=============================================================================
preprocess_features.py

Preprocessing van de arousal-featurematrix (output van feature_matrix.py),
vóór clustering.

Stappenplan (dit script implementeert nu STAP 1 en STAP 2):
  1. Visualiseer de verdeling van elke feature (histogram + skewness/kurtosis),
     en check op missings/inf.
  2. Zoek per feature naar outliers via de adjusted-boxplot-methode (medcouple-
     gecorrigeerde IQR-grenzen, robuust voor scheve verdelingen) + boxplots,
     en list de meest extreme individuele events op zodat je kan checken of
     het legitieme waarden of artefacten zijn.
  3. Transform skewed features.
  4. Standardize (zero mean / unit variance).

Gebruik:
  python preprocess_features.py
  python preprocess_features.py --input pad/naar/arousal_feature_matrix.csv
  python preprocess_features.py --inspect-distributions   # print tabellen ook naar console
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from statsmodels.stats.stattools import medcouple

# =============================================================================
# CONFIGURATIE
# =============================================================================

# Zelfde EVENTS_DIR als in feature_matrix.py -> daar staat arousal_feature_matrix.csv
DEFAULT_INPUT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices\arousal_feature_matrix.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing"
)

# Kolommen die GEEN feature zijn (identifiers / metadata), dus uitgesloten van
# distributie-plots, transformatie en standaardisatie. stage_rk is metadata
# voor interpretatie achteraf, geen clustering-input.
METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]

N_COLS_GRID = 5  # aantal subplots per rij in de histogram-/boxplot-grid

# Stap 2: outlier-detectie. "adjusted" = medcouple-gecorrigeerde IQR-fences
# (Hubert & Vandervieren 2008), robuust voor scheve verdelingen -> aanbevolen
# hier, want de meeste features (ratio's) zijn sterk rechts-scheef. "classic"
# = de gewone symmetrische Tukey-regel (Q1 - mult*IQR / Q3 + mult*IQR), die
# bij scheve data structureel te veel punten aan de lange staart flagt.
OUTLIER_METHOD = "adjusted"  # "adjusted" of "classic"
OUTLIER_IQR_MULT = 1.5
TOP_N_OUTLIER_EVENTS = 20  # max aantal individuele outlier-events per feature dat wordt weggeschreven

# Stap 3: features met |skew| boven deze drempel krijgen een signed-log1p-transform.
SKEW_THRESHOLD = 1.0


# =============================================================================
# SECTIE 1 — INLADEN
# =============================================================================

def load_feature_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Featurematrix geladen: {path}")
    print(f"Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Alle kolommen behalve de metadata-kolommen -> dit zijn de clustering-features."""
    return [c for c in df.columns if c not in METADATA_COLS]


# =============================================================================
# SECTIE 2 — MISSINGS / INF CHECK
# =============================================================================

def summarize_missingness(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Telt per feature: aantal NaN, aantal +-inf, en % van totaal.
    inf wordt apart geteld van NaN, want safe_ratio() in feature_matrix.py
    voorkomt exacte /0 maar niet per se een extreem kleine noemer.
    """
    rows = []
    n = len(df)
    for col in feature_cols:
        vals = df[col]
        n_nan = vals.isna().sum()
        n_inf = np.isinf(pd.to_numeric(vals, errors="coerce").to_numpy(dtype="float64", na_value=0.0)).sum()
        rows.append({
            "feature": col,
            "n_missing": n_nan,
            "pct_missing": round(100 * n_nan / n, 2) if n else np.nan,
            "n_inf": n_inf,
            "pct_inf": round(100 * n_inf / n, 2) if n else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)
    return summary


# =============================================================================
# SECTIE 3 — DISTRIBUTIES VISUALISEREN
# =============================================================================

def compute_distribution_stats(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Skewness en kurtosis per feature (op de niet-NaN, eindige waarden), als
    hulpmiddel om te bepalen welke features in STAP 2 een log-transform
    nodig hebben. Vuistregel: |skew| > 1 -> kandidaat voor transformatie.
    """
    rows = []
    for col in feature_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) < 3:
            rows.append({"feature": col, "skew": np.nan, "kurtosis": np.nan,
                          "min": np.nan, "max": np.nan, "has_negative": np.nan})
            continue
        rows.append({
            "feature": col,
            "skew": round(skew(vals), 3),
            "kurtosis": round(kurtosis(vals), 3),
            "min": round(vals.min(), 3),
            "max": round(vals.max(), 3),
            "has_negative": bool((vals < 0).any()),  # relevant voor log-transform (bv. ridge_drift)
        })
    stats = pd.DataFrame(rows)
    return stats.sort_values("skew", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def plot_distributions(df: pd.DataFrame, feature_cols: list[str], out_path: Path) -> None:
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

    # Lege subplots (als n_feats geen veelvoud van n_cols is) verbergen
    for j in range(n_feats, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Distributie-grid opgeslagen: {out_path}")

# =============================================================================
# SECTIE 4 — OUTLIER DETECTIE (STAP 2)
# =============================================================================

def compute_outlier_summary(df: pd.DataFrame, feature_cols: list[str],
                             iqr_mult: float = OUTLIER_IQR_MULT,
                             method: str = OUTLIER_METHOD) -> pd.DataFrame:
    """
    Per feature: Q1, Q3, IQR, de medcouple (robuuste scheefheidsmaat) en de
    outlier-grenzen.

    method="adjusted" (aanbevolen voor scheve features): de grenzen worden
    asymmetrisch aangepast a.d.h.v. de medcouple (MC), volgens Hubert &
    Vandervieren (2008):
        MC >= 0:  lower = Q1 - mult*IQR*exp(-4*MC),  upper = Q3 + mult*IQR*exp(3*MC)
        MC <  0:  lower = Q1 - mult*IQR*exp(-3*MC),  upper = Q3 + mult*IQR*exp(4*MC)
    Voor een rechts-scheve verdeling (MC > 0) schuift de bovengrens hierdoor
    verder naar rechts op -> minder valse outliers puur door de lange staart,
    terwijl de ondergrens juist iets strenger wordt (waar bij een rechts-scheve
    verdeling sowieso weinig lage uitschieters zitten).

    method="classic": de gewone symmetrische Tukey-regel (MC wordt genegeerd),
    puur ter vergelijking.
    """
    rows = []
    n = len(df)
    for col in feature_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) < 4:
            rows.append({"feature": col, "q1": np.nan, "q3": np.nan, "iqr": np.nan,
                         "medcouple": np.nan, "lower_bound": np.nan, "upper_bound": np.nan,
                         "n_outliers": np.nan, "pct_outliers": np.nan,
                         "n_low": np.nan, "n_high": np.nan})
            continue

        q1, q3 = vals.quantile([0.25, 0.75])
        iqr = q3 - q1
        mc = float(medcouple(vals.to_numpy())) if method == "adjusted" else 0.0

        if method == "adjusted":
            if mc >= 0:
                lower = q1 - iqr_mult * iqr * np.exp(-4 * mc)
                upper = q3 + iqr_mult * iqr * np.exp(3 * mc)
            else:
                lower = q1 - iqr_mult * iqr * np.exp(-3 * mc)
                upper = q3 + iqr_mult * iqr * np.exp(4 * mc)
        else:
            lower = q1 - iqr_mult * iqr
            upper = q3 + iqr_mult * iqr

        n_low = int((vals < lower).sum())
        n_high = int((vals > upper).sum())
        rows.append({
            "feature": col, "q1": round(q1, 3), "q3": round(q3, 3), "iqr": round(iqr, 3),
            "medcouple": round(mc, 3),
            "lower_bound": round(lower, 3), "upper_bound": round(upper, 3),
            "n_outliers": n_low + n_high,
            "pct_outliers": round(100 * (n_low + n_high) / n, 2) if n else np.nan,
            "n_low": n_low, "n_high": n_high,
        })
    summary = pd.DataFrame(rows).sort_values("pct_outliers", ascending=False).reset_index(drop=True)
    return summary


def flag_outlier_events(df: pd.DataFrame, outlier_summary: pd.DataFrame,
                         id_cols: list[str] = ("subject_id", "night_id", "event_idx"),
                         top_n: int = TOP_N_OUTLIER_EVENTS) -> pd.DataFrame:
    """
    Voor elke feature met n_outliers > 0: de top_n events die het verst buiten
    de IQR-grenzen vallen, met identifiers erbij (subject/nacht/event-index) en
    of het een lage of hoge outlier is. Handig om specifieke events op te zoeken
    en te checken of het een artefact is (bv. electrode-pop) of een legitiem
    extreem arousal-event.
    """
    id_cols = [c for c in id_cols if c in df.columns]
    rows = []
    for _, r in outlier_summary.dropna(subset=["n_outliers"]).iterrows():
        if r["n_outliers"] == 0:
            continue
        feat = r["feature"]
        vals = df[feat].replace([np.inf, -np.inf], np.nan)

        low_mask = vals < r["lower_bound"]
        high_mask = vals > r["upper_bound"]

        low = df.loc[low_mask, id_cols].copy()
        low["value"] = vals[low_mask]
        low["bound"] = "laag"
        low = low.sort_values("value", ascending=True).head(top_n)

        high = df.loc[high_mask, id_cols].copy()
        high["value"] = vals[high_mask]
        high["bound"] = "hoog"
        high = high.sort_values("value", ascending=False).head(top_n)

        combined = pd.concat([low, high], ignore_index=True)
        combined.insert(0, "feature", feat)
        rows.append(combined)

    if not rows:
        return pd.DataFrame(columns=["feature"] + id_cols + ["value", "bound"])
    return pd.concat(rows, ignore_index=True)


def _bxp_stats_from_bounds(vals: pd.Series, q1: float, q3: float,
                            lower_bound: float, upper_bound: float) -> dict:
    """
    Bouwt de stats-dict die ax.bxp() verwacht, maar met ONZE (evt. medcouple-
    aangepaste, asymmetrische) grenzen i.p.v. matplotlib's eigen standaard
    1.5*IQR-whiskers. Whiskers lopen tot de meest extreme datapunten die nog
    binnen de grenzen vallen; alles daarbuiten wordt als 'flier' (outlier-stip)
    getekend.
    """
    within = vals[(vals >= lower_bound) & (vals <= upper_bound)]
    whislo = within.min() if len(within) else vals.min()
    whishi = within.max() if len(within) else vals.max()
    fliers = vals[(vals < lower_bound) | (vals > upper_bound)]
    return {
        "med": vals.median(), "q1": q1, "q3": q3,
        "whislo": whislo, "whishi": whishi,
        "fliers": fliers.to_numpy(),
    }


def plot_boxplots(df: pd.DataFrame, feature_cols: list[str], outlier_summary: pd.DataFrame,
                   out_path: Path) -> None:
    """
    Grid van boxplots (1 per feature), met de outlier-grenzen zoals berekend
    in compute_outlier_summary (dus medcouple-aangepast als OUTLIER_METHOD =
    "adjusted"), niet matplotlib's eigen standaard-whiskers. Y-as op 'symlog'
    -schaal zodat een paar extreme outliers de box zelf niet onzichtbaar-klein
    maken -- puur voor visualisatie, de onderliggende waarden blijven ongewijzigd.
    """
    n_feats = len(feature_cols)
    n_cols = N_COLS_GRID
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    summary_by_feat = outlier_summary.set_index("feature")

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) and col in summary_by_feat.index and pd.notna(summary_by_feat.loc[col, "q1"]):
            row = summary_by_feat.loc[col]
            stats = _bxp_stats_from_bounds(vals, row["q1"], row["q3"], row["lower_bound"], row["upper_bound"])
            ax.bxp([stats], showfliers=True,
                   flierprops={"markersize": 3, "markeredgecolor": "crimson"})
            ax.set_yscale("symlog")
            n_out = int(row["n_outliers"])
            mc = row["medcouple"]
            ax.set_title(f"{col}\nn_outliers={n_out}, mc={mc:.2f}", fontsize=9)
        else:
            ax.set_title(f"{col}\n(geen data)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xticks([])

    for j in range(n_feats, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Boxplot-grid opgeslagen: {out_path}")


# =============================================================================
# SECTIE 5 — TRANSFORMATIE (STAP 3)
# =============================================================================

def signed_log1p(x: pd.Series) -> pd.Series:
    """
    log1p die ook met negatieve waarden overweg kan: sign(x) * log1p(|x|).
    Voor strikt-positieve features (de meeste ratio's) is dit identiek aan
    een gewone log1p; voor features als ridge_drift (kan negatief zijn)
    blijft het teken behouden terwijl de staarten worden samengedrukt.
    """
    x = x.replace([np.inf, -np.inf], np.nan)
    return np.sign(x) * np.log1p(np.abs(x))


def select_transform_columns(dist_stats: pd.DataFrame, threshold: float = SKEW_THRESHOLD) -> list[str]:
    """Features met |skew| > threshold -> kandidaten voor signed-log1p."""
    valid = dist_stats.dropna(subset=["skew"])
    return valid.loc[valid["skew"].abs() > threshold, "feature"].tolist()


def apply_transforms(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Past signed-log1p toe op de opgegeven kolommen, laat de rest ongemoeid."""
    out = df.copy()
    for c in cols:
        out[c] = signed_log1p(out[c])
    return out



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help="Pad naar arousal_feature_matrix.csv")
    parser.add_argument("--inspect-distributions", action="store_true",
                         help="Print de volledige missing/inf- en skew/kurtosis-tabellen ook naar de console "
                              "(de 3 outputbestanden worden hoe dan ook altijd opgeslagen)")
    args = parser.parse_args()

    df = load_feature_matrix(args.input)
    feature_cols = get_feature_columns(df)
    print(f"\n{len(feature_cols)} features (metadata-kolommen uitgesloten): {feature_cols}")

    # Stap 1 draait altijd -> deze 3 bestanden worden sowieso opgeslagen.
    missing_summary = summarize_missingness(df, feature_cols)
    dist_stats = compute_distribution_stats(df, feature_cols)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing_summary.to_csv(OUTPUT_DIR / "missing_inf_summary.csv", index=False)
    dist_stats.to_csv(OUTPUT_DIR / "distribution_stats.csv", index=False)
    plot_distributions(df, feature_cols, OUTPUT_DIR / "feature_distributions.png")
    print(f"\nOverzichten opgeslagen in: {OUTPUT_DIR}")
    print("  - missing_inf_summary.csv")
    print("  - distribution_stats.csv")
    print("  - feature_distributions.png")

    # Stap 2 draait ook altijd -> outlier-detectie per feature (IQR-methode) + boxplots.
    outlier_summary = compute_outlier_summary(df, feature_cols)
    outlier_events = flag_outlier_events(df, outlier_summary)

    outlier_summary.to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)
    outlier_events.to_csv(OUTPUT_DIR / "outlier_events.csv", index=False)
    plot_boxplots(df, feature_cols, outlier_summary, OUTPUT_DIR / "feature_boxplots.png")
    print("  - outlier_summary.csv")
    print("  - outlier_events.csv")
    print("  - feature_boxplots.png")

    if args.inspect_distributions:
        print("\n--- Missing / inf overzicht ---")
        print(missing_summary.to_string(index=False))
        print("\n--- Skewness / kurtosis (gesorteerd op |skew|) ---")
        print(dist_stats.to_string(index=False))
        print("\nVuistregel: |skew| > 1 -> kandidaat voor log1p-transform in stap 3.")
        print("Let op: features met has_negative=True (bv. ridge_drift) kunnen niet")
        print("zomaar met log/log1p getransformeerd worden -> signed-log overwegen.")
        print("\n--- Outlier-overzicht (IQR-methode, gesorteerd op %) ---")
        print(outlier_summary.to_string(index=False))


if __name__ == "__main__":
    main()