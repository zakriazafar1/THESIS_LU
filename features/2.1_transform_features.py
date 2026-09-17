"""
=============================================================================
2.1_transform_features.py

Transformatie (log) van de arousal-featurematrix (output van 1_feature_matrix.py). 

Stappenplan:
  1. Visualiseer de verdeling van elke feature (histogram + skewness/kurtosis),
     en check op missings/inf.
  2. Transform skewed features: voor ELKE feature wordt automatisch gekozen
     tussen geen transform / signed-log1p / signed-sqrt, op basis van welke
     variant de laagste |skew| oplevert (signed = sign(x)*f(|x|), zodat het
     ook correct zou werken mocht een feature negatieve waarden bevatten).
     -- Zet features in FORCE_UNTOUCHED_COLS als je ze expliciet nooit wil
     transformeren. Skew wordt na transformatie opnieuw berekend en
     weggeschreven, samen met welke variant per feature gekozen is
     (transform_choices.csv).

  Scaling gebeurt in 2.2_scale_features.py, dat de hier weggeschreven
  arousal_feature_matrix_transformed.csv als input gebruikt.

Gebruik:
  python 2.1_transform_features.py
  python 2.1_transform_features.py --inspect-distributions   # print tabellen ook naar console
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# =============================================================================
# CONFIGURATIE
# =============================================================================

DEFAULT_INPUT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices\arousal_feature_matrix.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\transformed"
)

# Metadata kolommen, dus uitgesloten van distributie-plots en transformatie. 
# stage_rk is metadata voor interpretatie/descriptive_note, geen clustering-input. 
# duration_sec is WEL een feature. 

METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk"
]

N_COLS_GRID = 5  # aantal subplots per rij in de histogram-grid

# Zet hier features in die je expliciet NOOIT wil transformeren (bv. om domein-redenen) 
FORCE_UNTOUCHED_COLS: list[str] = []

# =============================================================================
# STAP 1 — INLADEN
# =============================================================================

def load_feature_matrix(path: Path) -> pd.DataFrame:
    """
    Leest de featurematrix in. Detecteert het scheidingsteken automatisch
    (sep=None + engine="python") i.p.v. altijd komma aan te nemen, en checkt
    daarna of numerieke kolommen alsnog als tekst zijn binnengekomen (het
    Excel-NL-scenario: puntkomma als veld-scheiding EN komma als decimaal-
    teken, bv. "1,234" i.p.v. "1.234") -- zo ja, dan wordt opnieuw ingelezen
    met decimal=",". 1_feature_matrix.py zelf schrijft altijd standaard-CSV,
    maar garandeert niet dat het bestand nooit per ongeluk in Excel met een
    NL-locale geopend en opgeslagen wordt.
    """
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

    print(f"Featurematrix geladen: {path}")
    print(f"Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Alle kolommen behalve de metadata-kolommen -> dit zijn de clustering-features."""
    return [c for c in df.columns if c not in METADATA_COLS]


# =============================================================================
# STAP 2 - MISSINGS / INF CHECK
# =============================================================================

def summarize_missingness(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Telt per feature: aantal NaN, aantal +-inf, en % van totaal."""
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
# STAP 3 — DISTRIBUTIES VISUALISEREN (voor/na transformatie)
# =============================================================================

def compute_distribution_stats(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Skewness en kurtosis per feature (op de niet-NaN, eindige waarden)."""
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
            "has_negative": bool((vals < 0).any()),
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

    for j in range(n_feats, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Distributie-grid opgeslagen: {out_path}")


# =============================================================================
# STAP 4 — TRANSFORMATIE 
# =============================================================================

def signed_log1p(x: pd.Series) -> pd.Series:
    """sign(x) * log1p(|x|) -- voor strikt-positieve data identiek aan log1p."""
    x = x.replace([np.inf, -np.inf], np.nan)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_sqrt(x: pd.Series) -> pd.Series:
    """sign(x) * sqrt(|x|) -- voor strikt-positieve data identiek aan sqrt."""
    x = x.replace([np.inf, -np.inf], np.nan)
    return np.sign(x) * np.sqrt(np.abs(x))


def classify_transform_columns(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Verdeelt de features in: auto_cols (gaan door de none/log1p/sqrt-selectie)
    en forced_untouched_cols (expliciet uitgesloten via FORCE_UNTOUCHED_COLS).
    """
    forced_untouched = [c for c in feature_cols if c in FORCE_UNTOUCHED_COLS]
    auto_cols = [c for c in feature_cols if c not in FORCE_UNTOUCHED_COLS]
    return auto_cols, forced_untouched


def apply_best_transform_group(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Voor elke kolom in cols: probeer geen transform, signed-log1p en signed-sqrt,
    en kies de variant met de laagste |skew|. Geeft de getransformeerde df terug
    plus een keuze-overzicht (skew voor elke variant + welke gekozen is).
    """
    out = df.copy()
    choices = []
    for c in cols:
        raw = df[c].replace([np.inf, -np.inf], np.nan)
        candidates = {
            "geen (origineel)": raw,
            "log1p (signed)": signed_log1p(raw),
            "sqrt (signed)": signed_sqrt(raw),
        }
        skews = {}
        for name, vals in candidates.items():
            finite = vals.dropna()
            skews[name] = skew(finite) if len(finite) >= 3 else np.nan

        valid = {k: v for k, v in skews.items() if pd.notna(v)}
        if not valid:
            chosen = "geen (origineel)"
        else:
            chosen = min(valid, key=lambda k: abs(valid[k]))

        out[c] = candidates[chosen]
        choices.append({
            "feature": c,
            "skew_geen": round(skews["geen (origineel)"], 3) if pd.notna(skews["geen (origineel)"]) else np.nan,
            "skew_log1p": round(skews["log1p (signed)"], 3) if pd.notna(skews["log1p (signed)"]) else np.nan,
            "skew_sqrt": round(skews["sqrt (signed)"], 3) if pd.notna(skews["sqrt (signed)"]) else np.nan,
            "chosen_transform": chosen,
            "skew_after": round(skews[chosen], 3) if pd.notna(skews[chosen]) else np.nan,
        })
    return out, pd.DataFrame(choices)


# =============================================================================
# HOOFDLOOP
# =============================================================================

def run_steps_1_and_2(df: pd.DataFrame, feature_cols: list[str], verbose: bool) -> pd.DataFrame:
    # --- Stap 1 ---
    missing_summary = summarize_missingness(df, feature_cols)
    dist_stats = compute_distribution_stats(df, feature_cols)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing_summary.to_csv(OUTPUT_DIR / "missing_inf_summary.csv", index=False)
    dist_stats.to_csv(OUTPUT_DIR / "distribution_stats.csv", index=False)
    plot_distributions(df, feature_cols, OUTPUT_DIR / "feature_distributions.png")
    print(f"\nStap 1 opgeslagen in: {OUTPUT_DIR}")
    print("  - missing_inf_summary.csv\n  - distribution_stats.csv\n  - feature_distributions.png")

    # --- Stap 2 ---
    auto_cols, forced_untouched_cols = classify_transform_columns(feature_cols)

    df_t = df.copy()
    df_t_auto, transform_choices = apply_best_transform_group(df, auto_cols)
    for c in auto_cols:
        df_t[c] = df_t_auto[c]
    # forced_untouched_cols: blijven ongemoeid, df_t heeft daar nog de originele waarden.

    dist_stats_after = compute_distribution_stats(df_t, feature_cols)
    df_t.to_csv(OUTPUT_DIR / "arousal_feature_matrix_transformed.csv", index=False)
    dist_stats_after.to_csv(OUTPUT_DIR / "distribution_stats_transformed.csv", index=False)
    if len(transform_choices):
        transform_choices.to_csv(OUTPUT_DIR / "transform_choices.csv", index=False)
    plot_distributions(df_t, feature_cols, OUTPUT_DIR / "feature_distributions_transformed.png")

    n_log = (transform_choices["chosen_transform"] == "log1p (signed)").sum() if len(transform_choices) else 0
    n_sqrt = (transform_choices["chosen_transform"] == "sqrt (signed)").sum() if len(transform_choices) else 0
    n_none = (transform_choices["chosen_transform"] == "geen (origineel)").sum() if len(transform_choices) else 0
    print(f"\nStap 2: per feature automatisch gekozen tussen geen/log1p/sqrt o.b.v. laagste |skew|: "
          f"{n_log} log1p, {n_sqrt} sqrt, {n_none} geen transform.")
    if forced_untouched_cols:
        print(f"  Expliciet ongemoeid (FORCE_UNTOUCHED_COLS): {forced_untouched_cols}")
    if len(transform_choices):
        print(transform_choices.to_string(index=False))
    print("  - arousal_feature_matrix_transformed.csv\n  - distribution_stats_transformed.csv"
          "\n  - transform_choices.csv\n  - feature_distributions_transformed.png")

    if verbose:
        print("\n--- Missing / inf overzicht ---")
        print(missing_summary.to_string(index=False))
        print("\n--- Skewness / kurtosis vóór transformatie ---")
        print(dist_stats.to_string(index=False))
        print("\n--- Skewness / kurtosis NA transformatie ---")
        print(dist_stats_after.to_string(index=False))

    return df_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help="Pad naar arousal_feature_matrix.csv")
    parser.add_argument("--inspect-distributions", action="store_true",
                         help="Print de volledige tabellen ook naar de console")
    args = parser.parse_args()

    df = load_feature_matrix(args.input)
    feature_cols = get_feature_columns(df)
    print(f"\n{len(feature_cols)} features (metadata-kolommen uitgesloten): {feature_cols}")

    run_steps_1_and_2(df, feature_cols, verbose=args.inspect_distributions)


if __name__ == "__main__":
    main()
