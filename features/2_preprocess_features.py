"""
=============================================================================
preprocess_features.py

Preprocessing van de arousal-featurematrix (output van feature_matrix.py),
vóór clustering.

Stappenplan:
  1. Visualiseer de verdeling van elke feature (histogram + skewness/kurtosis),
     en check op missings/inf.
  2. Transform skewed features (domein-gebaseerd, niet automatisch op skew):
       - log1p op alle *_ratio / *_peak_ratio kolommen en op ridge_onset_hz,
         ridge_peak_hz, ridge_end_hz.
       - op duration_sec en ridge_drift: zowel log1p (signed, want ridge_drift
         kan negatief zijn) als sqrt (signed) proberen, en de variant met de
         laagste |skew| kiezen.
       - motion_rms en oxy_amp_ratio blijven ongemoeid.
     Skew wordt na transformatie opnieuw berekend en weggeschreven.
  3. Outlier-detectie OP DE GETRANSFORMEERDE SCHAAL, met de klassieke IQR-regel
     (1.5*IQR) -- op de log-schaal is de verdeling redelijk symmetrisch, dus is
     de symmetrische regel weer betrouwbaar. Elke outlier-event wordt
     weggeschreven met een automatisch gegenereerde, PUUR BESCHRIJVENDE notitie
     (duur, sleep stage, welk kanaal/band, hoeveel buiten de grens) -- dit is
     GEEN klinisch oordeel, gewoon de relevante feiten verzameld zodat jij/
     Lucija kan beoordelen of het een plausibel arousal is of een artefact.
  4. Beslissing toepassen: vul de 'decision'-kolom in outlier_events.csv met
     'winsorize' / 'remove' / 'keep' (leeg = keep) en run het script opnieuw
     met --apply-decisions <pad naar ingevulde outlier_events.csv>.
       - winsorize: die ene (event, feature)-waarde capt op het 1e/99e
         percentiel van die feature (op de getransformeerde schaal).
       - remove: het hele event (alle features) wordt uit de matrix verwijderd
         -- als een event een artefact is, geldt dat voor de hele opname op
         dat moment, niet alleen voor de ene feature die de grens overschreed.
  5. Standaardiseren: StandardScaler (z-score) als default; voor kolommen die
     na stap 2/4 nog steeds |skew| > ROBUST_SKEW_THRESHOLD hebben wordt
     RobustScaler gebruikt (mediaan/IQR i.p.v. mean/std), minder gevoelig voor
     de resterende extremen.

Gebruik:
  python preprocess_features.py
      -> stap 1, 2, 3. Schrijft o.a. outlier_events.csv weg met een lege
         'decision'-kolom.
  python preprocess_features.py --apply-decisions pad/naar/ingevulde_outlier_events.csv
      -> stap 4 (winsorize/remove op basis van de ingevulde decisions) + stap 5
         (standaardiseren). Schrijft de uiteindelijke, klaar-voor-clustering
         featurematrix weg.
  python preprocess_features.py --inspect-distributions   # print tabellen ook naar console
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
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing"
)

# Kolommen die GEEN feature zijn (identifiers / metadata), dus uitgesloten van
# distributie-plots, transformatie en standaardisatie. stage_rk is metadata
# voor interpretatie/descriptive_note, geen clustering-input. duration_sec is
# WEL een feature (zie stap 2).
METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]
ID_COLS = ["subject_id", "night_id", "event_idx"]  # voor het opzoeken van individuele events

N_COLS_GRID = 5  # aantal subplots per rij in de histogram-grid

# Stap 2: expliciete, domein-gebaseerde transformatiegroepen (i.p.v. automatisch op skew-drempel).
LOG1P_SUFFIX = "_ratio"                                            # vangt zowel *_ratio als *_peak_ratio
LOG1P_EXPLICIT_COLS = ["ridge_onset_hz", "ridge_peak_hz", "ridge_end_hz"]
DUAL_TRANSFORM_COLS = ["duration_sec", "ridge_drift"]               # log1p vs sqrt, beste van de twee kiezen
UNTOUCHED_COLS = ["motion_rms", "oxy_amp_ratio"]

# Stap 3: IQR-methode voor outlier-detectie op de getransformeerde schaal.
OUTLIER_IQR_MULT = 1.5

# Stap 4: winsorize-grenzen (percentiel op de getransformeerde schaal).
WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99

# Stap 5: kolommen met |skew| boven deze drempel (ná stap 2/4) krijgen RobustScaler i.p.v. StandardScaler.
ROBUST_SKEW_THRESHOLD = 2.0

STAGE_LABELS = {0: "wake", 1: "N1", 2: "N2", 3: "N3", 4: "N4", 5: "REM"}


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
# SECTIE 3 — DISTRIBUTIES VISUALISEREN (STAP 1)
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
# SECTIE 4 — TRANSFORMATIE (STAP 2)
# =============================================================================

def signed_log1p(x: pd.Series) -> pd.Series:
    """sign(x) * log1p(|x|) -- voor strikt-positieve data identiek aan log1p."""
    x = x.replace([np.inf, -np.inf], np.nan)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_sqrt(x: pd.Series) -> pd.Series:
    """sign(x) * sqrt(|x|) -- voor strikt-positieve data identiek aan sqrt."""
    x = x.replace([np.inf, -np.inf], np.nan)
    return np.sign(x) * np.sqrt(np.abs(x))


def classify_transform_columns(feature_cols: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Verdeelt de features in de 3 domein-gebaseerde groepen uit stap 2.
    Geeft ook een lijst 'unclassified' terug -- kolommen die in geen van de
    groepen vallen, bv. als de featurematrix in de toekomst uitgebreid wordt.
    Die worden NIET getransformeerd, maar wel gemeld, zodat het niet stilletjes
    genegeerd wordt.
    """
    log1p_cols = [c for c in feature_cols if c.endswith(LOG1P_SUFFIX) or c in LOG1P_EXPLICIT_COLS]
    dual_cols = [c for c in feature_cols if c in DUAL_TRANSFORM_COLS]
    untouched_cols = [c for c in feature_cols if c in UNTOUCHED_COLS]

    classified = set(log1p_cols) | set(dual_cols) | set(untouched_cols)
    unclassified = [c for c in feature_cols if c not in classified]
    return log1p_cols, dual_cols, untouched_cols, unclassified


def apply_log1p_group(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = signed_log1p(out[c])
    return out


def apply_dual_transform_group(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Voor elke kolom in cols: probeer zowel signed-log1p als signed-sqrt, en
    kies de variant met de laagste |skew|. Geeft de getransformeerde df terug
    plus een keuze-overzicht (welke transform gekozen is en de skew vóór/na).
    """
    out = df.copy()
    choices = []
    for c in cols:
        raw = df[c].replace([np.inf, -np.inf], np.nan)
        skew_before = skew(raw.dropna()) if raw.dropna().shape[0] >= 3 else np.nan

        log_vals = signed_log1p(raw)
        sqrt_vals = signed_sqrt(raw)
        skew_log = skew(log_vals.dropna()) if log_vals.dropna().shape[0] >= 3 else np.nan
        skew_sqrt = skew(sqrt_vals.dropna()) if sqrt_vals.dropna().shape[0] >= 3 else np.nan

        if pd.isna(skew_log) and pd.isna(skew_sqrt):
            chosen, out[c] = "geen (te weinig data)", raw
            skew_after = skew_before
        elif pd.isna(skew_sqrt) or (not pd.isna(skew_log) and abs(skew_log) <= abs(skew_sqrt)):
            chosen, out[c], skew_after = "log1p (signed)", log_vals, skew_log
        else:
            chosen, out[c], skew_after = "sqrt (signed)", sqrt_vals, skew_sqrt

        choices.append({
            "feature": c, "skew_before": round(skew_before, 3) if pd.notna(skew_before) else np.nan,
            "chosen_transform": chosen,
            "skew_after": round(skew_after, 3) if pd.notna(skew_after) else np.nan,
        })
    return out, pd.DataFrame(choices)


# =============================================================================
# SECTIE 5 — OUTLIER DETECTIE (STAP 3)
# =============================================================================

def compute_outlier_summary(df: pd.DataFrame, feature_cols: list[str],
                             iqr_mult: float = OUTLIER_IQR_MULT) -> pd.DataFrame:
    """
    Klassieke Tukey-IQR-regel (Q1 - mult*IQR, Q3 + mult*IQR), toegepast op de
    getransformeerde (stap 2) kolommen -- daar redelijk symmetrisch, dus deze
    symmetrische regel is hier op zijn plaats.
    """
    rows = []
    n = len(df)
    for col in feature_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) < 4:
            rows.append({"feature": col, "q1": np.nan, "q3": np.nan, "iqr": np.nan,
                         "lower_bound": np.nan, "upper_bound": np.nan,
                         "n_outliers": np.nan, "pct_outliers": np.nan})
            continue
        q1, q3 = vals.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - iqr_mult * iqr
        upper = q3 + iqr_mult * iqr
        n_out = int(((vals < lower) | (vals > upper)).sum())
        rows.append({
            "feature": col, "q1": round(q1, 3), "q3": round(q3, 3), "iqr": round(iqr, 3),
            "lower_bound": round(lower, 3), "upper_bound": round(upper, 3),
            "n_outliers": n_out,
            "pct_outliers": round(100 * n_out / n, 2) if n else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values("pct_outliers", ascending=False).reset_index(drop=True)
    return summary


def _describe_feature(feature: str) -> str:
    """Vertaalt een kolomnaam naar een leesbare kanaal/band-omschrijving, voor de notitie."""
    if feature.startswith(("L_", "R_")):
        channel = "links (L)" if feature.startswith("L_") else "rechts (R)"
        band = feature.split("_")[1] if "_" in feature else feature
        metric = "piekamplitude" if "peak" in feature else "gemiddelde amplitude"
        return f"{band}-band, {channel} kanaal, {metric} t.o.v. whole-night mediaan"
    if feature.startswith("mean_"):
        band = feature.split("_")[1]
        return f"{band}-band, gemiddeld over L/R kanalen"
    if feature.startswith("ridge_"):
        return f"Morlet-ridge kenmerk ({feature})"
    return feature


def generate_descriptive_note(feature: str, raw_value: float, transformed_value: float,
                               bound: str, stage_rk, duration_sec) -> str:
    """
    PUUR BESCHRIJVEND, geen klinisch oordeel: verzamelt de feiten die relevant
    zijn om te beoordelen of dit een plausibel arousal-event is of een
    vermoedelijk artefact -- welk kanaal/band, hoe ver buiten de grens, sleep
    stage, en event-duur. De uiteindelijke beoordeling (winsorize/remove/keep)
    is aan jou/Lucija.
    """
    parts = [_describe_feature(feature)]
    parts.append(f"ruwe waarde={raw_value:.3g}, log/sqrt-getransformeerd={transformed_value:.3g} "
                 f"({'boven' if bound == 'hoog' else 'onder'} de IQR-grens op de getransformeerde schaal)")
    if pd.notna(stage_rk):
        parts.append(f"sleep stage={STAGE_LABELS.get(int(stage_rk), stage_rk)}")
    if pd.notna(duration_sec):
        parts.append(f"event-duur={duration_sec:.1f}s")
    return "; ".join(parts)


def flag_outlier_events(df_raw: pd.DataFrame, df_transformed: pd.DataFrame,
                         outlier_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Voor elke feature met outliers: alle events die buiten de IQR-grenzen
    vallen (op de getransformeerde schaal), met identifiers, ruwe + getrans-
    formeerde waarde, een automatisch gegenereerde descriptive_note, en een
    LEGE 'decision'-kolom (in te vullen met winsorize/remove/keep vóór je
    --apply-decisions draait). '_row_index' is de originele rij-index in de
    featurematrix, nodig om de decision straks weer terug te koppelen.
    """
    id_cols = [c for c in ID_COLS if c in df_raw.columns]
    rows = []
    for _, r in outlier_summary.dropna(subset=["n_outliers"]).iterrows():
        if r["n_outliers"] == 0:
            continue
        feat = r["feature"]
        vals = df_transformed[feat].replace([np.inf, -np.inf], np.nan)
        mask = (vals < r["lower_bound"]) | (vals > r["upper_bound"])
        idx = df_raw.index[mask]

        for i in idx:
            bound = "laag" if vals.loc[i] < r["lower_bound"] else "hoog"
            stage = df_raw.loc[i, "stage_rk"] if "stage_rk" in df_raw.columns else np.nan
            duration = df_raw.loc[i, "duration_sec"] if "duration_sec" in df_raw.columns else np.nan
            note = generate_descriptive_note(
                feat, df_raw.loc[i, feat], df_transformed.loc[i, feat], bound, stage, duration
            )
            row = {"_row_index": i, "feature": feat}
            for c in id_cols:
                row[c] = df_raw.loc[i, c]
            row.update({
                "raw_value": df_raw.loc[i, feat],
                "transformed_value": df_transformed.loc[i, feat],
                "bound": bound,
                "descriptive_note": note,
                "decision": "",  # in te vullen: winsorize / remove / keep (leeg = keep)
            })
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["_row_index", "feature"] + id_cols +
                             ["raw_value", "transformed_value", "bound", "descriptive_note", "decision"])
    return pd.DataFrame(rows)


# =============================================================================
# SECTIE 6 — DECISIONS TOEPASSEN: WINSORIZE / REMOVE (STAP 4)
# =============================================================================

def apply_outlier_decisions(df_transformed: pd.DataFrame, outlier_events: pd.DataFrame) -> pd.DataFrame:
    """
    Past de handmatig ingevulde 'decision'-kolom toe op de getransformeerde
    featurematrix:
      - 'winsorize': capt DIE ENE (event, feature)-waarde op het 1e/99e
        percentiel van die feature (op de getransformeerde schaal).
      - 'remove': verwijdert het HELE event (alle features), niet alleen de
        ene kolom die de grens overschreed.
      - leeg of 'keep': geen wijziging.
    """
    df_clean = df_transformed.copy()
    decisions = outlier_events["decision"].astype(str).str.strip().str.lower()

    winsor_rows = outlier_events[decisions == "winsorize"]
    for _, r in winsor_rows.iterrows():
        feat = r["feature"]
        idx = r["_row_index"]
        lower_cap = df_clean[feat].quantile(WINSOR_LOWER_Q)
        upper_cap = df_clean[feat].quantile(WINSOR_UPPER_Q)
        cap = lower_cap if r["bound"] == "laag" else upper_cap
        df_clean.loc[idx, feat] = cap

    remove_idx = outlier_events.loc[decisions == "remove", "_row_index"].unique()
    n_before = len(df_clean)
    df_clean = df_clean.drop(index=[i for i in remove_idx if i in df_clean.index])

    print(f"Decisions toegepast: {len(winsor_rows)} waarden gewinsorized, "
          f"{n_before - len(df_clean)} events volledig verwijderd.")
    return df_clean


# =============================================================================
# SECTIE 7 — STANDAARDISEREN (STAP 5)
# =============================================================================

def scale_features(df: pd.DataFrame, feature_cols: list[str],
                    robust_threshold: float = ROBUST_SKEW_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    StandardScaler (z-score) als default; RobustScaler (mediaan/IQR) voor
    kolommen die na stap 2 (en evt. stap 4) nog steeds |skew| > robust_threshold
    hebben. Geeft de geschaalde df terug plus een overzicht van welke scaler
    per kolom is gebruikt.
    """
    out = df.copy()
    rows = []
    for col in feature_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        finite = vals.dropna()
        s = skew(finite) if len(finite) >= 3 else np.nan

        if pd.notna(s) and abs(s) > robust_threshold:
            median = finite.median()
            iqr = finite.quantile(0.75) - finite.quantile(0.25)
            out[col] = (vals - median) / iqr if iqr != 0 else vals - median
            method = "RobustScaler (median/IQR)"
        else:
            mean = finite.mean()
            std = finite.std()
            out[col] = (vals - mean) / std if std != 0 else vals - mean
            method = "StandardScaler (mean/std)"

        rows.append({"feature": col, "skew_before_scaling": round(s, 3) if pd.notna(s) else np.nan,
                      "scaler_used": method})
    return out, pd.DataFrame(rows)


# =============================================================================
# HOOFDLOOP
# =============================================================================

def run_steps_1_to_3(df: pd.DataFrame, feature_cols: list[str], verbose: bool) -> pd.DataFrame:
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
    log1p_cols, dual_cols, untouched_cols, unclassified = classify_transform_columns(feature_cols)
    if unclassified:
        print(f"\n[WAARSCHUWING] niet-geclassificeerde features (blijven ongemoeid): {unclassified}")

    df_t = apply_log1p_group(df, log1p_cols)
    df_t, dual_choices = apply_dual_transform_group(df_t, dual_cols)
    # untouched_cols: al ongemoeid in df_t, want apply_log1p_group/apply_dual_transform_group
    # raken alleen hun eigen cols aan.

    dist_stats_after = compute_distribution_stats(df_t, feature_cols)
    df_t.to_csv(OUTPUT_DIR / "arousal_feature_matrix_transformed.csv", index=False)
    dist_stats_after.to_csv(OUTPUT_DIR / "distribution_stats_transformed.csv", index=False)
    if len(dual_choices):
        dual_choices.to_csv(OUTPUT_DIR / "dual_transform_choices.csv", index=False)
    plot_distributions(df_t, feature_cols, OUTPUT_DIR / "feature_distributions_transformed.png")

    print(f"\nStap 2: log1p op {len(log1p_cols)} ratio/ridge-freq features; "
          f"log1p-vs-sqrt gekozen voor {dual_cols}; {untouched_cols} ongemoeid.")
    if len(dual_choices):
        print(dual_choices.to_string(index=False))
    print("  - arousal_feature_matrix_transformed.csv\n  - distribution_stats_transformed.csv"
          "\n  - dual_transform_choices.csv\n  - feature_distributions_transformed.png")

    # --- Stap 3 ---
    outlier_summary = compute_outlier_summary(df_t, feature_cols)
    outlier_events = flag_outlier_events(df, df_t, outlier_summary)

    outlier_summary.to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)
    outlier_events.to_csv(OUTPUT_DIR / "outlier_events.csv", index=False)
    print(f"\nStap 3: {len(outlier_events)} outlier-(event, feature)-paren gevonden "
          f"(IQR op getransformeerde schaal).")
    print("  - outlier_summary.csv\n  - outlier_events.csv  <- vul hier de 'decision'-kolom in")
    print("\nVul de 'decision'-kolom in outlier_events.csv in (winsorize/remove/keep) en run:")
    print("  python preprocess_features.py --apply-decisions <pad naar ingevulde outlier_events.csv>")

    if verbose:
        print("\n--- Missing / inf overzicht ---")
        print(missing_summary.to_string(index=False))
        print("\n--- Skewness / kurtosis vóór transformatie ---")
        print(dist_stats.to_string(index=False))
        print("\n--- Skewness / kurtosis NA transformatie ---")
        print(dist_stats_after.to_string(index=False))
        print("\n--- Outlier-overzicht (IQR op getransformeerde schaal) ---")
        print(outlier_summary.to_string(index=False))

    return df_t


def run_steps_4_and_5(df_t: pd.DataFrame, feature_cols: list[str], decisions_path: Path) -> None:
    outlier_events = pd.read_csv(decisions_path)
    if "decision" not in outlier_events.columns:
        raise ValueError(f"{decisions_path} heeft geen 'decision'-kolom -- eerst invullen.")

    # --- Stap 4 ---
    df_clean = apply_outlier_decisions(df_t, outlier_events)
    df_clean.to_csv(OUTPUT_DIR / "arousal_feature_matrix_cleaned.csv", index=False)
    print("  - arousal_feature_matrix_cleaned.csv")

    # --- Stap 5 ---
    df_scaled, scaler_summary = scale_features(df_clean, feature_cols)
    df_scaled.to_csv(OUTPUT_DIR / "arousal_feature_matrix_scaled.csv", index=False)
    scaler_summary.to_csv(OUTPUT_DIR / "scaler_summary.csv", index=False)
    n_robust = (scaler_summary["scaler_used"].str.startswith("Robust")).sum()
    print(f"\nStap 5: {n_robust} van de {len(feature_cols)} features geschaald met RobustScaler "
          f"(rest StandardScaler).")
    print("  - arousal_feature_matrix_scaled.csv  <- klaar voor clustering")
    print("  - scaler_summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help="Pad naar arousal_feature_matrix.csv")
    parser.add_argument("--apply-decisions", type=Path, default=None,
                         help="Pad naar een outlier_events.csv met ingevulde 'decision'-kolom "
                              "-> voert stap 4 (winsorize/remove) en stap 5 (standaardiseren) uit")
    parser.add_argument("--inspect-distributions", action="store_true",
                         help="Print de volledige tabellen ook naar de console")
    args = parser.parse_args()

    df = load_feature_matrix(args.input)
    feature_cols = get_feature_columns(df)
    print(f"\n{len(feature_cols)} features (metadata-kolommen uitgesloten): {feature_cols}")

    df_t = run_steps_1_to_3(df, feature_cols, verbose=args.inspect_distributions)

    if args.apply_decisions:
        run_steps_4_and_5(df_t, feature_cols, args.apply_decisions)


if __name__ == "__main__":
    main()