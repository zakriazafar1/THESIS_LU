"""
=============================================================================
1.4_visualise_features.py

Visualiseert de verdeling van elke feature in de (opgeschoonde) featurematrix
als histogram. Om het leesbaar te houden worden de features over TWEE
figuren (grids) verdeeld:

  Figuur 1 - EEG band-ratio's per kanaal (20 features)
             rijen = banden (delta ... beta)
             kolommen = L_ratio, L_peak_ratio, R_ratio, R_peak_ratio

  Figuur 2 - Overige features (mean_*_ratio, timing, motion, oxy)

Leesbaarheid:
  - Ratio-features zijn sterk scheef verdeeld (lange staart naar rechts).
    Daarom wordt de x-as per plot afgekapt op het 0.5e-99.5e percentiel;
    het aantal events BUITEN dat bereik staat in de titel ("outside axis").
    Er worden geen data verwijderd, alleen de weergave wordt afgekapt.
  - Rode stippellijn  = mediaan
  - Grijze lijn (x=1) = baseline van de nacht (alleen bij ratio-features:
    waarde 1 betekent "gelijk aan de mediane amplitude van die nacht")

Niet-feature kolommen (subject_id, group, night_id, stage_rk, event_idx,
start_sec, end_sec) worden overgeslagen.

Input : arousal_feature_matrix_CLEAN.csv  (valt terug op _ORIGIN.csv)
Output:
  feature_distributions_1_eeg_bands.png
  feature_distributions_2_other.png

Gebruik:
  python 1.4_visualise_features.py
  python 1.4_visualise_features.py --input "pad\\naar\\arousal_feature_matrix_FILTERED.csv"

Met --input wordt de naam van het input-bestand aan de output toegevoegd,
zodat de standaard-plots niet overschreven worden:
  ..._FILTERED.csv  ->  feature_distributions_eeg_FILTERED.png, ..._other_FILTERED.png
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # geen venster nodig, alleen PNG's wegschrijven
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATIE
# =============================================================================

EVENTS_DIR = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices\.feature info")

INPUT_FILE = EVENTS_DIR / "arousal_feature_matrix_CLEAN.csv"
FALLBACK_FILE = EVENTS_DIR / "arousal_feature_matrix_ORIGIN.csv"

OUT_EEG = EVENTS_DIR / "feature_distributions_eeg.png"
OUT_OTHER = EVENTS_DIR / "feature_distributions_other.png"

NON_FEATURE_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec",
]

BANDS = ["delta", "theta", "alpha", "sigma", "beta"]
EEG_METRICS = ["L_ratio", "L_peak_ratio", "R_ratio", "R_peak_ratio"]   

CLIP_PCT = (1, 99)   # weergavebereik x-as (percentielen)
N_BINS = 60
DPI = 150

BAR_COLOR = "#3B6EA8"
MEDIAN_COLOR = "#C0392B"
BASELINE_COLOR = "#7F7F7F"

# Nette titels 
PRETTY = {
    "duration_sec": "duration (s)",
    "sec_prev_event": "time prev. event (s)",
    "motion_rms": "motion RMS",
    "oxy_amp_ratio": "oxy amplitude ratio",
}

# =============================================================================
# INLEZEN 
# =============================================================================

def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Detecteert komma/puntkomma-scheiding en zet decimale komma's om."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if col in ("subject_id", "group", "night_id") or pd.api.types.is_numeric_dtype(df[col]):
            continue
        converted = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if converted.notna().any():
            df[col] = converted
    return df.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# PLOTTEN
# =============================================================================

def plot_hist(ax, values: pd.Series, title: str, is_ratio: bool):
    """Eén histogram, met afgekapt weergavebereik, mediaan en (bij ratio's) x=1."""
    x = values.dropna().to_numpy()
    if len(x) == 0:
        ax.text(0.5, 0.5, "geen data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return

    lo, hi = np.percentile(x, CLIP_PCT)
    if lo == hi:                                   # constante feature
        lo, hi = lo - 0.5, hi + 0.5
    n_out = int(((x < lo) | (x > hi)).sum())

    ax.hist(x[(x >= lo) & (x <= hi)], bins=N_BINS, range=(lo, hi),
            color=BAR_COLOR, edgecolor="white", linewidth=0.4)

    med = np.median(x)
    ax.axvline(med, color=MEDIAN_COLOR, linestyle="--", linewidth=1.2)
    if is_ratio and lo <= 1 <= hi:
        ax.axvline(1.0, color=BASELINE_COLOR, linewidth=1.0)

    ax.set_xlim(lo, hi)
    sub = f"median = {med:.2f}   n = {len(x)}"
    if n_out:
        sub += f"   ({n_out} outside axis)"
    ax.set_title(f"{title}\n{sub}", fontsize=10)

    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def add_legend(fig, show_baseline: bool = True):
    handles = [plt.Line2D([], [], color=MEDIAN_COLOR, linestyle="--", label="median")]
    if show_baseline:
        handles.append(plt.Line2D([], [], color=BASELINE_COLOR, label="ratio = 1 (night baseline)"))
    fig.legend(handles=handles, loc="upper right", fontsize=9, frameon=False)


def figure_eeg_bands(df: pd.DataFrame, n_events: int, source: str):
    """Figuur 1: rijen = banden, kolommen = L/R x mean/peak."""
    fig, axes = plt.subplots(len(BANDS), len(EEG_METRICS),
                             figsize=(16, 3.2 * len(BANDS)), squeeze=False)
    for r, band in enumerate(BANDS):
        for c, metric in enumerate(EEG_METRICS):
            ch, kind = metric.split("_", 1)            # "L", "ratio" / "peak_ratio"
            col = f"{ch}_{band}_{kind}"
            ax = axes[r, c]
            if col in df.columns:
                plot_hist(ax, df[col], col, is_ratio=True)
            else:
                ax.set_visible(False)
            if c == 0:
                ax.set_ylabel("events", fontsize=9)

    fig.suptitle(f"EEG band ratios per channel  -  {n_events} events ({source})",
                 fontsize=14, x=0.01, ha="left")
    add_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def figure_other(df: pd.DataFrame, cols: list[str], n_events: int, source: str):
    """Figuur 2: alle overige features, 3 kolommen breed."""
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.4 * n_rows), squeeze=False)

    for i, col in enumerate(cols):
        ax = axes[i // n_cols, i % n_cols]
        plot_hist(ax, df[col], PRETTY.get(col, col), is_ratio=col.endswith("ratio"))
        if i % n_cols == 0:
            ax.set_ylabel("events", fontsize=9)
    for j in range(len(cols), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.suptitle(f"Other features  -  {n_events} events ({source})",
                 fontsize=14, x=0.01, ha="left")
    add_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# =============================================================================
# HOOFDLOOP
# =============================================================================

def input_tag(path: Path) -> str:
    """'arousal_feature_matrix_FILTERED' -> 'FILTERED'; andere namen blijven heel."""
    prefix = "arousal_feature_matrix_"
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None,
                        help="Pad naar een andere featurematrix (standaard: INPUT_FILE)")
    args = parser.parse_args()

    out_eeg, out_other = OUT_EEG, OUT_OTHER
    if args.input:
        path = args.input
        if not path.exists():
            print(f"Featurematrix niet gevonden: {path}")
            return
        tag = input_tag(path)
        out_eeg = OUT_EEG.with_name(f"{OUT_EEG.stem}_{tag}{OUT_EEG.suffix}")
        out_other = OUT_OTHER.with_name(f"{OUT_OTHER.stem}_{tag}{OUT_OTHER.suffix}")
    else:
        path = INPUT_FILE if INPUT_FILE.exists() else FALLBACK_FILE
        if not path.exists():
            print(f"Geen featurematrix gevonden in {EVENTS_DIR}")
            return
        if path == FALLBACK_FILE:
            print(f"LET OP: {INPUT_FILE.name} niet gevonden, gebruik {FALLBACK_FILE.name}")

    df = load_feature_matrix(path)
    feature_cols = [c for c in df.columns
                    if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        print("Geen numerieke features gevonden. Kolommen zoals ingelezen:")
        print(list(df.columns))
        return

    eeg_cols = [f"{m.split('_', 1)[0]}_{b}_{m.split('_', 1)[1]}" for b in BANDS for m in EEG_METRICS]
    other_cols = [c for c in feature_cols if c not in eeg_cols]

    print(f"Ingeladen: {path.name} - {len(df)} events, {len(feature_cols)} features")

    out_eeg.parent.mkdir(parents=True, exist_ok=True)
    out_other.parent.mkdir(parents=True, exist_ok=True)

    fig1 = figure_eeg_bands(df, len(df), path.name)
    fig1.savefig(out_eeg, dpi=DPI)
    plt.close(fig1)
    print(f"Opgeslagen: {out_eeg}")

    fig2 = figure_other(df, other_cols, len(df), path.name)
    fig2.savefig(out_other, dpi=DPI)
    plt.close(fig2)
    print(f"Opgeslagen: {out_other}")


if __name__ == "__main__":
    main()