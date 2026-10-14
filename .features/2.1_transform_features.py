"""
=============================================================================
2.1_transform_features.py

Log transformation of the arousal feature matrix (arousal_feature_matrix_FILTERED.csv).

Steps:
  1. Plot the distribution of every feature (histogram with skewness),
     before and after transformation.
  2. Transform with a FIXED rule:
       - all features -> natural log (ln)
     Rationale: the spectral features are ratios (strictly positive, strongly
     right-skewed, many values < 1). ln makes them symmetric around 0
     (halving and doubling are equally far from 0) and removes most of the
     skew. 
     ln requires strictly positive values: the script stops with an error
     if a log feature contains a value <= 0.
     Skewness before/after is written per feature (transform_choices.csv).

  Scaling is done in 2.2_scale_features.py, which uses the
  arousal_feature_matrix_transformed.csv written here as input.

Usage:
  python 2.1_transform_features.py
  python 2.1_transform_features.py --input <path> --output-dir <folder>
=============================================================================
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import skew

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INPUT = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices\.feature info\arousal_feature_matrix_FILTERED.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\transformed"
)

# Metadata columns, excluded from distribution plots and transformation.
METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk"
]

# Nights excluded because of a night-level baseline/recording artefact.
#EMPTY if not needed.
EXCLUDE_NIGHTS: list[tuple[str, str]] = [ ]

# Features that are NOT transformed 
FORCE_UNTOUCHED_COLS: list[str] = []

# Frequency bands (rows of the spectral panels) with their label in the figure.
BANDS: list[tuple[str, str]] = [
    ("delta", "Delta\n(0.5-4 Hz)"),
    ("theta", "Theta\n(4-8 Hz)"),
    ("alpha", "Alpha\n(8-12 Hz)"),
    ("sigma", "Sigma\n(12-16 Hz)"),
    ("beta",  "Beta\n(16-30 Hz)"),
]

# Labels and units for the non-spectral features (bottom row).
# feature: (panel label, x-axis label on the original scale)
FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "duration_sec":  ("Duration",            "Duration (s)"),
    "motion_rms":    ("Motion RMS",          "RMS Acceleration (g)"),
    "oxy_amp_ratio": ("PPG Amplitude Ratio", "EventRatio"),
}

# Before-transformation figure: x-axis of the spectral panels is cut at this
# quantile (pooled over the panel) so the distribution is readable.
DISPLAY_QUANTILE = 0.99
N_BINS = 40

# =============================================================================
# APA 7 FIGURE STYLE 
# =============================================================================

FIG_DPI = 300

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.default": "regular",
})

COLOR_LEFT = "#2F7FD8"     
COLOR_RIGHT = "#EE6A2C"    
COLOR_FILL = "#C8C8C8"    
COLOR_EDGE = "#4D4D4D"     
COLOR_REF = "#8C8C8C"      

# =============================================================================
# STEP 1: LOADING
# =============================================================================

def load_feature_matrix(path: Path) -> pd.DataFrame:
    """
    Reads the feature matrix. Detects the delimiter automatically, then checks 
    whether numeric columns came in as text (Dutch Excel scenario: semicolon as 
    delimiter AND comma as decimal separator) -- if so, reads again with decimal=",".
    """
    df = pd.read_csv(path, sep=None, engine="python")

    if df.shape[1] == 1:
        raise ValueError(
            f"{path} appears to have only 1 column ({df.columns[0]!r}) -- "
            "the delimiter could not be detected automatically, or the file "
            "is corrupted. Open the file in a text editor to check its contents."
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
            print(f"[WARNING] {n_non_numeric} column(s) were read as text -- "
                  f"decimal comma detected (Dutch Excel format), re-read with decimal=','.")
            df = df_comma_decimal

    print(f"Feature matrix loaded: {path}")
    print(f"Shape: {df.shape}")
    return df


def exclude_nights(df: pd.DataFrame, exclude: list[tuple[str, str]],
                   log_path: Path) -> pd.DataFrame:
    """
    Removes all events of the given (subject_id, night_id) combinations.
    Filters on the combination, because night_id (T0_N1 etc.) recurs for every subject.
    Writes how many events were removed per night.
    """
    if not exclude:
        return df

    keys = df["subject_id"].astype(str).str.strip() + "|" + df["night_id"].astype(str).str.strip()
    excl_keys = {f"{s}|{n}" for s, n in exclude}

    not_found = sorted(excl_keys - set(keys))
    if not_found:
        print(f"[WARNING] these nights are not in the data: {not_found}")

    mask = keys.isin(excl_keys)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (df[mask].groupby(["subject_id", "night_id"]).size()
       .rename("n_events_removed").to_csv(log_path))
    print(f"{int(mask.sum())} events from {len(excl_keys) - len(not_found)} night(s) removed "
          f"({int((~mask).sum())} of {len(df)} events remaining). Log: {log_path}")
    return df.loc[~mask].reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the metadata columns -> clustering features."""
    return [c for c in df.columns if c not in METADATA_COLS]


# =============================================================================
# STEP 2: DISTRIBUTION PLOTS (before/after transformation)
# =============================================================================

def _skew_or_nan(vals: pd.Series) -> float:
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    return skew(vals) if len(vals) >= 3 else np.nan


def _finite(vals: pd.Series) -> np.ndarray:
    return vals.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


def _thousands(ax) -> None:
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, integer=True))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))


def _stats_text(ax, lines: list[str]) -> None:
    ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=9, linespacing=1.3)


def _panel_overlay(ax, left: np.ndarray, right: np.ndarray, log_scale: bool) -> None:
    """Left and right channel as overlaid step histograms with shared bins."""
    pooled = np.concatenate([left, right])
    hi = pooled.max() if log_scale else np.quantile(pooled, DISPLAY_QUANTILE)
    lo = pooled.min() if log_scale else 0.0  # ratios on the original scale start at 0
    bins = np.linspace(lo, hi, N_BINS + 1)
    top = 0
    for vals, color in ((left, COLOR_LEFT), (right, COLOR_RIGHT)):
        counts, _, _ = ax.hist(vals, bins=bins, histtype="step", color=color, linewidth=1.4)
        top = max(top, counts.max())
        ax.axvline(np.median(vals), color=color, linestyle="--", linewidth=1.2, ymax=0.68)
    ax.axvline(0.0 if log_scale else 1.0, color=COLOR_REF, linewidth=0.9, ymax=0.68, zorder=0)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, top * 1.5)  # headroom for the text
    _stats_text(ax, [f"Skewness L = {skew(left):.2f}", f"Skewness R = {skew(right):.2f}"])
    _thousands(ax)


def _panel_single(ax, vals: np.ndarray, log_scale: bool, clip: bool, ref_line: bool) -> None:
    """One distribution as a grey filled histogram (L/R average and other features)."""
    hi = np.quantile(vals, DISPLAY_QUANTILE) if clip else vals.max()
    lo = 0.0 if clip else vals.min()  # clipped = ratio on the original scale -> start at 0
    bins = np.linspace(lo, hi, N_BINS + 1)
    counts, _, _ = ax.hist(vals, bins=bins, histtype="stepfilled",
                           facecolor=COLOR_FILL, edgecolor=COLOR_EDGE, linewidth=1.2)
    ax.axvline(np.median(vals), color=COLOR_EDGE, linestyle="--", linewidth=1.2, ymax=0.68)
    if ref_line:
        ax.axvline(0.0 if log_scale else 1.0, color=COLOR_REF, linewidth=0.9, ymax=0.68, zorder=0)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, counts.max() * 1.5)
    _stats_text(ax, [f"Skewness = {skew(vals):.2f}"])
    _thousands(ax)


def split_feature_columns(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Spectral features that fit the band x (mean, peak, L/R average) grid, and the rest."""
    spectral = []
    for band, _ in BANDS:
        spectral += [f"L_{band}_ratio", f"R_{band}_ratio",
                     f"L_{band}_peak_ratio", f"R_{band}_peak_ratio", f"mean_{band}_ratio"]
    spectral = [c for c in spectral if c in feature_cols]
    other = [c for c in feature_cols if c not in spectral]
    return spectral, other


def other_x_label(col: str, log_transformed: bool) -> str:
    _, xlab = FEATURE_LABELS.get(col, (col, "Value"))
    if not log_transformed:
        return xlab
    return f"ln {xlab}" if "(" in xlab else f"ln({xlab})"


def plot_distributions(df: pd.DataFrame, feature_cols: list[str], out_path: Path,
                       log_cols: list[str] | None = None) -> None:
    """
    APA 7 figure.
      Top block: rows = bands; columns = Mean Amplitude (L/R overlaid),
                 Peak Amplitude (L/R overlaid), L/R Average.
      Bottom row: the non-spectral features (duration, motion, PPG).
    log_cols: features that are on the ln scale in df (None = original scale,
              then the spectral x-axes are cut at DISPLAY_QUANTILE).
    """
    log_cols = set(log_cols or [])
    log_scale = len(log_cols) > 0
    spectral, other = split_feature_columns(feature_cols)
    bands = [(b, lab) for b, lab in BANDS if f"mean_{b}_ratio" in spectral]
    n_other_rows = int(np.ceil(len(other) / 3))

    fig = plt.figure(figsize=(7.5, 1.85 * len(bands) + 2.3 * n_other_rows + 1.3),
                     layout="constrained")
    top_fig, bottom_fig = fig.subfigures(2, 1, height_ratios=[1.85 * len(bands) + 0.5,
                                                           2.3 * n_other_rows + 0.8])

    # ---- top block: spectral features --------------------------------------
    axes = top_fig.subplots(len(bands), 3, squeeze=False)
    for col_i, title in enumerate(["Mean Amplitude", "Peak Amplitude", "L/R Average"]):
        axes[0, col_i].set_title(title, fontweight="bold", pad=8)

    for r, (band, band_label) in enumerate(bands):
        g = lambda c: _finite(df[c])
        _panel_overlay(axes[r, 0], g(f"L_{band}_ratio"), g(f"R_{band}_ratio"), log_scale)
        _panel_overlay(axes[r, 1], g(f"L_{band}_peak_ratio"), g(f"R_{band}_peak_ratio"), log_scale)
        _panel_single(axes[r, 2], g(f"mean_{band}_ratio"), log_scale,
                      clip=not log_scale, ref_line=True)
        axes[r, 0].text(-0.38, 0.5, band_label, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10, fontweight="bold")

    x_lab = ("ln(Event Ratio)" if log_scale
             else "Event Ratio")
    top_fig.supxlabel(x_lab, fontsize=10)
    top_fig.supylabel("Number of Events", fontsize=10)

    # ---- bottom row(s): other features -------------------------------------
    if other:
        bax = bottom_fig.subplots(n_other_rows, 3, squeeze=False).flatten()
        for i, c in enumerate(other):
            ax = bax[i]
            _panel_single(ax, _finite(df[c]), c in log_cols, clip=False, ref_line=False)
            ax.set_title(FEATURE_LABELS.get(c, (c, ""))[0], fontweight="bold", pad=8)
            ax.set_xlabel(other_x_label(c, c in log_cols))
            if i % 3 == 0:
                ax.set_ylabel("Number of Events")
        for j in range(len(other), len(bax)):
            bax[j].axis("off")

    # ---- legend (inside the image, at the bottom) ---------------------------
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=COLOR_LEFT, linewidth=2, label="Left channel"),
        Line2D([], [], color=COLOR_RIGHT, linewidth=2, label="Right channel"),
        Patch(facecolor=COLOR_FILL, edgecolor=COLOR_EDGE, label="L/R average or single feature"),
        Line2D([], [], color="black", linestyle="--", linewidth=1.2, label="Median"),
        Line2D([], [], color=COLOR_REF, linewidth=0.9,
               label="ln(Ratio) = 0" if log_scale else "Ratio = 1"),
    ]
    bottom_fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
                      handlelength=1.8, columnspacing=1.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Distribution figure saved: {out_path}")


# =============================================================================
# STEP 3: TRANSFORMATION 
# =============================================================================

def classify_transform_columns(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Splits the features into log_cols (-> ln) and untouched_cols (FORCE_UNTOUCHED_COLS)."""
    untouched = [c for c in feature_cols if c in FORCE_UNTOUCHED_COLS]
    log_cols = [c for c in feature_cols if c not in FORCE_UNTOUCHED_COLS]
    missing = [c for c in FORCE_UNTOUCHED_COLS if c not in feature_cols]
    if missing:
        print(f"[WARNING] FORCE_UNTOUCHED_COLS contains columns that are not in the matrix: {missing}")
    return log_cols, untouched


def check_strictly_positive(df: pd.DataFrame, log_cols: list[str]) -> None:
    """ln is only defined for x > 0 -- stop with a clear error if that does not hold."""
    problems = []
    for c in log_cols:
        vals = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        n_nonpos = int((vals <= 0).sum())
        if n_nonpos:
            problems.append(f"  {c}: {n_nonpos} value(s) <= 0 (min = {vals.min():.4g})")
    if problems:
        raise ValueError(
            "ln transformation not possible, these features contain values <= 0:\n"
            + "\n".join(problems)
            + "\nAdd them to FORCE_UNTOUCHED_COLS or choose a different transformation."
        )


def apply_log_transform(df: pd.DataFrame, log_cols: list[str], untouched_cols: list[str]
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies ln to log_cols; untouched_cols stay original. Returns df + overview."""
    out = df.copy()
    choices = []
    for c in log_cols + untouched_cols:
        raw = df[c].replace([np.inf, -np.inf], np.nan)
        if c in log_cols:
            out[c] = np.log(raw)
            transform = "ln"
        else:
            out[c] = raw
            transform = "none (FORCE_UNTOUCHED)"
        sb, sa = _skew_or_nan(raw), _skew_or_nan(out[c])
        choices.append({
            "feature": c,
            "transform": transform,
            "skew_before": round(sb, 3) if pd.notna(sb) else np.nan,
            "skew_after": round(sa, 3) if pd.notna(sa) else np.nan,
        })
    return out, pd.DataFrame(choices)


# =============================================================================
# MAIN
# =============================================================================

def run(df: pd.DataFrame, feature_cols: list[str], output_dir: Path,
        nights_excluded: bool) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: distributions before transformation ---
    plot_distributions(df, feature_cols, output_dir / "feature_distributions.png")

    # --- Step 2: transformation ---
    log_cols, untouched_cols = classify_transform_columns(feature_cols)
    check_strictly_positive(df, log_cols)
    df_t, transform_choices = apply_log_transform(df, log_cols, untouched_cols)

    df_t.to_csv(output_dir / "arousal_feature_matrix_transformed.csv", index=False)
    transform_choices.to_csv(output_dir / "transform_choices.csv", index=False)
    plot_distributions(df_t, feature_cols, output_dir / "feature_distributions_transformed.png",
                       log_cols=log_cols)

    print(f"\nFixed rule -- {len(log_cols)} features ln-transformed, "
          f"{len(untouched_cols)} left untransformed ({untouched_cols}).")
    print(transform_choices.to_string(index=False))
    print(f"\nSaved in {output_dir}:"
          "\n  - arousal_feature_matrix_transformed.csv\n  - transform_choices.csv"
          + ("\n  - excluded_nights.csv" if nights_excluded else "")
          + "\n  - feature_distributions.png\n  - feature_distributions_transformed.png")

    return df_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to arousal_feature_matrix_FILTERED.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output folder")
    args = parser.parse_args()

    df = load_feature_matrix(args.input)
    df = exclude_nights(df, EXCLUDE_NIGHTS, args.output_dir / "excluded_nights.csv")
    feature_cols = get_feature_columns(df)
    print(f"\n{len(feature_cols)} features (metadata columns excluded): {feature_cols}")

    run(df, feature_cols, args.output_dir, nights_excluded=bool(EXCLUDE_NIGHTS))


if __name__ == "__main__":
    main()