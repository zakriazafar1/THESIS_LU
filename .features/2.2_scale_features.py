"""
=============================================================================
2.2_scale_features.py

Standardisation (z-score) of the transformed arousal
feature matrix.

Why scale:
  After 2.1 the features still have different units and spreads
  (ln seconds, ln ratios, untransformed motion_rms/oxy_amp_ratio).
  PCA is variance-based and HDBSCAN is distance-based: without scaling, a
  feature with a large spread would dominate purely because of its scale.
  After z-scoring every feature has mean 0 and SD 1, so PCA works on the
  correlation matrix and every feature weighs equally in distances.

Steps:
  1. Load the transformed feature matrix (arousal_feature_matrix_transformed.csv).
  2. Fit and apply StandardScaler per feature (mean -> 0, SD -> 1).
     Mean and SD per feature are written to scaler_params.csv, so scaled
     values can be converted back later: x = z * sd + mean.
  3. Check (mean ~ 0, SD ~ 1) per feature -> scaling_check.csv,
     + distribution figure after scaling.
  4. Write the scaled feature matrix.

Usage:
  python 2.2_scale_features.py
  python 2.2_scale_features.py --input <path> --output-dir <folder>
=============================================================================
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\transformed"
)

OUTPUT_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled"
)

DEFAULT_INPUT = INPUT_DIR / "arousal_feature_matrix_transformed.csv"

METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]

# Frequency bands (rows of the spectral panels) with their label in the figure.
BANDS: list[tuple[str, str]] = [
    ("delta", "Delta\n(0.5-4 Hz)"),
    ("theta", "Theta\n(4-8 Hz)"),
    ("alpha", "Alpha\n(8-12 Hz)"),
    ("sigma", "Sigma\n(12-16 Hz)"),
    ("beta",  "Beta\n(16-30 Hz)"),
]

# Panel labels for the non-spectral features (bottom row).
FEATURE_LABELS: dict[str, str] = {
    "duration_sec":  "Duration",
    "motion_rms":    "Motion RMS",
    "oxy_amp_ratio": "PPG Amplitude Ratio",
}

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
    # italic statistical symbols ($\mathit{z}$) in the same sans-serif font
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.default": "regular",
})

COLOR_LEFT = "#2F7FD8"     
COLOR_RIGHT = "#EE6A2C"   
COLOR_FILL = "#C8C8C8"     
COLOR_EDGE = "#4D4D4D"   
COLOR_REF = "#8C8C8C"    

Z_LABEL = r"$\mathit{z}$ score"

# =============================================================================
# STEP 1 - LOADING
# =============================================================================

def load_transformed_matrix(path: Path) -> pd.DataFrame:
    """
    Reads the transformed feature matrix. Detects the delimiter automatically,
    and re-reads with decimal="," if numeric columns come in as text (Dutch
    Excel scenario). 
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run 2.1_transform_features.py first, or pass the "
            "correct path with --input."
        )
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

    print(f"Transformed feature matrix loaded: {path}")
    print(f"Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the metadata columns -> these are the features to scale."""
    return [c for c in df.columns if c not in METADATA_COLS]


# =============================================================================
# STEP 2: STANDARDSCALER
# =============================================================================

def scale_with_standard_scaler(df: pd.DataFrame, feature_cols: list[str]
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    z-score per feature: (x - mean) / SD. Fitted per column so any NaNs stay
    NaN per feature, and n_used is tracked per feature.
    """
    out = df.copy()
    rows = []
    for col in feature_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        mask = vals.notna()

        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(vals[mask].to_numpy().reshape(-1, 1)).ravel()

        out[col] = np.nan
        out.loc[mask, col] = scaled_vals
        rows.append({
            "feature": col,
            "mean": float(scaler.mean_[0]),
            "sd": float(scaler.scale_[0]),
            "n_used": int(mask.sum()),
            "n_missing_skipped": int((~mask).sum()),
        })
    return out, pd.DataFrame(rows)


def check_scaling(df_scaled: pd.DataFrame, feature_cols: list[str], tol: float = 1e-6) -> pd.DataFrame:
    """
    Check: after z-scoring every feature must have mean ~ 0 and SD ~ 1.
    Returns the mean and SD per feature AFTER scaling (ddof=0, as in sklearn).
    """
    means = df_scaled[feature_cols].mean()
    sds = df_scaled[feature_cols].std(ddof=0)
    check = pd.DataFrame({
        "feature": feature_cols,
        "mean_after": means.round(6).to_numpy(),
        "sd_after": sds.round(6).to_numpy(),
    })
    check["mean_after"] = check["mean_after"].replace(-0.0, 0.0)
    check["ok"] = (means.abs() <= tol).to_numpy() & ((sds - 1).abs() <= tol).to_numpy()

    bad = check.loc[~check["ok"], "feature"].tolist()
    if bad:
        print(f"[WARNING] mean/SD deviates from 0/1 for: {bad}")
    else:
        print(f"Check OK: all {len(feature_cols)} features have mean = 0 and SD = 1.")
    return check


# =============================================================================
# STEP 3: FIGURE (APA 7)
# =============================================================================

def _finite(vals: pd.Series) -> np.ndarray:
    return vals.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


def _format_axes(ax) -> None:
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, integer=True))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))


def _stats_text(ax, lines: list[str]) -> None:
    ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=9, linespacing=1.3)


def _panel_overlay(ax, left: np.ndarray, right: np.ndarray) -> None:
    """Left and right channel as overlaid step histograms with shared bins."""
    pooled = np.concatenate([left, right])
    bins = np.linspace(pooled.min(), pooled.max(), N_BINS + 1)
    top = 0
    for vals, color in ((left, COLOR_LEFT), (right, COLOR_RIGHT)):
        counts, _, _ = ax.hist(vals, bins=bins, histtype="step", color=color, linewidth=1.4)
        top = max(top, counts.max())
        ax.axvline(np.median(vals), color=color, linestyle="--", linewidth=1.2, ymax=0.68)
    ax.axvline(0.0, color=COLOR_REF, linewidth=0.9, ymax=0.68, zorder=0)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, top * 1.5)  # headroom for the text
    _stats_text(ax, [f"Skewness L = {skew(left):.2f}", f"Skewness R = {skew(right):.2f}"])
    _format_axes(ax)


def _panel_single(ax, vals: np.ndarray) -> None:
    """One distribution as a grey filled histogram (L/R average and other features)."""
    bins = np.linspace(vals.min(), vals.max(), N_BINS + 1)
    counts, _, _ = ax.hist(vals, bins=bins, histtype="stepfilled",
                           facecolor=COLOR_FILL, edgecolor=COLOR_EDGE, linewidth=1.2)
    ax.axvline(np.median(vals), color=COLOR_EDGE, linestyle="--", linewidth=1.2, ymax=0.68)
    ax.axvline(0.0, color=COLOR_REF, linewidth=0.9, ymax=0.68, zorder=0)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, counts.max() * 1.5)
    _stats_text(ax, [f"Skewness = {skew(vals):.2f}"])
    _format_axes(ax)


def split_feature_columns(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Spectral features that fit the band x (mean, peak, L/R average) grid, and the rest."""
    spectral = []
    for band, _ in BANDS:
        spectral += [f"L_{band}_ratio", f"R_{band}_ratio",
                     f"L_{band}_peak_ratio", f"R_{band}_peak_ratio", f"mean_{band}_ratio"]
    spectral = [c for c in spectral if c in feature_cols]
    other = [c for c in feature_cols if c not in spectral]
    return spectral, other


def plot_distributions(df: pd.DataFrame, feature_cols: list[str], out_path: Path) -> None:
    """
    APA 7 figure of the z-scored features in the style of 1.4_visualise_features.py.
      Top block: rows = bands; columns = Mean Amplitude (L/R overlaid),
                 Peak Amplitude (L/R overlaid), L/R Average.
      Bottom row: the non-spectral features (duration, motion, PPG).
    """
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
        _panel_overlay(axes[r, 0], g(f"L_{band}_ratio"), g(f"R_{band}_ratio"))
        _panel_overlay(axes[r, 1], g(f"L_{band}_peak_ratio"), g(f"R_{band}_peak_ratio"))
        _panel_single(axes[r, 2], g(f"mean_{band}_ratio"))
        axes[r, 0].text(-0.38, 0.5, band_label, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10, fontweight="bold")

    top_fig.supxlabel(Z_LABEL, fontsize=10)
    top_fig.supylabel("Number of Events", fontsize=10)

    # ---- bottom row(s): other features -------------------------------------
    if other:
        bax = bottom_fig.subplots(n_other_rows, 3, squeeze=False).flatten()
        for i, c in enumerate(other):
            ax = bax[i]
            _panel_single(ax, _finite(df[c]))
            ax.set_title(FEATURE_LABELS.get(c, c), fontweight="bold", pad=8)
            ax.set_xlabel(Z_LABEL)
            if i % 3 == 0:
                ax.set_ylabel("Number of Events")
        for j in range(len(other), len(bax)):
            bax[j].axis("off")

    # ---- legend (inside the image, at the bottom) ---------------------------
    handles = [
        Line2D([], [], color=COLOR_LEFT, linewidth=2, label="Left channel"),
        Line2D([], [], color=COLOR_RIGHT, linewidth=2, label="Right channel"),
        Patch(facecolor=COLOR_FILL, edgecolor=COLOR_EDGE, label="L/R average or single feature"),
        Line2D([], [], color="black", linestyle="--", linewidth=1.2, label="Median"),
        Line2D([], [], color=COLOR_REF, linewidth=0.9, label=r"$\mathit{z}$ = 0 (mean)"),
    ]
    bottom_fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
                      handlelength=1.8, columnspacing=1.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Distribution figure saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to arousal_feature_matrix_transformed.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output folder")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_transformed_matrix(args.input)
    feature_cols = get_feature_columns(df)
    print(f"\n{len(feature_cols)} features (metadata columns excluded): {feature_cols}")

    # --- Step 2: StandardScaler ---
    df_scaled, scaler_params = scale_with_standard_scaler(df, feature_cols)
    scaler_params.to_csv(output_dir / "scaler_params.csv", index=False)
    print(f"\nStandardScaler applied to all {len(feature_cols)} features (mean -> 0, SD -> 1).")
    print(scaler_params.round(4).to_string(index=False))

    # --- Step 3: check + figure ---
    scaling_check = check_scaling(df_scaled, feature_cols)
    scaling_check.to_csv(output_dir / "scaling_check.csv", index=False)
    plot_distributions(df_scaled, feature_cols, output_dir / "feature_distributions_scaled.png")

    # --- Step 4: write ---
    df_scaled.to_csv(output_dir / "arousal_feature_matrix_scaled.csv", index=False)

    print(f"\nSaved in: {output_dir}")
    print("  - scaler_params.csv          (mean/SD before scaling, used for the z-scores)")
    print("  - scaling_check.csv          (mean/SD AFTER scaling, should be 0/1)")
    print("  - feature_distributions_scaled.png")
    print("  - arousal_feature_matrix_scaled.csv  <- input for feature reduction / PCA")


if __name__ == "__main__":
    main()