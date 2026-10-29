"""
1.5_figure1_transformation.py

Figure 1 voor de thesis: verdeling van alle 28 candidate features voor (A) en
na (B) log-transformatie en standaardisatie.

Layout (A boven B, elk over de volle paginabreedte):
  kolommen = delta, theta, alpha, sigma, beta
  rij 1    = mean amplitude ratio   (L en R over elkaar)
  rij 2    = peak amplitude ratio   (L en R over elkaar)
  rij 3    = L/R average            (grijs)
  rij 4    = duration, head movement, PPG ratio (grijs)

Gemaakt op A4-tekstbreedte (6.3 in) zodat Word de figuur niet hoeft te verkleinen.

Input : INPUT_FILE, or --input <csv>
Output: figure1_transformation.png / .pdf

Usage
  python 1.5_figure1_transformation.py
  python 1.5_figure1_transformation.py --input "path/to/arousal_feature_matrix_FILTERED.csv"
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import skew


# CONFIGURATION
EVENTS_DIR = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices\.feature info")
INPUT_FILE = EVENTS_DIR / "arousal_feature_matrix_FILTERED.csv"
OUT_DIR = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing")
OUT_NAME = "figure1_transformation"

BANDS = {
    "delta": "Delta (0.5-4 Hz)",
    "theta": "Theta (4-8 Hz)",
    "alpha": "Alpha (8-12 Hz)",
    "sigma": "Sigma (12-16 Hz)",
    "beta":  "Beta (16-30 Hz)",
}

# (rijlabel, kolom-templates, L/R overlay?)
EEG_ROWS = [
    ("Mean\namplitude", ["L_{b}_ratio", "R_{b}_ratio"],           True),
    ("Peak\namplitude", ["L_{b}_peak_ratio", "R_{b}_peak_ratio"], True),
    ("L/R\naverage",    ["mean_{b}_ratio"],                       False),
]

# (kolom, titel, x-label origineel, ratio-feature?)
OTHER_FEATURES = [
    ("duration_sec",  "Duration",      "Duration (s)", False),
    ("motion_rms",    "Head movement", "RMS (g)",      False),
    ("oxy_amp_ratio", "PPG ratio",     "Event ratio",  True),
]

CLIP_PCT = 99.5          # x-as toont 0.5e-99.5e percentiel; skewness op alle data
N_BINS = 40
Y_LABEL = "Number of events"

# Layout (inches); 6.3 in = A4 met 2.5 cm marges
FIG_WIDTH_IN = 6.3
ROW_HEIGHT_IN = 1.0
DPI = 300

COLOR_L = "#2a78d6"
COLOR_R = "#eb6834"
COLOR_SINGLE = "#595959"
REF_GREY = "#8c8c8c"
BLACK = "#000000"
FONT_CANDIDATES = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def pick_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONT_CANDIDATES if f in available), "DejaVu Sans")


FONT = pick_font()
plt.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


# DATA
def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            conv = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False),
                                 errors="coerce")
            if conv.notna().any():
                df[col] = conv
    return df.replace([np.inf, -np.inf], np.nan)


def values(df: pd.DataFrame, col: str, transformed: bool) -> np.ndarray:
    x = df[col].dropna().to_numpy(dtype=float)
    x = x[x > 0]
    if transformed:                       # ln, dan z (zoals StandardScaler, ddof=0)
        lx = np.log(x)
        x = (lx - lx.mean()) / lx.std()
    return x


def skew_str(x: np.ndarray) -> str:
    return f"{round(float(skew(x)), 1) + 0.0:.1f}"     # + 0.0 voorkomt "-0.0"


# PANELS
def fmt_count(v, _pos=None):
    return f"{int(v):,}"


def fmt_num(v, _pos=None):
    return f"{v:g}"


def draw_panel(ax, arrays, colors, ref):
    """arrays: 1 (grijs, gevuld) of 2 (L/R, alleen omlijning) series."""
    pooled = np.concatenate(arrays)
    lo, hi = np.percentile(pooled, (100 - CLIP_PCT, CLIP_PCT))
    bins = np.linspace(lo, hi, N_BINS + 1)
    single = len(arrays) == 1

    for x, color in zip(arrays, colors):
        shown = x[(x >= lo) & (x <= hi)]
        if single:
            ax.hist(shown, bins=bins, histtype="stepfilled", color=color, alpha=0.30, linewidth=0)
        ax.hist(shown, bins=bins, histtype="step", color=color, linewidth=0.9)

    # skewness rechtsboven, met ruimte erboven; verticale lijnen stoppen eronder
    if single:
        label = f"Skew = {skew_str(arrays[0])}"
        headroom, ymax_line = 1.30, 0.80
    else:
        label = f"Skew L = {skew_str(arrays[0])}\nSkew R = {skew_str(arrays[1])}"
        headroom, ymax_line = 1.55, 0.66
    ax.set_ylim(0, ax.get_ylim()[1] * headroom)
    ax.text(0.98, 0.99, label, transform=ax.transAxes, ha="right", va="top",
            fontsize=7, linespacing=1.15)

    if ref is not None and lo <= ref <= hi:
        ax.axvline(ref, ymax=ymax_line, color=REF_GREY, linewidth=0.75, linestyle=":", zorder=0)
    ax.axvline(np.median(pooled), ymax=ymax_line, color=BLACK,
               linestyle=(0, (3, 2)), linewidth=0.9)

    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_num))
    ax.yaxis.set_major_locator(MaxNLocator(2, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_count))


def build_block(sub, df: pd.DataFrame, transformed: bool, label: str):
    n_cols = len(BANDS)
    axes = sub.subplots(len(EEG_ROWS) + 1, n_cols, squeeze=False)
    sub.suptitle(label, x=0.0, ha="left", fontweight="bold", fontsize=11)
    sub.supylabel(Y_LABEL, fontsize=8)
    eeg_xlabel = r"$\it{z}$" if transformed else "Event ratio"
    eeg_ref = 0.0 if transformed else 1.0

    # EEG: rijen = mean / peak / average, kolommen = banden
    for r, (row_label, templates, overlay) in enumerate(EEG_ROWS):
        for c, (band, band_label) in enumerate(BANDS.items()):
            ax = axes[r, c]
            arrays = [values(df, t.format(b=band), transformed) for t in templates]
            colors = [COLOR_L, COLOR_R] if overlay else [COLOR_SINGLE]
            draw_panel(ax, arrays, colors, eeg_ref)
            if r == 0:
                ax.set_title(band_label, fontweight="bold", pad=4)
            if c == 0:
                ax.set_ylabel(row_label, fontweight="bold", labelpad=4)
            if r == len(EEG_ROWS) - 1 and c == n_cols // 2:
                ax.set_xlabel(eeg_xlabel)

    # Overige features in de onderste rij
    r = len(EEG_ROWS)
    for c in range(n_cols):
        ax = axes[r, c]
        if c >= len(OTHER_FEATURES):
            ax.set_visible(False)
            continue
        col, title, xlabel_raw, is_ratio = OTHER_FEATURES[c]
        ref = 0.0 if transformed else (1.0 if is_ratio else None)
        draw_panel(ax, [values(df, col, transformed)], [COLOR_SINGLE], ref)
        ax.set_title(title, fontweight="bold", pad=4)
        ax.set_xlabel(r"$\it{z}$" if transformed else xlabel_raw)


def build_figure(df: pd.DataFrame):
    n_rows = len(EEG_ROWS) + 1
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 2 * n_rows * ROW_HEIGHT_IN + 1.0),
                     layout="constrained")
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.04, hspace=0.04, wspace=0.03)
    sub_a, sub_b = fig.subfigures(2, 1, hspace=0.03)
    build_block(sub_a, df, transformed=False, label="A")
    build_block(sub_b, df, transformed=True, label="B")

    handles = [Line2D([], [], color=COLOR_L, linewidth=1.5, label="Left channel"),
               Line2D([], [], color=COLOR_R, linewidth=1.5, label="Right channel"),
               Patch(facecolor=COLOR_SINGLE, alpha=0.35, edgecolor=COLOR_SINGLE,
                     label="L/R average or single feature"),
               Line2D([], [], color=BLACK, linestyle=(0, (3, 2)), linewidth=1.0, label="Median"),
               Line2D([], [], color=REF_GREY, linestyle=":", linewidth=1.0,
                      label=r"Ratio = 1 (A) / $\it{z}$ = 0 (B)")]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False,
               handlelength=1.8, columnspacing=1.2, handletextpad=0.5)
    return fig


# MAIN
def all_columns() -> list[str]:
    cols = [t.format(b=b) for _, tmpl, _ in EEG_ROWS for b in BANDS for t in tmpl]
    return cols + [c for c, *_ in OTHER_FEATURES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Feature matrix not found: {args.input}")
        return
    df = load(args.input)
    missing = [c for c in all_columns() if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return

    fig = build_figure(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = args.out_dir / f"{OUT_NAME}.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved: {out}")
    plt.close(fig)

    # Skewness per feature (op alle data) - voor de tekst in de Methods
    raw = {c: float(skew(values(df, c, False))) for c in all_columns()}
    trans = {c: float(skew(values(df, c, True))) for c in all_columns()}
    print(f"\n{len(raw)} features. x-axes show the {100 - CLIP_PCT:g}th-{CLIP_PCT:g}th "
          f"percentile (display only); skewness uses all events.")
    print(f"  Original:    M = {np.mean(list(raw.values())):.1f}, "
          f"range = {min(raw.values()):.1f} to {max(raw.values()):.1f}")
    print(f"  Transformed: M = {np.mean(list(trans.values())):.1f}, "
          f"range = {min(trans.values()):.1f} to {max(trans.values()):.1f}")
    for c in all_columns():
        print(f"    {c:<22} {raw[c]:7.2f}  ->  {trans[c]:5.2f}")


if __name__ == "__main__":
    main()