"""
=============================================================================
1.4_visualise_features.py

Figuren van de verdeling van elke feature in de feature matrix.

Non-feature kolommen (subject_id, group, night_id, stage_rk, event_idx,
start_sec, end_sec) worden geskipt.

Input : INPUT_FILE, or --input <csv>
Output: feature_distributions_eeg.png / .pdf & feature_distributions_other.png / .pdf  
        met --input wordt de input naam toegevoegd.

Usage
  python 1.4_visualise_features.py
  python 1.4_visualise_features.py --input "path/to/arousal_feature_matrix_FILTERED.csv"
=============================================================================
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
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

# =============================================================================
# CONFIGURATION
# =============================================================================

EVENTS_DIR = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\1. feature matrices\.feature info")

INPUT_FILE = EVENTS_DIR / "arousal_feature_matrix_FILTERED.csv"
FALLBACK_FILE = EVENTS_DIR / "arousal_feature_matrix_ORIGIN_clipped.csv"

OUT_DIR = EVENTS_DIR
OUT_NAME = "feature_distributions"          # -> .png and .pdf

NON_FEATURE_COLUMNS = [
    "subject_id", "group", "night_id", "stage_rk", "event_idx",
    "start_sec", "end_sec",
]

# Frequency bands
BANDS = {
    "delta": "Delta\n(0.5-4 Hz)",
    "theta": "Theta\n(4-8 Hz)",
    "alpha": "Alpha\n(8-12 Hz)",
    "sigma": "Sigma\n(12-16 Hz)",
    "beta":  "Beta\n(16-30 Hz)",
}

# EEG kolommen
EEG_COLUMNS = [
    ("Mean Amplitude",                ["L_{b}_ratio", "R_{b}_ratio"],           True),
    ("Peak Amplitude",                ["L_{b}_peak_ratio", "R_{b}_peak_ratio"], True),
    ("L/R Average",                   ["mean_{b}_ratio"],                       False),
]
EEG_XLABEL = "Event Ratio"   # one shared label under panel A

# Andere features 
OTHER_FEATURES = {
    "duration_sec":   ("Arousal Duration",            "Duration (s)",             False),
    "motion_rms":     ("Head Movement",               "RMS Acceleration (g)",     False),
    "oxy_amp_ratio":  ("PPG Pulse Amplitude",         "Event Ratio",  True),
    "sec_prev_event": ("Time Since Previous Arousal", "Interval (s)",             False),
}
Y_LABEL = "Number of Events"

CLIP_PCT = (0.5, 99.5)     # displayed x-range in figuren (data NIET verwijderd)
N_BINS = 40
LOG_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Figuur layout 
FIG_WIDTH_IN = 6.0
ROW_HEIGHT_IN = 1.15       
ROW_GAP_IN = 0.15          
OTHER_N_COLS = 3           
OTHER_ROW_HEIGHT_IN = 2.3 
DPI = 300

# Whitespace 
PAD_IN = 0.08            
PANEL_HSPACE = 0.10       
PANEL_WSPACE = 0.12       
OUTER_MARGIN_IN = 0.15    
FONT_CANDIDATES = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]   # first available used

# Colours
COLOR_L = "#2a78d6"        
COLOR_R = "#eb6834"        
COLOR_SINGLE = "#595959"   
BLACK = "#000000"
REF_GREY = "#8c8c8c"       # reference line (ratio = 1)


def pick_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONT_CANDIDATES if f in available), "DejaVu Sans")


FONT = pick_font()
plt.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "custom",            
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.sf": FONT,
    "mathtext.tt": FONT,
    "mathtext.cal": FONT,
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "normal",
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": BLACK,
    "axes.linewidth": 0.75,
    "axes.labelcolor": BLACK,
    "text.color": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,          
    "svg.fonttype": "none",
})

# =============================================================================
# READING
# =============================================================================

def load_feature_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if col in ("subject_id", "group", "night_id") or pd.api.types.is_numeric_dtype(df[col]):
            continue
        converted = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
        if converted.notna().any():
            df[col] = converted
    return df.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# HELPERS
# =============================================================================

def clean(values: pd.Series, positive: bool = False) -> np.ndarray:
    x = values.dropna().to_numpy(dtype=float)
    return x[x > 0] if positive else x


def clip_range(arrays: list[np.ndarray], log: bool) -> tuple[float, float]:
    pooled = np.concatenate([a for a in arrays if len(a)]) if any(len(a) for a in arrays) else np.array([1.0])
    lo, hi = np.percentile(pooled, CLIP_PCT)
    if lo == hi:
        lo, hi = (lo / 2, hi * 2) if log else (lo - 0.5, hi + 0.5)
    return float(lo), float(hi)


def fmt_number(v: float, _pos=None) -> str:
    return f"{v:g}"


def fmt_count(v: float, _pos=None) -> str:
    return f"{int(v):,}"


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_count))
    ax.set_ylim(bottom=0)


def set_log_axis(ax, lo: float, hi: float):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [t for t in LOG_TICKS if lo <= t <= hi]
    if np.log10(hi / lo) > 2.3:              
        ticks = [t for t in ticks if np.isclose(np.log10(t), round(np.log10(t)))]
    elif len(ticks) > 5:                    
        ticks = [t for t in ticks if t in (0.5, 1, 2, 5, 10, 20, 50)][:5]
    if len(ticks) < 2:                      
        ticks = [lo, hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_number))
    ax.xaxis.set_minor_locator(NullLocator())


def draw_hist(ax, x: np.ndarray, bins: np.ndarray, color: str, lo: float, hi: float,
              fill: bool = True):
    """Outline histogram, filled only for single series."""
    shown = x[(x >= lo) & (x <= hi)]
    if fill:
        ax.hist(shown, bins=bins, histtype="stepfilled", color=color, alpha=0.30, linewidth=0)
    ax.hist(shown, bins=bins, histtype="step", color=color, linewidth=1.0 if fill else 1.1)
    ax.axvline(np.median(x), color=color, linestyle=(0, (3, 2)), linewidth=1.0)


def median_label(ax, values: list[tuple[str, float]]):
    """Medians in the top-right corner"""
    lines = [rf"$\it{{Mdn}}${(' ' + name) if name else ''} = {v:.2f}" for name, v in values]
    ax.text(0.98, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, linespacing=1.3)
    # headroom above the histogram so the label never overlaps the bars,
    # and vertical lines (median, ratio = 1) stop just below the label
    factor = 1.24 + 0.18 * (len(lines) - 1)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * factor)
    for line in ax.lines:
        xdata = line.get_xdata()
        if len(xdata) == 2 and xdata[0] == xdata[1]:          
            line.set_ydata([0, 1.02 / factor])


# =============================================================================
# FIGURE
# =============================================================================

def new_figure(height_in: float):
    fig = plt.figure(figsize=(FIG_WIDTH_IN, height_in), layout="constrained")
    # Whitespace: *_pad = margin around each panel (inches), *space = gap between panels (fraction)
    fig.get_layout_engine().set(h_pad=PAD_IN, w_pad=PAD_IN, hspace=PANEL_HSPACE, wspace=PANEL_WSPACE)
    return fig


def build_eeg_figure(df: pd.DataFrame, log_lines: list[str]):
    """Figure 1: EEG band amplitude (rows = bands, columns = mean / peak / L-R average)."""
    bands = [b for b in BANDS if any(t.format(b=b) in df.columns
                                     for _, tmpl, _ in EEG_COLUMNS for t in tmpl)]
    if not bands:
        return None
    sub_eeg = new_figure((ROW_HEIGHT_IN + ROW_GAP_IN) * len(bands) + 1.0)   

    axes = sub_eeg.subplots(len(bands), len(EEG_COLUMNS), squeeze=False)
    sub_eeg.supylabel(Y_LABEL, fontsize=8)

    for c, (heading, templates, overlay) in enumerate(EEG_COLUMNS):
        col_data = {b: [clean(df[t.format(b=b)], positive=True)
                        for t in templates if t.format(b=b) in df.columns] for b in bands}
        lo, hi = clip_range([a for arrs in col_data.values() for a in arrs], log=True)
        bins = np.logspace(np.log10(lo), np.log10(hi), N_BINS + 1)

        for r, band in enumerate(bands):
            ax = axes[r, c]
            arrays = col_data[band]
            if not arrays:
                ax.set_visible(False)
                continue
            if lo <= 1 <= hi:
                ax.axvline(1.0, color=REF_GREY, linewidth=0.75, zorder=0)
            colors = [COLOR_L, COLOR_R] if overlay else [COLOR_SINGLE]
            for x, color in zip(arrays, colors):
                draw_hist(ax, x, bins, color, lo, hi, fill=not overlay)

            set_log_axis(ax, lo, hi)
            style_axis(ax)
            if overlay and len(arrays) == 2:
                median_label(ax, [("L", np.median(arrays[0])), ("R", np.median(arrays[1]))])
            else:
                median_label(ax, [("", np.median(arrays[0]))])
            if r == 0:
                ax.set_title(heading, fontweight="bold", pad=6)
            if r < len(bands) - 1:
                ax.tick_params(labelbottom=False)
            elif c == len(EEG_COLUMNS) // 2:          
                ax.set_xlabel(EEG_XLABEL)
            if c == 0:
                ax.set_ylabel(BANDS[band], fontweight="bold", rotation=0,
                              ha="right", va="center", labelpad=6, fontsize=8)

            n_out = sum(int(((x < lo) | (x > hi)).sum()) for x in arrays)
            cols = "/".join(t.format(b=band) for t in templates)
            log_lines.append(f"  {cols:<34} {n_out:6d} outside axis")

    handles = [Line2D([], [], color=COLOR_L, linewidth=1.5, label="Left channel"),
               Line2D([], [], color=COLOR_R, linewidth=1.5, label="Right channel"),
               Patch(facecolor=COLOR_SINGLE, alpha=0.35, edgecolor=COLOR_SINGLE, label="L/R average"),
               Line2D([], [], color=BLACK, linestyle=(0, (3, 2)), linewidth=1.0, label="Median"),
               Line2D([], [], color=REF_GREY, linewidth=0.75, label="Ratio = 1")]
    sub_eeg.legend(handles=handles, loc="outside lower center", ncols=5, frameon=False,
                   handlelength=1.8, columnspacing=1.4, handletextpad=0.5)
    return sub_eeg


def build_other_figure(df: pd.DataFrame, other_cols: list[str], log_lines: list[str]):
    """Figure 2: all non-EEG features, OTHER_N_COLS panels per row, linear axes."""
    if not other_cols:
        return None
    n_rows = int(np.ceil(len(other_cols) / OTHER_N_COLS))
    sub_oth = new_figure(OTHER_ROW_HEIGHT_IN * n_rows + 0.5)               

    axes = sub_oth.subplots(n_rows, OTHER_N_COLS, squeeze=False)
    has_ratio_line = False
    for i, col in enumerate(other_cols):
        ax = axes[i // OTHER_N_COLS, i % OTHER_N_COLS]
        heading, xlabel, ratio_line = OTHER_FEATURES.get(col, (col, col, False))
        x = clean(df[col])
        lo, hi = clip_range([x], log=False)
        bins = np.linspace(lo, hi, N_BINS + 1)
        if ratio_line and lo <= 1 <= hi:
            ax.axvline(1.0, color=REF_GREY, linewidth=0.75, zorder=0)
            has_ratio_line = True
        draw_hist(ax, x, bins, COLOR_SINGLE, lo, hi)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_number))
        style_axis(ax)
        median_label(ax, [("", np.median(x))])
        ax.set_title(heading, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel)
        if i % OTHER_N_COLS == 0:
            ax.set_ylabel(Y_LABEL)
        n_out = int(((x < lo) | (x > hi)).sum())
        log_lines.append(f"  {col:<34} {n_out:6d} outside axis")
    for j in range(len(other_cols), n_rows * OTHER_N_COLS):
        axes[j // OTHER_N_COLS, j % OTHER_N_COLS].set_visible(False)

    handles = [Line2D([], [], color=BLACK, linestyle=(0, (3, 2)), linewidth=1.0, label="Median")]
    if has_ratio_line:
        handles.append(Line2D([], [], color=REF_GREY, linewidth=0.75, label="Ratio = 1"))
    sub_oth.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False,
                   handlelength=1.8, columnspacing=1.4, handletextpad=0.5)
    return sub_oth


# =============================================================================
# MAIN
# =============================================================================

def input_tag(path: Path) -> str:
    prefix = "arousal_feature_matrix_"
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to another feature matrix (default: INPUT_FILE)")
    args = parser.parse_args()

    out_stem = OUT_NAME
    if args.input:
        path = args.input
        if not path.exists():
            print(f"Feature matrix not found: {path}")
            return
        out_stem = f"{OUT_NAME}_{input_tag(path)}"
    else:
        path = INPUT_FILE if INPUT_FILE.exists() else FALLBACK_FILE
        if not path.exists():
            print(f"No feature matrix found in {EVENTS_DIR}")
            return
        if path == FALLBACK_FILE:
            print(f"NOTE: {INPUT_FILE.name} not found, using {FALLBACK_FILE.name}")

    df = load_feature_matrix(path)
    feature_cols = [c for c in df.columns
                    if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        print("No numeric features found. Columns as read:")
        print(list(df.columns))
        return

    eeg_cols = {t.format(b=b) for b in BANDS for _, tmpl, _ in EEG_COLUMNS for t in tmpl}
    other_cols = [c for c in OTHER_FEATURES if c in feature_cols]                     # fixed order
    other_cols += [c for c in feature_cols if c not in eeg_cols and c not in other_cols]

    print(f"Loaded: {path.name} - {len(df)} events, {len(feature_cols)} features (font: {FONT})")

    log_lines: list[str] = []
    figures = {
        "eeg": build_eeg_figure(df, log_lines),
        "other": build_other_figure(df, other_cols, log_lines),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        if fig is None:
            continue
        for ext in ("png", "pdf"):
            out = OUT_DIR / f"{out_stem}_{name}.{ext}"
            fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=OUTER_MARGIN_IN)
            print(f"Saved: {out}")
        plt.close(fig)

    print(f"\nEvents outside the displayed x-range ({CLIP_PCT[0]:g}th-{CLIP_PCT[1]:g}th percentile):")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()