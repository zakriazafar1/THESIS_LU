"""
=============================================================================
3.2_check_corr.py

Visual check of the feature pairs where Pearson and Spearman diverge strongly
(output of 3.1_correlation.py: pearson_vs_spearman_divergent.csv).

Spearman is the main analysis (3.1); this script is a diagnostic check.
Question: is Pearson's r inflated by isolated islands of extreme events in
the LOWER LEFT (both features very low, z < TAIL_Z)? On a log scale, a z of
-4 to -8 means a ratio close to 0: event power far below the baseline ->
suspicious of a contaminated baseline or signal dropout.

Per pair:
  - Scatterplot (hexbin density, log scale) of the scaled values.
  - "Tail events" highlighted: events where BOTH features < TAIL_Z (z-score).
  - In the panel: Pearson r, Spearman rho, and both WITHOUT the tail events.
    If r drops markedly without the tail -> the tail drives the correlation.

In addition, a table of all tail events (across all plotted pairs) with
metadata (subject, night, start_sec, duration, motion_rms), so they can be
looked up in the raw EEG. And a count per subject/night: are the tail events
spread out, or concentrated in a few nights (-> artefact or subject-specific)?

Steps:
  1. Load the scaled feature matrix + divergent pairs.
  2. Select the top-N pairs (by |rho - r|), or a single pair via --pair.
  3. Per pair: statistics + scatterplot (grid + individual figures).
  4. Save tail events + counts per subject/night.

Figures follow APA 7 style: no embedded titles (number + title go above the
figure in the manuscript), sans-serif font, sentence-case axis labels,
readable feature names, statistical symbols in italics, correlations without
leading zero, panel letters in multi-panel figures, no chart junk.

Usage:
  python 3.2_check_corr.py
  python 3.2_check_corr.py --top 9 --tail-z -4
  python 3.2_check_corr.py --pair mean_theta_ratio mean_beta_ratio
=============================================================================
"""

import argparse
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.stats import pearsonr, spearmanr

# =============================================================================
# CONFIGURATION
# =============================================================================

SCALED_PATH = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled\arousal_feature_matrix_scaled.csv"
)

CORR_VIF_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\3. feature selection"
)
DIVERGENT_PATH = CORR_VIF_DIR / "pearson_vs_spearman_divergent.csv"

OUTPUT_DIR = CORR_VIF_DIR / "scatter_divergent"

METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]
# extra columns included in the tail table (if present)
CONTEXT_COLS = ["duration_sec", "motion_rms", "oxy_amp_ratio"]

TOP_N_DEFAULT = 6   # number of pairs to plot
TAIL_Z = -3.0       # tail = both features < TAIL_Z (in SDs, since z-scores)
N_COLS_GRID = 3

FIG_DPI = 300

# =============================================================================
# APA 7 FIGURE STYLE
# =============================================================================

TEXT_COLOR = "#000000"
TAIL_COLOR = "#C0392B"
DENSITY_CMAP = "Blues"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.edgecolor": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.default": "it",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

BAND_NAMES = {"delta": "Delta", "theta": "Theta", "alpha": "Alpha",
              "sigma": "Sigma", "beta": "Beta"}
SPECIAL_LABELS = {
    "duration_sec": "Duration",
    "oxy_amp_ratio": "PPG amplitude ratio",
    "motion_rms": "Head movement RMS",
}


def pretty_label(name: str) -> str:
    """
    Readable feature name for figures, e.g.
      L_delta_peak_ratio -> 'Delta peak ratio (L)'
      mean_alpha_ratio   -> 'Alpha ratio (mean)'
    """
    if name in SPECIAL_LABELS:
        return SPECIAL_LABELS[name]
    m = re.fullmatch(r"(L|R|mean)_(delta|theta|alpha|sigma|beta)_(peak_)?ratio", name)
    if m:
        side, band, peak = m.groups()
        kind = "peak ratio" if peak else "ratio"
        return f"{BAND_NAMES[band]} {kind} ({side})"
    return name.replace("_", " ")


def apa_corr(v: float) -> str:
    """APA format for correlations: two decimals, no leading zero (.85, −.32)."""
    if pd.isna(v):
        return "NA"
    s = f"{v:.2f}"
    s = s.replace("0.", ".", 1) if abs(v) < 1 else s
    return s.replace("-", "\u2212")


def z_label(name: str) -> str:
    """Axis label with italic z, e.g. 'Theta ratio (R; z)' or 'Duration (z)'."""
    lab = pretty_label(name)
    if lab.endswith(")"):
        return lab[:-1] + "; $z$)"
    return f"{lab} ($z$)"


def minus(v: float) -> str:
    """Number with a typographic minus sign (APA)."""
    return f"{v:g}".replace("-", "\u2212")


def save_figure(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# SECTION 1 — LOADING
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Reads a CSV with automatic delimiter detection (',' or ';') and falls back
    to decimal=',' if numeric columns are read as text (Dutch Excel).
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    df = pd.read_csv(path, sep=None, engine="python")
    text_cols = {"subject_id", "group", "night_id", "feature_1", "feature_2"}
    check = [c for c in df.columns if c not in text_cols]
    n_bad = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in check)
    if n_bad > 0:
        df2 = pd.read_csv(path, sep=None, engine="python", decimal=",")
        n_bad2 = sum(not pd.api.types.is_numeric_dtype(df2[c]) for c in check if c in df2.columns)
        if n_bad2 < n_bad:
            print(f"[NOTE] decimal comma detected in {path.name}, re-read with decimal=','.")
            df = df2
    return df


def choose_pairs(args, df: pd.DataFrame) -> list[tuple[str, str]]:
    """A single pair via --pair, otherwise the top N from the divergent CSV."""
    if args.pair:
        pairs = [tuple(args.pair)]
    else:
        div = read_csv_robust(args.divergent)
        div = div.reindex(div["diff_rho_minus_r"].abs().sort_values(ascending=False).index)
        pairs = list(zip(div["feature_1"], div["feature_2"]))[: args.top]
        print(f"Top {len(pairs)} pairs from {args.divergent.name} (largest |rho - r|).")

    missing = {f for p in pairs for f in p if f not in df.columns}
    if missing:
        raise KeyError(f"These features are not in the feature matrix: {sorted(missing)}")
    return pairs


# =============================================================================
# SECTION 2 — STATISTICS PER PAIR
# =============================================================================

def pair_stats(df: pd.DataFrame, f1: str, f2: str, tail_z: float) -> dict:
    """Pearson, Spearman, and both without tail events, for one pair."""
    sub = df[[f1, f2]].replace([np.inf, -np.inf], np.nan).dropna()
    x, y = sub[f1].to_numpy(), sub[f2].to_numpy()
    tail = (x < tail_z) & (y < tail_z)

    r = pearsonr(x, y)[0]
    rho = spearmanr(x, y)[0]
    r_no_tail = pearsonr(x[~tail], y[~tail])[0] if (~tail).sum() > 2 else np.nan
    rho_no_tail = spearmanr(x[~tail], y[~tail])[0] if (~tail).sum() > 2 else np.nan

    return {
        "feature_1": f1, "feature_2": f2,
        "n": len(sub), "n_tail": int(tail.sum()),
        "pct_tail": round(100 * tail.mean(), 2),
        "pearson_r": round(r, 4), "spearman_rho": round(rho, 4),
        "pearson_r_no_tail": round(r_no_tail, 4),
        "spearman_rho_no_tail": round(rho_no_tail, 4),
        "tail_index": sub.index[tail],   # for the tail table, not written to CSV
    }


# =============================================================================
# SECTION 3 — PLOTS
# =============================================================================

def draw_pair(ax, df: pd.DataFrame, st: dict, tail_z: float, panel: str | None = None) -> None:
    """
    Hexbin density (log scale) + tail events in red + tail cut-off as dotted
    lines. Correlations with and without the tail as text in the panel.
    """
    f1, f2 = st["feature_1"], st["feature_2"]
    sub = df[[f1, f2]].replace([np.inf, -np.inf], np.nan).dropna()

    hb = ax.hexbin(sub[f1], sub[f2], gridsize=60, cmap=DENSITY_CMAP,
                   norm=LogNorm(), mincnt=1, linewidths=0)
    tail = df.loc[st["tail_index"], [f1, f2]]
    tz = minus(tail_z)
    ax.scatter(tail[f1], tail[f2], s=8, c=TAIL_COLOR, alpha=0.7, linewidths=0,
               label=f"Both < {tz} $SD$ ($n$ = {st['n_tail']:,}, {st['pct_tail']:.1f}%)")
    ax.axvline(tail_z, color=TAIL_COLOR, ls=":", lw=0.9)
    ax.axhline(tail_z, color=TAIL_COLOR, ls=":", lw=0.9)

    ax.set_xlabel(z_label(f1))
    ax.set_ylabel(z_label(f2))

    stats_txt = (
        f"All events: $r$ = {apa_corr(st['pearson_r'])}, "
        f"$\\rho$ = {apa_corr(st['spearman_rho'])}\n"
        f"Without tail: $r$ = {apa_corr(st['pearson_r_no_tail'])}, "
        f"$\\rho$ = {apa_corr(st['spearman_rho_no_tail'])}"
    )
    ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3)

    if panel:
        ax.text(-0.16, 1.04, panel, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="bottom", color=TEXT_COLOR)

    cb = plt.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Number of events (log scale)", fontsize=8)
    cb.ax.yaxis.set_major_locator(LogLocator(base=10))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)


def plot_grid(df: pd.DataFrame, stats: list[dict], tail_z: float, out_path: Path) -> None:
    n = len(stats)
    n_cols = min(N_COLS_GRID, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    letters = string.ascii_uppercase
    for i, (ax, st) in enumerate(zip(axes, stats)):
        draw_pair(ax, df, st, tail_z, panel=letters[i] if n > 1 else None)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, out_path)
    print(f"Grid saved: {out_path} (+ .pdf)")


def plot_single(df: pd.DataFrame, st: dict, tail_z: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.2))
    draw_pair(ax, df, st, tail_z)
    fig.tight_layout()
    save_figure(fig, out_path)


# =============================================================================
# SECTION 4 — TAIL EVENTS
# =============================================================================

def tail_events_table(df: pd.DataFrame, stats: list[dict]) -> pd.DataFrame:
    """
    All events that fall in the tail in at least one plotted pair, with
    metadata, context columns and the number of pairs in which they are in
    the tail.
    """
    counts = pd.Series(0, index=df.index)
    for st in stats:
        counts.loc[st["tail_index"]] += 1
    idx = counts[counts > 0].index

    meta = [c for c in METADATA_COLS if c in df.columns]
    ctx = [c for c in CONTEXT_COLS if c in df.columns]
    feats = sorted({f for st in stats for f in (st["feature_1"], st["feature_2"])})
    ctx = [c for c in ctx if c not in feats]

    out = df.loc[idx, meta + ctx + feats].copy()
    out.insert(0, "n_pairs_in_tail", counts.loc[idx])
    return out.sort_values("n_pairs_in_tail", ascending=False)


def tail_by_group(tail: pd.DataFrame, df: pd.DataFrame, key: list[str]) -> pd.DataFrame | None:
    """Number of tail events per subject/night, relative to the total number of events there."""
    if any(k not in df.columns for k in key) or tail.empty:
        return None
    total = df.groupby(key).size().rename("n_events")
    in_tail = tail.groupby(key).size().rename("n_tail_events")
    out = pd.concat([total, in_tail], axis=1).fillna(0).astype(int)
    out["pct_tail"] = (100 * out["n_tail_events"] / out["n_events"]).round(2)
    out["share_of_all_tail"] = (100 * out["n_tail_events"] / len(tail)).round(2)
    return out.sort_values("n_tail_events", ascending=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SCALED_PATH,
                        help="Path to arousal_feature_matrix_scaled.csv")
    parser.add_argument("--divergent", type=Path, default=DIVERGENT_PATH,
                        help="Path to pearson_vs_spearman_divergent.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT,
                        help="Number of pairs from the divergent CSV")
    parser.add_argument("--pair", nargs=2, metavar=("FEATURE_1", "FEATURE_2"),
                        help="Plot only this single pair")
    parser.add_argument("--tail-z", type=float, default=TAIL_Z,
                        help="Tail cut-off in SDs (default -3.0)")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Steps 1+2 ---
    df = read_csv_robust(args.input)
    print(f"Feature matrix loaded: {args.input}  shape={df.shape}")
    pairs = choose_pairs(args, df)
    if not pairs:
        print("No divergent pairs (Spearman and Pearson never differ by more than the "
              "threshold) -- nothing to plot. The correlations are therefore not driven by a tail.")
        return

    # --- Step 3: statistics + plots ---
    stats = [pair_stats(df, f1, f2, args.tail_z) for f1, f2 in pairs]
    stats_df = pd.DataFrame([{k: v for k, v in s.items() if k != "tail_index"} for s in stats])
    stats_df.to_csv(out_dir / "pair_stats.csv", index=False)
    print(f"\nPer pair (tail = both features < {args.tail_z} SD):")
    print(stats_df.to_string(index=False))

    plot_grid(df, stats, args.tail_z, out_dir / "scatter_grid.png")
    for st in stats:
        plot_single(df, st, args.tail_z,
                    out_dir / f"scatter_{st['feature_1']}__{st['feature_2']}.png")

    # --- Step 4: tail events ---
    tail = tail_events_table(df, stats)
    tail.to_csv(out_dir / "tail_events.csv", index=False)
    print(f"\n{len(tail)} unique tail event(s) across all plotted pairs "
          f"({100 * len(tail) / len(df):.2f}% of all events).")

    # night = subject + night_id, in case night_id restarts per subject
    for name, key in (("subject", ["subject_id"]), ("night", ["subject_id", "night_id"])):
        g = tail_by_group(tail, df, key)
        if g is not None:
            g.to_csv(out_dir / f"tail_by_{name}.csv")
            print(f"\nTail events per {name} (top 10):")
            print(g.head(10).to_string())

    if not tail.empty:
        ctx = [c for c in CONTEXT_COLS if c in df.columns]
        if ctx:
            cmp = pd.DataFrame({
                "median_tail": tail[ctx].median(),
                "median_rest": df.drop(index=tail.index)[ctx].median(),
            }).round(3)
            print("\nContext features: tail vs rest (median, z-scores):")
            print(cmp.to_string())

    print(f"\nAll output saved in: {out_dir}")
    print("  - scatter_grid.png(.pdf) + scatter_<f1>__<f2>.png(.pdf)")
    print("  - pair_stats.csv       <- r/rho with and without the tail")
    print("  - tail_events.csv      <- tail events with metadata, to look up in the EEG")
    print("  - tail_by_subject.csv / tail_by_night.csv  <- concentrated in a few nights? -> artefact?")


if __name__ == "__main__":
    main()