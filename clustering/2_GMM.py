"""
2_GMM.py - Gaussian mixture models on the reduced arousal features.

What the script does:
1. Reads the reduced, standardised feature matrix (z-scores).
2. Fits a full-covariance GMM for K = 1..K_MAX on the 8 features
   (or on PC1..PCn when the input is a PCA-score file) and reports per K:
     - log-likelihood, number of parameters, AIC, BIC, ICL
     - normalised entropy (0 = perfect separation, 1 = complete overlap)
     - mean maximum posterior probability
     - % of events with maximum posterior >= 0.85 
     - smallest mixing proportion
3. Saves per-event labels + posteriors and component means for K_BIC and K_ICL.
4. Makes APA figures:
     - Figure A/B: BIC and ICL against K, and classification certainty against K
     - Component profiles for K_ICL on the 8 (standardised) features/5 PCs
5. All output goes to <outdir>/<tag>/, so different runs are kept apart

Usage:
    python 2_GMM.py  # default input
    python 2_GMM.py --input other.csv --tag centered_participant
    python 2_GMM.py --input other.csv --outdir other_folder --tag test
    python 2_GMM.py --input pca_scores_5pc_unwhitened.csv --tag PCA    # PC1..PCn detected automatically
    python 2_GMM.py --input x.csv --cols PC1,PC2,PC3 --tag PC1-3       # choose columns yourself
    python 2_GMM.py --inspect # data check only, no fitting
"""

import argparse
import re
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
BASE = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2")
DEFAULT_INPUT = BASE / "reduced feature matrix" / "arousal_features_reduced_scaled.csv"
DEFAULT_OUTDIR = BASE / "4. clustering" / "2. GMM"
DEFAULT_TAG = "features_8"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_FEATURES = [
    "mean_delta_ratio",
    "mean_theta_ratio",
    "mean_alpha_ratio",
    "mean_sigma_ratio",
    "mean_beta_ratio",
    "duration_sec",
    "oxy_amp_ratio",
    "motion_rms",
]
FEATURES = list(DEFAULT_FEATURES)   
FEATURE_LABELS = {
    "mean_delta_ratio": "Mean delta ratio",
    "mean_theta_ratio": "Mean theta ratio",
    "mean_alpha_ratio": "Mean alpha ratio",
    "mean_sigma_ratio": "Mean sigma ratio",
    "mean_beta_ratio": "Mean beta ratio",
    "duration_sec": "Duration",
    "oxy_amp_ratio": "PPG ratio",
    "motion_rms": "Motion RMS",
}
ID_COLS = ["subject_id", "group", "night_id", "event_idx", "stage_rk"]

K_MAX = 10
N_INIT = 20
SEED = 2554542
REG_COVAR = 1e-6      # sklearn default
MAX_ITER = 1500
CERTAIN_THR = 0.85    # threshold for a "clearly assigned" event

# Colours
C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
COMP_COLORS = [C_BLUE, C_ORANGE, C_AQUA, C_VIOLET, "#eda100", "#e87ba4", "#008300", "#e34948"]
COMP_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK = "#000000"
GREY = "#7f7f7f"


def ensure_parent(path: Path) -> Path:
    """Create the parent folder of a file if it does not exist (again)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return Path(path)


def write_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    frame.to_csv(ensure_parent(path), **kwargs)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_features(path: Path, cols_arg: str | None = None) -> pd.DataFrame:
    """Reads the CSV and decides which columns to cluster on (sets the global FEATURES):
    1. --cols if given (comma-separated);
    2. otherwise the 8 reduced features, if all present;
    3. otherwise all columns named PC1, PC2, ... (e.g. a PCA-score file)."""
    global FEATURES
    sep = ";"
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] == 1:                         
        sep = ","
        df = pd.read_csv(path, sep=sep)
    if cols_arg:
        cols = [c.strip() for c in cols_arg.split(",") if c.strip()]
    elif all(c in df.columns for c in DEFAULT_FEATURES):
        cols = list(DEFAULT_FEATURES)
    else:
        cols = sorted([c for c in df.columns if re.fullmatch(r"PC\d+", str(c))], key=lambda c: int(c[2:]))
    missing = [c for c in cols if c not in df.columns]
    if not cols or missing:
        raise ValueError(f"Could not find the columns to cluster on (missing: {missing or 'none found'}). "
                         f"Available columns: {list(df.columns)}. Use --cols to name them.")
    FEATURES = cols
    if sep == ";" and df[FEATURES].dtypes.eq(object).any():   # Dutch decimal commas
        df = pd.read_csv(path, sep=";", decimal=",")
    n_before = len(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Note: removed {n_before - len(df)} events with missing values.")
    print(f"Clustering on {len(FEATURES)} columns: {', '.join(FEATURES)}")
    return df


def inspect(df: pd.DataFrame) -> None:
    n_nights = df.groupby(["subject_id", "night_id"]).ngroups if "night_id" in df else "n/a"
    print(f"\nEvents: {len(df)} | participants: {df['subject_id'].nunique()} | nights: {n_nights}")
    print("\nInput columns (features: approx. mean 0, SD 1; PCs: mean 0, SD = sqrt(eigenvalue)):")
    print(df[FEATURES].describe().T[["mean", "std", "min", "max"]].round(2).to_string())
    print("\nSkewness:")
    print(df[FEATURES].skew().round(2).to_string())


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------
def entropy_terms(resp: np.ndarray) -> float:
    """Total classification entropy EN = -sum_i sum_k tau_ik * log(tau_ik)."""
    r = np.clip(resp, 1e-300, 1.0)
    return float(-(resp * np.log(r)).sum())


def fit_one_k(X: np.ndarray, k: int) -> tuple[GaussianMixture, dict, list]:
    t0 = time.time()
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=N_INIT,
        max_iter=MAX_ITER,
        reg_covar=REG_COVAR,
        random_state=SEED,
    ).fit(X)

    n = X.shape[0]
    resp = gmm.predict_proba(X)
    labels = resp.argmax(axis=1)
    maxpost = resp.max(axis=1)

    bic = float(gmm.bic(X))
    E = entropy_terms(resp)
    row = {
        "k": k,
        "loglik": float(gmm.score(X) * n),
        "n_params": int(gmm._n_parameters()),
        "AIC": float(gmm.aic(X)),
        "BIC": bic,
        "ICL": bic + 2.0 * E,
        "entropy": E,
        "norm_entropy": E / (n * np.log(k)) if k > 1 else 0.0,
        "mean_max_posterior": float(maxpost.mean()),
        "pct_certain": float((maxpost >= CERTAIN_THR).mean() * 100),
        "min_weight_pct": float(gmm.weights_.min() * 100),
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "fit_sec": round(time.time() - t0, 1),
    }
    comps = []
    for c in range(k):
        m = labels == c
        comps.append({
            "k": k,
            "component": c + 1,
            "weight_pct": float(gmm.weights_[c] * 100),
            "n_assigned": int(m.sum()),
            "mean_posterior_assigned": float(maxpost[m].mean()) if m.any() else np.nan,
        })
    return gmm, row, comps


# ---------------------------------------------------------------------------
# Figures (APA style)
# ---------------------------------------------------------------------------
def apa_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,          
        "legend.frameon": False,
    })
    return plt


def panel_letters(fig, axes, letters):
    """Place panel letters at the same height, aligned with the left edge of each
    panel including its tick and axis labels (so A and B line up consistently)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    top = max(inv.transform(ax.get_tightbbox(r))[1, 1] for ax in axes)   # includes legends above axes
    for ax, letter in zip(axes, letters):
        x0 = inv.transform(ax.get_tightbbox(r))[0, 0]
        fig.text(x0, top + 0.02, letter, fontsize=10, fontweight="bold", ha="left", va="bottom", color=INK)


def save(fig, out_base: Path):
    fig.savefig(ensure_parent(out_base.with_suffix(".png")), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)


def plot_model_selection(res: pd.DataFrame, out_base: Path) -> None:
    plt = apa_style()
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    inner = 7  # font size inside the plotting area 

    # Two panels of identical size and margins 
    fig = plt.figure(figsize=(6.9, 2.9))
    w, h, bottom = 0.355, 0.74, 0.17
    axA = fig.add_axes([0.095, bottom, w, h])
    axB = fig.add_axes([0.095 + w + 0.115, bottom, w, h])
    k = res["k"].to_numpy()

    # A: BIC and ICL
    ax = axA
    ax.plot(k, res["BIC"] / 1000, color=C_BLUE, marker="o", ms=4, lw=1.3, label="BIC")
    ax.plot(k, res["ICL"] / 1000, color=C_ORANGE, marker="s", ms=4, lw=1.3, ls="--", label="ICL")
    for col, mk in [("BIC", "o"), ("ICL", "s")]:
        i = res[col].idxmin()
        ax.plot(res.loc[i, "k"], res.loc[i, col] / 1000, marker=mk, ms=9, mfc="none", mec=INK, mew=1.0,
                ls="none", zorder=5)
    ax.plot([], [], marker="o", ms=7, mfc="none", mec=INK, mew=1.0, ls="none", label="Best K")
    ax.set_ylabel("Criterion value (\u00d7 1,000)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.1f}".rstrip("0").rstrip(".")))
    ax.legend(loc="center right", handlelength=2.0, fontsize=inner, borderaxespad=0.3)

    # B: classification certainty
    ax = axB
    ax.plot(k, res["mean_max_posterior"], color=C_BLUE, marker="o", ms=4, lw=1.3,
            label="Mean maximum posterior")
    ax.plot(k, 1 - res["norm_entropy"], color=C_ORANGE, marker="s", ms=4, lw=1.3, ls="--",
            label="1 \u2212 normalised entropy")
    ax.plot(k, res["pct_certain"] / 100, color=C_AQUA, marker="^", ms=4.5, lw=1.3, ls=":",
            label=f"Proportion posterior \u2265 {CERTAIN_THR:.2f}".replace("0.", "."))
    ax.set_ylim(0, 1.03)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "0" if abs(v) < 1e-9 else ("1.0" if v >= 1 else f"{v:.1f}".replace("0.", "."))))
    ax.set_ylabel("Classification certainty")
    ax.legend(loc="lower left", handlelength=2.0, fontsize=inner, borderaxespad=0.3)

    for ax, letter in [(axA, "A"), (axB, "B")]:
        ax.set_xlabel("Number of components (K)")
        ax.set_xticks(k)
        ax.set_xlim(k.min() - 0.5, k.max() + 0.5)
        ax.tick_params(labelsize=inner)
        ax.text(-0.2, 1.03, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="left", color=INK)

    fig.savefig(ensure_parent(out_base.with_suffix(".png")), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_component_profiles(df: pd.DataFrame, labels: np.ndarray, weights: np.ndarray, out_base: Path) -> None:
    """Mean (standardised) value of each original feature per component, with 95% CI."""
    plt = apa_style()
    k = int(labels.max()) + 1
    feats = FEATURES[::-1]                       
    y = np.arange(len(feats))
    offsets = np.linspace(0.2, -0.2, k) if k > 1 else [0.0]  

    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(feats) + 0.9))
    ax.axvline(0, color=GREY, lw=0.8, zorder=0)
    for yy in y[:-1]:
        ax.axhline(yy + 0.5, color="#e6e6e6", lw=0.6, zorder=0)
    handles = []
    for c in range(k):
        sub = df.loc[labels == c, feats]
        m = sub.mean().to_numpy()
        se = sub.std(ddof=1).to_numpy() / np.sqrt(len(sub))
        col = COMP_COLORS[c % len(COMP_COLORS)]
        mk = COMP_MARKERS[c % len(COMP_MARKERS)]
        ax.errorbar(m, y + offsets[c], xerr=1.96 * se, fmt="none", ecolor=col, elinewidth=1.0, capsize=0)
        ax.plot(m, y + offsets[c], ls="none", marker=mk, ms=5.5, color=col, mec="white", mew=0.5)
        handles.append(Line2D([], [], ls="none", marker=mk, ms=6, color=col, mec="white", mew=0.5,
                              label=f"Component {c + 1} ({100 * weights[c]:.0f}%)"))
    ax.set_yticks(y, [FEATURE_LABELS.get(f, f) for f in feats])
    is_pc = all(re.fullmatch(r"PC\d+", str(f)) for f in feats)
    ax.set_xlabel("Mean principal component score" if is_pc else "Mean standardised value (z)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(feats) - 0.4)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(k, 4), handletextpad=0.3)
    save(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fit all models for one input
# ---------------------------------------------------------------------------
def run_gmm(tag: str, X: np.ndarray, df: pd.DataFrame, outdir: Path, input_path: Path) -> None:
    cols = list(FEATURES)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, comps, models = [], [], {}
    print(f"\n[{tag}] Fitting GMMs: K = 1..{K_MAX}, covariance = 'full', "
          f"n_init = {N_INIT}, seed = {SEED}, dimensions = {X.shape[1]}")
    for k in range(1, K_MAX + 1):
        gmm, row, comp = fit_one_k(X, k)
        rows.append(row)
        comps.extend(comp)
        models[k] = gmm
        print(f"  K={k:2d}: BIC={row['BIC']:.0f}  ICL={row['ICL']:.0f}  "
              f"mean max post={row['mean_max_posterior']:.3f}  certain={row['pct_certain']:.1f}%  "
              f"smallest={row['min_weight_pct']:.1f}%  "
              f"{'' if row['converged'] else '[NOT CONVERGED] '}({row['fit_sec']}s)")

    res = pd.DataFrame(rows)
    res["delta_BIC"] = res["BIC"] - res["BIC"].min()
    res["delta_ICL"] = res["ICL"] - res["ICL"].min()
    k_bic = int(res.loc[res["BIC"].idxmin(), "k"])
    k_icl = int(res.loc[res["ICL"].idxmin(), "k"])

    write_csv(res.round(4), outdir / "gmm_model_selection.csv", sep=";", index=False)
    write_csv(pd.DataFrame(comps).round(4), outdir / "gmm_components.csv", sep=";", index=False)

    id_cols = [c for c in ID_COLS if c in df.columns]
    for sel, k in {"bic": k_bic, "icl": k_icl}.items():
        resp = models[k].predict_proba(X)
        out = df[id_cols].copy()
        out["label"] = resp.argmax(axis=1)
        out["max_posterior"] = resp.max(axis=1)
        for c in range(k):
            out[f"p_{c}"] = resp[:, c]
        write_csv(out.round(4), outdir / f"gmm_labels_k{k}_{sel}.csv", sep=";", index=False)

        # component means in the fitted space
        means = pd.DataFrame(models[k].means_, columns=cols)
        means.insert(0, "weight_pct", models[k].weights_ * 100)
        means.index = pd.Index(range(1, k + 1), name="component")
        write_csv(means.round(3), outdir / f"gmm_means_k{k}_{sel}.csv", sep=";")

        # component means on the 8 original standardised features 
        prof = df[FEATURES].groupby(out["label"].to_numpy() + 1).mean()
        prof.insert(0, "n_events", out["label"].value_counts().sort_index().to_numpy())
        prof.index.name = "component"
        write_csv(prof.round(3), outdir / f"gmm_feature_profiles_k{k}_{sel}.csv", sep=";")

    plot_model_selection(res, outdir / "fig_gmm_model_selection")
    if k_icl > 1:
        lab = models[k_icl].predict(X)
        plot_component_profiles(df, lab, models[k_icl].weights_, outdir / f"fig_gmm_profiles_k{k_icl}_icl")

    summary = {
        "tag": tag,
        "input": str(input_path),
        "features": cols,
        "n_events": int(len(df)),
        "n_participants": int(df["subject_id"].nunique()),
        "settings": {"k_max": K_MAX, "covariance_type": "full", "n_init": N_INIT, "seed": SEED,
                     "reg_covar": REG_COVAR, "max_iter": MAX_ITER, "certain_threshold": CERTAIN_THR},
        "k_best_BIC": k_bic,
        "k_best_ICL": k_icl,
    }
    ensure_parent(outdir / "gmm_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[{tag}] Results")
    print(res[["k", "BIC", "delta_BIC", "ICL", "delta_ICL", "norm_entropy",
               "mean_max_posterior", "pct_certain", "min_weight_pct"]].round(3).to_string(index=False))
    print(f"[{tag}] Best K by BIC: {k_bic} | by ICL: {k_icl}")

    print("Note: BIC tends to favour extra overlapping components for skewed data; "
          "interpret K together with ICL and the certainty measures.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global K_MAX, N_INIT, SEED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Reduced, standardised feature CSV")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Parent output folder")
    ap.add_argument("--tag", type=str, default=DEFAULT_TAG,
                    help="Name of the subfolder for this run (e.g. features_8, centered_participant)")
    ap.add_argument("--cols", type=str, default=None,
                    help="Comma-separated columns to cluster on (default: the 8 features, else PC1..PCn)")
    ap.add_argument("--kmax", type=int, default=K_MAX)
    ap.add_argument("--n-init", type=int, default=N_INIT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--inspect", action="store_true", help="Data check only, no fitting")
    args = ap.parse_args()
    K_MAX, N_INIT, SEED = args.kmax, args.n_init, args.seed

    print(f"Reading: {args.input}")
    df = load_features(args.input, args.cols)
    inspect(df)
    if args.inspect:
        return

    outdir = args.outdir / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    X = df[FEATURES].to_numpy(dtype=float)
    run_gmm(args.tag, X, df, outdir, args.input)
    print(f"\nOutput in: {outdir.resolve()}")


if __name__ == "__main__":
    main()