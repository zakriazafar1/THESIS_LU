"""
2_GMM.py - Gaussian Mixture Models op de geschaalde arousal-features

-------------------
1. Leest de geschaalde feature-matrix (z-scores).
2. Fit per k = 1..K_MAX een GMM met covariance_type="full" en n_init starts.
3. Rapporteert per k:
     - log-likelihood, AIC, BIC
     - ICL  = BIC + 2 * entropie (straft overlappende componenten af)
     - genormaliseerde entropie (0 = perfecte scheiding, 1 = volledige overlap)
     - gemiddelde max-posterior (hoe zeker het model gemiddeld is over de toewijzing)
     - % events met max-posterior >= 0.80 ("zeker toegewezen")
     - kleinste componentgewicht (vangt 'mini-componenten' voor uitschieters)
     - per component: gewicht en gemiddelde posterior van de toegewezen events
4. Slaat per event de labels + posteriors op voor k_BIC en k_ICL.
5. Maakt een figuur: BIC/ICL tegen k, en de zekerheid tegen k.

Interpretatie
---------------------------------
- BIC: bij scheve data en grote n kiest BIC vaak k > 1 omdat meerdere
  Gaussians één scheve wolk benaderen. Dáárom ICL + zekerheidsmaten ernaast.
- Lage gemiddelde max-posterior / hoge genormaliseerde entropie = componenten
  overlappen sterk -> eerder een continuüm dan discrete clusters/subtypes.

Gebruik
-------
    python 2_GMM.py                     # standaardpaden (DEFAULT_INPUT / DEFAULT_OUTDIR)
    python 2_GMM.py --input             ander.csv --outdir andere_map
    python 2_GMM.py --inspect           # alleen data-check, geen fits
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Standaardpaden 
# ---------------------------------------------------------------------------
BASE = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2")
DEFAULT_INPUT = BASE / "reduced feature matrix" / "arousal_features_reduced_scaled.csv"
DEFAULT_OUTDIR = BASE / "4. clustering" / "2. GMM"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURES = [
    "mean_delta_ratio",
    "mean_theta_ratio",
    "mean_alpha_ratio",
    "mean_sigma_ratio",
    "mean_beta_ratio",
    "duration_sec",
    "oxy_amp_ratio",
    "motion_rms",
]
ID_COLS = ["subject_id", "group", "night_id", "event_idx", "stage_rk"]

K_MAX = 10
N_INIT = 20
SEED = 2554542
REG_COVAR = 1e-6      # sklearn-default; kleine ridge op de diagonaal voor numerieke stabiliteit
MAX_ITER = 1500
CERTAIN_THR = 0.85    # drempel voor "zeker toegewezen" event

# ---------------------------------------------------------------------------
# Data laden
# ---------------------------------------------------------------------------
def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if df[FEATURES].dtypes.eq(object).any():
        df = pd.read_csv(path, sep=";", decimal=",")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Ontbrekende feature-kolommen: {missing}")
    n_before = len(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Let op: {n_before - len(df)} events met NaN in features verwijderd.")
    return df


def inspect(df: pd.DataFrame) -> None:
    print(f"\nEvents: {len(df)} | subjects: {df['subject_id'].nunique()} | "
          f"nachten: {df.groupby(['subject_id', 'night_id']).ngroups}")
    print("\nFeatures (moeten ~mean 0, sd 1 zijn):")
    print(df[FEATURES].describe().T[["mean", "std", "min", "max"]].round(2).to_string())
    print("\nScheefheid:")
    print(df[FEATURES].skew().round(2).to_string())


# ---------------------------------------------------------------------------
# Maten
# ---------------------------------------------------------------------------
def entropy_terms(resp: np.ndarray) -> float:
    """Totale classificatie-entropie E = -sum_i sum_k tau_ik * log(tau_ik)."""
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

    loglik = float(gmm.score(X) * n)
    bic = float(gmm.bic(X))
    E = entropy_terms(resp)
    icl = bic + 2.0 * E
    norm_entropy = E / (n * np.log(k)) if k > 1 else 0.0

    row = {
        "k": k,
        "loglik": loglik,
        "n_params": int(gmm._n_parameters()),
        "AIC": float(gmm.aic(X)),
        "BIC": bic,
        "ICL": icl,
        "entropy": E,
        "norm_entropy": norm_entropy,
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
            "component": c,
            "weight_pct": float(gmm.weights_[c] * 100),
            "n_assigned": int(m.sum()),
            "mean_posterior_assigned": float(maxpost[m].mean()) if m.any() else np.nan,
        })
    return gmm, row, comps


# ---------------------------------------------------------------------------
# Figuur
# ---------------------------------------------------------------------------
def plot_results(res: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(res["k"], res["BIC"], "o-", label="BIC")
    ax.plot(res["k"], res["ICL"], "s--", label="ICL")
    for col, mk in [("BIC", "o"), ("ICL", "s")]:
        kb = res.loc[res[col].idxmin(), "k"]
        ax.plot(kb, res[col].min(), mk, ms=12, mfc="none", mec="black")
    ax.set_xlabel("Aantal componenten k")
    ax.set_ylabel("Criterium (lager = beter)")
    ax.set_title("Modelselectie")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(res["k"], res["mean_max_posterior"], "o-", label="Gem. max-posterior")
    ax.plot(res["k"], 1 - res["norm_entropy"], "s--", label="1 − genorm. entropie")
    ax.plot(res["k"], res["pct_certain"] / 100, "^:", label=f"Aandeel posterior ≥ {CERTAIN_THR}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Aantal componenten k")
    ax.set_ylabel("Zekerheid van toewijzing")
    ax.set_title("Scheiding tussen componenten")
    ax.legend(frameon=False, fontsize=8)

    for a in axes:
        a.set_xticks(res["k"])
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global K_MAX, N_INIT, SEED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--kmax", type=int, default=K_MAX)
    ap.add_argument("--n-init", type=int, default=N_INIT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--inspect", action="store_true", help="Alleen data-check, geen fits")
    args = ap.parse_args()
    K_MAX, N_INIT, SEED = args.kmax, args.n_init, args.seed

    print(f"Inlezen: {args.input}")
    df = load_features(args.input)
    inspect(df)
    if args.inspect:
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    X = df[FEATURES].to_numpy(dtype=float)

    rows, comps, models = [], [], {}
    print(f"\nGMM fitten: k = 1..{K_MAX}, covariance='full', n_init={N_INIT}, seed={SEED}")
    for k in range(1, K_MAX + 1):
        gmm, row, comp = fit_one_k(X, k)
        rows.append(row)
        comps.extend(comp)
        models[k] = gmm
        print(f"  k={k}: BIC={row['BIC']:.0f}  ICL={row['ICL']:.0f}  "
              f"max-post={row['mean_max_posterior']:.3f}  zeker={row['pct_certain']:.1f}%  "
              f"min-gewicht={row['min_weight_pct']:.1f}%  "
              f"{'' if row['converged'] else '[NIET GECONVERGEERD] '}({row['fit_sec']}s)")

    res = pd.DataFrame(rows)
    res["delta_BIC"] = res["BIC"] - res["BIC"].min()
    res["delta_ICL"] = res["ICL"] - res["ICL"].min()
    k_bic = int(res.loc[res["BIC"].idxmin(), "k"])
    k_icl = int(res.loc[res["ICL"].idxmin(), "k"])

    res.round(4).to_csv(args.outdir / "gmm_model_selection.csv", sep=";", index=False)
    pd.DataFrame(comps).round(4).to_csv(args.outdir / "gmm_components.csv", sep=";", index=False)

    # Per-event labels + posteriors voor k_BIC en k_ICL (stabiliteit/interpretatie)
    id_cols = [c for c in ID_COLS if c in df.columns]
    for tag, k in {"bic": k_bic, "icl": k_icl}.items():
        resp = models[k].predict_proba(X)
        out = df[id_cols].copy()
        out["label"] = resp.argmax(axis=1)
        out["max_posterior"] = resp.max(axis=1)
        for c in range(k):
            out[f"p_{c}"] = resp[:, c]
        out.round(4).to_csv(args.outdir / f"gmm_labels_k{k}_{tag}.csv", sep=";", index=False)

    # Componentmiddens (op z-schaal) van de gekozen modellen
    for tag, k in {"bic": k_bic, "icl": k_icl}.items():
        means = pd.DataFrame(models[k].means_, columns=FEATURES)
        means.insert(0, "weight_pct", models[k].weights_ * 100)
        means.index.name = "component"
        means.round(3).to_csv(args.outdir / f"gmm_means_z_k{k}_{tag}.csv", sep=";")

    plot_results(res, args.outdir / "gmm_model_selection.png")

    summary = {
        "input": str(args.input),
        "n_events": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()),
        "features": FEATURES,
        "settings": {"k_max": K_MAX, "covariance_type": "full", "n_init": N_INIT,
                     "seed": SEED, "reg_covar": REG_COVAR, "max_iter": MAX_ITER},
        "k_best_BIC": k_bic,
        "k_best_ICL": k_icl,
    }
    (args.outdir / "gmm_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nResultaat")
    print(res[["k", "BIC", "delta_BIC", "ICL", "delta_ICL", "norm_entropy",
               "mean_max_posterior", "pct_certain", "min_weight_pct"]].round(3).to_string(index=False))
    print(f"\nBeste k volgens BIC: {k_bic} | volgens ICL: {k_icl}")
    print("Let op: BIC is geen toets. Beoordeel k pas na vergelijking met het nulmodel (stap 2).")
    print(f"Output in: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()