"""
4.2_check_motion.py 
Is de tweede bult in motion_rms (na centreren) echt, of een artefact?

Wat het script doet: 
1. Leest de geschaalde features (vóór centreren) en de gecentreerde features.
2. Histogram van motion_rms vóór en na centreren (+ gladde dichtheidscurve).
3. Zoekt automatisch het dal tussen de twee bulten in de gecentreerde verdeling
   (of gebruik --thr om zelf een grens te kiezen) en telt per deelnemer welk
   deel van zijn events in de rechterbult ("bult 2") valt.
4. Rapporteert:
     - hoeveel deelnemers events in bult 2 hebben (>= 1 en >= 10% van hun events)
     - hoeveel deelnemers samen 50% van bult 2 leveren
     - het grootste aandeel van één deelnemer
     - hoeveel unieke motion-waarden er per nacht zijn (veel gelijke waarden
       = getrapte verdeling, wat bij centreren nep-bulten kan geven)
5. Figuur met drie panelen: histogram vóór, histogram na (met grens),
   en per deelnemer het aandeel events in bult 2 (gesorteerd).

Interpretatie:
- Bult 2 komt bij VEEL deelnemers voor, elk met een deel van hun events
  -> echt fenomeen: binnen personen zijn er arousals met en zonder hoofdbeweging.
- Bult 2 komt van een HANDVOL deelnemers
  -> persoons-/opname-effect (bijv. een paar mensen die veel bewegen of een
     afwijkende sensor), geen algemeen kenmerk van arousals.

Gebruik:
    python 4.2_check_motion.py
    python 4.2_check_motion.py --thr 1.0        # zelf de grens kiezen (z-schaal, gecentreerd)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

BASE = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2")
RED = BASE / "3. feature selection" / "reduced"
DEFAULT_BEFORE = RED / "arousal_features_reduced_scaled.csv"
DEFAULT_AFTER = RED / "arousal_features_reduced_scaled_centered_participant.csv"
DEFAULT_OUTDIR = BASE / "4. clustering" / "checks"

FEAT = "motion_rms"
KEYS = ["subject_id", "night_id", "event_idx"]
MIN_SHARE = 0.10

GREY = "#a9a89f"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#2b2b29"


def read_csv_nl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if df[FEAT].dtype == object:
        df = pd.read_csv(path, sep=";", decimal=",")
    return df


def find_valley(x: np.ndarray) -> float:
    """Laagste punt van de dichtheid tussen de twee hoogste toppen."""
    grid = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 512)
    dens = gaussian_kde(x)(grid)
    peaks = argrelextrema(dens, np.greater, order=10)[0]
    if len(peaks) < 2:
        raise RuntimeError("Geen twee toppen gevonden in de gecentreerde verdeling; geef zelf --thr op.")
    top2 = sorted(peaks[np.argsort(dens[peaks])[-2:]])
    seg = slice(top2[0], top2[1] + 1)
    return float(grid[seg][np.argmin(dens[seg])])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    ap.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--thr", type=float, default=None, help="Grens voor bult 2 (gecentreerde z-schaal)")
    args = ap.parse_args()

    b = read_csv_nl(args.before)
    a = read_csv_nl(args.after)
    df = b[KEYS + [FEAT]].rename(columns={FEAT: "before"}).merge(
        a[KEYS + [FEAT]].rename(columns={FEAT: "after"}), on=KEYS, how="inner", validate="one_to_one")
    print(f"Gekoppeld: {len(df)} events, {df['subject_id'].nunique()} deelnemers")

    thr = args.thr if args.thr is not None else find_valley(df["after"].to_numpy())
    df["bump2"] = df["after"] > thr
    print(f"Grens bult 2 (gecentreerd): {thr:.2f}  -> {100 * df['bump2'].mean():.1f}% van de events")

    # --- per deelnemer -------------------------------------------------------
    per = df.groupby("subject_id")["bump2"].agg(n_events="size", n_bump2="sum", share="mean")
    per = per.sort_values("share", ascending=False)
    counts = df.loc[df["bump2"], "subject_id"].value_counts()
    cum = counts.cumsum() / counts.sum()
    n_subj = len(per)
    summary = {
        "threshold_centered_z": thr,
        "pct_events_in_bump2": 100 * df["bump2"].mean(),
        "n_subjects_total": n_subj,
        "n_subjects_any_bump2": int((per["n_bump2"] > 0).sum()),
        f"n_subjects_ge_{int(MIN_SHARE * 100)}pct": int((per["share"] >= MIN_SHARE).sum()),
        "n_subjects_for_50pct_of_bump2": int((cum < 0.5).sum() + 1) if len(counts) else 0,
        "max_share_of_bump2_from_one_subject_pct": float(100 * counts.iloc[0] / counts.sum()) if len(counts) else 0,
        "median_share_within_subject_pct": float(100 * per["share"].median()),
    }

    # --- getraptheid: unieke waarden per nacht ------------------------------
    raw = b.groupby(["subject_id", "night_id"])[FEAT].agg(n="size", n_unique="nunique")
    raw["pct_unique"] = 100 * raw["n_unique"] / raw["n"]
    summary["median_pct_unique_motion_values_per_night"] = float(raw["pct_unique"].median())

    args.outdir.mkdir(parents=True, exist_ok=True)
    per.round(3).to_csv(args.outdir / "motion_bump2_per_participant.csv", sep=";")
    pd.Series(summary).round(3).to_csv(args.outdir / "motion_bump2_summary.csv", sep=";", header=["value"])

    # --- figuur ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, col, title in [(axes[0], "before", "Vóór centreren"), (axes[1], "after", "Na centreren per deelnemer")]:
        x = df[col].to_numpy()
        ax.hist(x, bins=80, color=GREY, alpha=0.6, density=True)
        g = np.linspace(x.min(), x.max(), 400)
        ax.plot(g, gaussian_kde(x)(g), color=INK, lw=1.2)
        ax.set_title(f"{title}: {FEAT}", loc="left")
        ax.set_xlabel("z-score")
        ax.set_ylabel("dichtheid")
    axes[1].axvline(thr, color=ORANGE, lw=1.5, ls="--")
    axes[1].text(thr, axes[1].get_ylim()[1] * 0.95, f"  grens {thr:.2f}", color=ORANGE, va="top", fontsize=8)

    ax = axes[2]
    ax.bar(range(n_subj), 100 * per["share"].to_numpy(), color=BLUE, width=0.85)
    ax.axhline(100 * df["bump2"].mean(), color=INK, lw=1, ls=":")
    ax.text(n_subj - 1, 100 * df["bump2"].mean(), "gemiddeld", ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_xlim(-1, n_subj)
    ax.set_xticks([])
    ax.set_xlabel(f"Deelnemers (gesorteerd), n = {n_subj}")
    ax.set_ylabel("% van eigen events in bult 2")
    ax.set_title("Per deelnemer: aandeel events in bult 2", loc="left")
    for a_ in axes:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.outdir / "motion_bump_check.png", dpi=200)
    plt.close(fig)

    print("\nSamenvatting:")
    for k, v in summary.items():
        print(f"  {k:45s} {v:.2f}" if isinstance(v, float) else f"  {k:45s} {v}")
    print(f"\nOutput in: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()