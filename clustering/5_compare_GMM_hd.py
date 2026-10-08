"""
4_compare_GMM_HDBSCAN.py — Vergelijk de GMM-indeling (k = 2, ICL) met de HDBSCAN-indeling.

Doel
----
Laten zien of twee methoden met verschillende aannames (GMM = modelgebaseerd,
HDBSCAN = dichtheidsgebaseerd) dezelfde structuur in de data aanwijzen, en of
die structuur over veel deelnemers verspreid is (en niet van een handvol mensen komt).

Wat het script doet
-------------------
1. Leest de GMM-labels (gmm_labels_k2_icl.csv uit 2_GMM.py), de HDBSCAN-labels
   en de geschaalde features, en koppelt ze per event (subject_id, night_id, event_idx).
2. Overeenkomst tussen de methoden:
     - kruistabel GMM-component x HDBSCAN-label (aantallen, rij- en kolom-%)
     - ARI en AMI, met ruis (-1) als eigen label én met ruis weggelaten
     - per HDBSCAN-label: gemiddelde GMM-kans op elke component (p_0, p_1)
       -> laat zien wáár op het GMM-continuüm elk HDBSCAN-cluster ligt
3. Spreiding over deelnemers, per GMM-component en per HDBSCAN-label:
     - aantal deelnemers met >= 1 event en met >= 10% van hun events in die groep
     - grootste aandeel van één deelnemer
     - minimaal aantal deelnemers dat samen 50% van de groep levert
     - Cramér's V tussen groep en deelnemer (0 = geen samenhang, 1 = groep = persoon)
4. Verdeling over slaapstadia per groep.
5. Featureprofiel (mediaan z-score) per groep.
6. Figuur: kruistabel-heatmap, featureprofielen, en GMM-kans per HDBSCAN-label.

Invoer HDBSCAN-bestand
----------------------
';'-CSV met per event een label-kolom (ruis = -1). Bij voorkeur ook
subject_id, night_id, event_idx om op te koppelen. Ontbreken die, dan wordt op
rijvolgorde gekoppeld (alleen als het aantal rijen gelijk is; je krijgt een waarschuwing).
De naam van de label-kolom wordt automatisch gezocht (zie HDB_LABEL_CANDIDATES)
of geef hem op met --hdb-col.

Gebruik
-------
    python 4_compare_GMM_HDBSCAN.py
    python 4_compare_GMM_HDBSCAN.py --hdbscan pad\\naar\\hdbscan_labels.csv --hdb-col cluster
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

# ---------------------------------------------------------------------------
# Standaardpaden
# ---------------------------------------------------------------------------
BASE = Path(r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2")
DEFAULT_GMM = BASE / "4. clustering" / "2. GMM" / "gmm_labels_k2_icl.csv"
DEFAULT_HDB = BASE / "4. clustering" / "4. hdbscan" / "whitened" / "hdbscan_final_labels.csv"   # <-- aanpassen aan jouw bestand
DEFAULT_FEATURES = BASE / "reduced feature matrix" / "arousal_features_reduced_scaled.csv"
DEFAULT_OUTDIR = BASE / "4. clustering" / "comparison gmm_hdbscan"

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
KEYS = ["subject_id", "night_id", "event_idx"]
HDB_LABEL_CANDIDATES = ["hdbscan_label", "hdb_label", "cluster", "label", "labels"]
STAGE_NAMES = {1: "N1", 2: "N2", 3: "N3", 4: "N3", 5: "REM"}
MIN_SHARE = 0.10   # "deelnemer heeft >= 10% van zijn events in deze groep"


# ---------------------------------------------------------------------------
# Inlezen
# ---------------------------------------------------------------------------
def read_csv_nl(path: Path) -> pd.DataFrame:
    """';'-CSV; valt terug op komma-decimalen als numerieke kolommen als tekst binnenkomen."""
    df = pd.read_csv(path, sep=";")
    num_like = [c for c in df.columns if c not in ("subject_id", "group", "night_id")]
    if any(df[c].dtype == object for c in num_like):
        df2 = pd.read_csv(path, sep=";", decimal=",")
        if sum(df2[c].dtype == object for c in num_like) < sum(df[c].dtype == object for c in num_like):
            df = df2
    return df


def find_label_col(df: pd.DataFrame, user_col: str | None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"Kolom '{user_col}' niet in HDBSCAN-bestand. Kolommen: {list(df.columns)}")
        return user_col
    for c in HDB_LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Geen label-kolom gevonden in HDBSCAN-bestand. Kolommen: {list(df.columns)}. "
                     f"Geef hem op met --hdb-col.")


def merge_all(gmm: pd.DataFrame, hdb: pd.DataFrame, feats: pd.DataFrame, hdb_col: str) -> pd.DataFrame:
    hdb = hdb.rename(columns={hdb_col: "hdb"})
    if all(k in hdb.columns for k in KEYS):
        df = gmm.merge(hdb[KEYS + ["hdb"]], on=KEYS, how="inner", validate="one_to_one")
        if len(df) < len(gmm):
            print(f"  Let op: {len(gmm) - len(df)} GMM-events niet gevonden in HDBSCAN-bestand.")
    else:
        if len(hdb) != len(gmm):
            raise ValueError("HDBSCAN-bestand heeft geen subject_id/night_id/event_idx én een ander "
                             f"aantal rijen ({len(hdb)}) dan de GMM-labels ({len(gmm)}). Kan niet koppelen.")
        print("  WAARSCHUWING: HDBSCAN-bestand heeft geen ID-kolommen; koppeling op rijvolgorde.")
        df = gmm.copy()
        df["hdb"] = hdb["hdb"].to_numpy()

    df = df.merge(feats[KEYS + FEATURES], on=KEYS, how="left", validate="one_to_one")
    if df[FEATURES].isna().any().any():
        print("  Let op: features ontbreken voor sommige events (featureprofiel op minder events).")
    df["gmm"] = df["label"].astype(int)
    df["hdb"] = df["hdb"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Maten
# ---------------------------------------------------------------------------
def name_hdb(x: int) -> str:
    return "ruis" if x == -1 else f"H{x}"


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a, b)
    if min(tab.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def participant_spread(df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
    n_subj_total = df["subject_id"].nunique()
    share = pd.crosstab(df["subject_id"], df[col], normalize="index")   # per deelnemer: aandeel per groep
    rows = []
    for g in sorted(df[col].unique()):
        sub = df[df[col] == g]
        counts = sub["subject_id"].value_counts()
        cum = counts.cumsum() / counts.sum()
        rows.append({
            "method": method,
            "group": name_hdb(g) if method == "HDBSCAN" else f"C{g}",
            "n_events": len(sub),
            "pct_events": 100 * len(sub) / len(df),
            "n_subjects_any": int(counts.size),
            f"n_subjects_ge_{int(MIN_SHARE*100)}pct": int((share[g] >= MIN_SHARE).sum()),
            "n_subjects_total": n_subj_total,
            "max_share_one_subject_pct": 100 * counts.iloc[0] / counts.sum(),
            "n_subjects_for_50pct": int((cum < 0.5).sum() + 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figuur
# ---------------------------------------------------------------------------
def plot_all(df: pd.DataFrame, tab_rowpct: pd.DataFrame, profiles: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.6, 1.1]})

    # 1. Kruistabel (rij-%: per HDBSCAN-label, verdeling over GMM-componenten)
    ax = axes[0]
    im = ax.imshow(tab_rowpct.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(tab_rowpct.shape[1]), tab_rowpct.columns)
    ax.set_yticks(range(tab_rowpct.shape[0]), tab_rowpct.index)
    for i in range(tab_rowpct.shape[0]):
        for j in range(tab_rowpct.shape[1]):
            v = tab_rowpct.iat[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", color="white" if v > 60 else "black", fontsize=9)
    ax.set_xlabel("GMM-component")
    ax.set_ylabel("HDBSCAN-label")
    ax.set_title("Per HDBSCAN-label: % in elke GMM-component")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 2. Featureprofielen (mediaan z)
    ax = axes[1]
    vmax = max(1.0, float(np.nanmax(np.abs(profiles.to_numpy()))))
    im = ax.imshow(profiles.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(FEATURES)), [f.replace("mean_", "").replace("_ratio", "") for f in FEATURES],
                  rotation=40, ha="right")
    ax.set_yticks(range(len(profiles)), profiles.index)
    for i in range(profiles.shape[0]):
        for j in range(profiles.shape[1]):
            ax.text(j, i, f"{profiles.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Featureprofiel per groep (mediaan z-score)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 3. Positie van HDBSCAN-labels op het GMM-continuüm
    ax = axes[2]
    pcols = sorted([c for c in df.columns if c.startswith("p_")])
    pc = "p_0" if "p_0" in pcols else pcols[0]
    groups = sorted(df["hdb"].unique())
    data = [df.loc[df["hdb"] == g, pc].to_numpy() for g in groups]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(groups) + 1), [name_hdb(g) for g in groups])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(f"GMM-kans op component {pc[2:]} ({pc})")
    ax.set_xlabel("HDBSCAN-label")
    ax.set_title("Waar liggen de HDBSCAN-groepen\nop het GMM-continuüm?")

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmm", type=Path, default=DEFAULT_GMM, help="gmm_labels_k2_icl.csv uit 2_GMM.py")
    ap.add_argument("--hdbscan", type=Path, default=DEFAULT_HDB, help="CSV met HDBSCAN-labels per event")
    ap.add_argument("--hdb-col", type=str, default=None, help="Naam van de label-kolom in het HDBSCAN-bestand")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="Geschaalde feature-matrix")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    print(f"GMM:      {args.gmm}\nHDBSCAN:  {args.hdbscan}\nFeatures: {args.features}")
    gmm = read_csv_nl(args.gmm)
    hdb = read_csv_nl(args.hdbscan)
    feats = read_csv_nl(args.features)
    hdb_col = find_label_col(hdb, args.hdb_col)
    print(f"  HDBSCAN-labelkolom: '{hdb_col}'")
    df = merge_all(gmm, hdb, feats, hdb_col)
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"  Gekoppeld: {len(df)} events, {df['subject_id'].nunique()} deelnemers")

    # --- 1. Kruistabel -----------------------------------------------------
    hdb_named = df["hdb"].map(name_hdb)
    gmm_named = "C" + df["gmm"].astype(str)
    order_h = [name_hdb(g) for g in sorted(df["hdb"].unique())]
    tab = pd.crosstab(hdb_named, gmm_named).reindex(order_h)
    tab_row = (pd.crosstab(hdb_named, gmm_named, normalize="index") * 100).reindex(order_h)
    tab_col = (pd.crosstab(hdb_named, gmm_named, normalize="columns") * 100).reindex(order_h)
    with open(args.outdir / "crosstab.csv", "w", encoding="utf-8") as f:
        f.write("# Aantallen\n"); tab.to_csv(f, sep=";")
        f.write("\n# Rij-% (per HDBSCAN-label: verdeling over GMM-componenten)\n"); tab_row.round(1).to_csv(f, sep=";")
        f.write("\n# Kolom-% (per GMM-component: verdeling over HDBSCAN-labels)\n"); tab_col.round(1).to_csv(f, sep=";")

    # --- 2. Overeenkomst ---------------------------------------------------
    no_noise = df["hdb"] != -1
    agree = pd.DataFrame([
        {"comparison": "met ruis als eigen label", "n_events": len(df),
         "ARI": adjusted_rand_score(df["gmm"], df["hdb"]),
         "AMI": adjusted_mutual_info_score(df["gmm"], df["hdb"])},
        {"comparison": "zonder ruis", "n_events": int(no_noise.sum()),
         "ARI": adjusted_rand_score(df.loc[no_noise, "gmm"], df.loc[no_noise, "hdb"]) if no_noise.any() else np.nan,
         "AMI": adjusted_mutual_info_score(df.loc[no_noise, "gmm"], df.loc[no_noise, "hdb"]) if no_noise.any() else np.nan},
    ])
    agree.round(4).to_csv(args.outdir / "agreement_ari_ami.csv", sep=";", index=False)

    pcols = sorted([c for c in df.columns if c.startswith("p_")])
    post = df.groupby(hdb_named)[pcols + ["max_posterior"]].agg(["mean", "median"]).reindex(order_h)
    post.columns = [f"{a}_{b}" for a, b in post.columns]
    post.round(3).to_csv(args.outdir / "gmm_posterior_per_hdbscan_label.csv", sep=";")

    # --- 3. Spreiding over deelnemers -------------------------------------
    spread = pd.concat([participant_spread(df, "gmm", "GMM"),
                        participant_spread(df, "hdb", "HDBSCAN")], ignore_index=True)
    cv = pd.DataFrame([
        {"method": "GMM", "cramers_v_group_x_subject": cramers_v(df["gmm"], df["subject_id"])},
        {"method": "HDBSCAN (incl. ruis)", "cramers_v_group_x_subject": cramers_v(df["hdb"], df["subject_id"])},
        {"method": "HDBSCAN (zonder ruis)",
         "cramers_v_group_x_subject": cramers_v(df.loc[no_noise, "hdb"], df.loc[no_noise, "subject_id"])},
    ])
    spread.round(2).to_csv(args.outdir / "participant_spread.csv", sep=";", index=False)
    cv.round(3).to_csv(args.outdir / "cramers_v_subject.csv", sep=";", index=False)

    # --- 4. Slaapstadia ------------------------------------------------------
    if "stage_rk" in df.columns:
        stage = df["stage_rk"].map(STAGE_NAMES).fillna(df["stage_rk"].astype(str))
        st_gmm = (pd.crosstab(gmm_named, stage, normalize="index") * 100)
        st_hdb = (pd.crosstab(hdb_named, stage, normalize="index") * 100).reindex(order_h)
        st = pd.concat([st_gmm, st_hdb])
        st.round(1).to_csv(args.outdir / "stage_distribution_pct.csv", sep=";")

    # --- 5. Featureprofielen ------------------------------------------------
    prof_gmm = df.groupby(gmm_named)[FEATURES].median()
    prof_hdb = df.groupby(hdb_named)[FEATURES].median().reindex(order_h)
    profiles = pd.concat([prof_gmm.add_prefix(""), prof_hdb])
    profiles.index = [f"GMM {i}" for i in prof_gmm.index] + [f"HDB {i}" for i in prof_hdb.index]
    profiles.round(3).to_csv(args.outdir / "feature_profiles_median_z.csv", sep=";")

    # --- 6. Figuur ----------------------------------------------------------
    plot_all(df, tab_row, profiles, args.outdir / "comparison_GMM_HDBSCAN.png")

    # --- Samenvatting op scherm --------------------------------------------
    pd.set_option("display.width", 160)
    print("\n=== Kruistabel (aantallen) ===")
    print(tab.to_string())
    print("\n=== Rij-% : per HDBSCAN-label, verdeling over GMM-componenten ===")
    print(tab_row.round(1).to_string())
    print("\n=== Overeenkomst ===")
    print(agree.round(3).to_string(index=False))
    print("\n=== Gemiddelde GMM-kans per HDBSCAN-label ===")
    print(post[[c for c in post.columns if c.endswith("_mean")]].round(3).to_string())
    print("\n=== Spreiding over deelnemers ===")
    print(spread.round(1).to_string(index=False))
    print("\n=== Cramér's V (groep x deelnemer; 0 = los van persoon, 1 = groep = persoon) ===")
    print(cv.round(3).to_string(index=False))
    if "stage_rk" in df.columns:
        print("\n=== Slaapstadium per groep (rij-%) ===")
        print(st.round(1).to_string())
    print("\n=== Featureprofiel (mediaan z) ===")
    print(profiles.round(2).to_string())
    print(f"\nOutput in: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()