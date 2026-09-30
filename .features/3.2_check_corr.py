"""
=============================================================================
3.1.1_scatter_divergent.py

Visuele check van de feature-paren waar Pearson en Spearman sterk van elkaar
afwijken (output van 3.1_corr_vif.py: pearson_vs_spearman_divergent.csv).

Vraag: wordt de Pearson-r opgeblazen door losse eilandjes extreme events
LINKSONDER (beide features zeer laag, z < TAIL_Z)? Op log-schaal betekent een
z van -4 tot -8 een ratio van bijna 0: event-vermogen veel lager dan de
baseline -> verdacht voor een vervuilde baseline of signaaluitval.

Per paar:
  - Scatterplot (hexbin-dichtheid, log-schaal) van de geschaalde waarden.
  - "Staart-events" gemarkeerd: events waar BEIDE features < TAIL_Z (z-score).
  - In de titel: Pearson r, Spearman rho, en Pearson r ZONDER staart-events.
    Zakt r flink zonder de staart -> de staart drijft de correlatie.

Daarnaast een tabel met alle staart-events (over alle geplotte paren) met
metadata (subject, nacht, start_sec, duration, motion_rms), zodat je ze in het
ruwe EEG kunt terugzoeken. En een telling per subject/nacht: zitten de
staart-events verspreid, of geclusterd in een paar nachten (-> artefact of
subject-specifiek)?

Stappenplan:
  1. Geschaalde featurematrix + divergent-paren inladen.
  2. Top-N paren (op |r - rho|) kiezen, of één paar via --pair.
  3. Per paar: statistiek + scatterplot (grid + losse PNG's).
  4. Staart-events wegschrijven + telling per subject/nacht.

Gebruik:
  python 3.2_check_corr.py
  python 3.2_check_corr.py --top 9 --tail-z -4
  python 3.2_check_corr.py --pair mean_theta_ratio mean_beta_ratio
=============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr

# =============================================================================
# CONFIGURATIE
# =============================================================================

SCALED_PATH = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\2. preprocessing\scaled\arousal_feature_matrix_scaled.csv"
)

CORR_VIF_DIR = Path(
    r"C:\Users\zafar\OneDrive - Netherlands Institute for Neuroscience\Documents\THESIS_OUTPUTS\PROJECT 2\3. feature selection\corr_vif"
)
DIVERGENT_PATH = CORR_VIF_DIR / "pearson_vs_spearman_divergent.csv"

OUTPUT_DIR = CORR_VIF_DIR / "scatter_divergent"

METADATA_COLS = [
    "subject_id", "group", "night_id", "event_idx",
    "start_sec", "end_sec", "sec_prev_event",
    "stage_rk",
]
# extra kolommen die in de staart-tabel komen (als ze bestaan)
CONTEXT_COLS = ["duration_sec", "motion_rms", "oxy_amp_ratio"]

TOP_N_DEFAULT = 6   # aantal paren om te plotten
TAIL_Z = -3.0       # staart = beide features < TAIL_Z (in SD's, want z-scores)
N_COLS_GRID = 3


# =============================================================================
# SECTIE 1 — INLADEN
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Leest een CSV met automatische separator-detectie (',' of ';') en valt terug
    op decimal=',' als numerieke kolommen als tekst binnenkomen (Excel-NL).
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} bestaat niet.")
    df = pd.read_csv(path, sep=None, engine="python")
    text_cols = {"subject_id", "group", "night_id", "feature_1", "feature_2"}
    check = [c for c in df.columns if c not in text_cols]
    n_bad = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in check)
    if n_bad > 0:
        df2 = pd.read_csv(path, sep=None, engine="python", decimal=",")
        n_bad2 = sum(not pd.api.types.is_numeric_dtype(df2[c]) for c in check if c in df2.columns)
        if n_bad2 < n_bad:
            print(f"[LET OP] decimaal-komma gedetecteerd in {path.name}, opnieuw ingelezen.")
            df = df2
    return df


def choose_pairs(args, df: pd.DataFrame) -> list[tuple[str, str]]:
    """Eén paar via --pair, anders de top-N uit de divergent-CSV."""
    if args.pair:
        pairs = [tuple(args.pair)]
    else:
        div = read_csv_robust(args.divergent)
        div = div.reindex(div["diff_r_minus_rho"].abs().sort_values(ascending=False).index)
        pairs = list(zip(div["feature_1"], div["feature_2"]))[: args.top]
        print(f"Top {len(pairs)} paren uit {args.divergent.name} (grootste |r - rho|).")

    missing = {f for p in pairs for f in p if f not in df.columns}
    if missing:
        raise KeyError(f"Deze features staan niet in de featurematrix: {sorted(missing)}")
    return pairs


# =============================================================================
# SECTIE 2 — STATISTIEK PER PAAR
# =============================================================================

def pair_stats(df: pd.DataFrame, f1: str, f2: str, tail_z: float) -> dict:
    """Pearson, Spearman, en Pearson zonder staart-events voor één paar."""
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
        "tail_index": sub.index[tail],   # voor de staart-tabel, niet naar CSV
    }


# =============================================================================
# SECTIE 3 — PLOTTEN
# =============================================================================

def draw_pair(ax, df: pd.DataFrame, st: dict, tail_z: float) -> None:
    """Hexbin-dichtheid + staart-events in rood + staart-grens als stippellijn."""
    f1, f2 = st["feature_1"], st["feature_2"]
    sub = df[[f1, f2]].replace([np.inf, -np.inf], np.nan).dropna()

    hb = ax.hexbin(sub[f1], sub[f2], gridsize=60, cmap="Blues",
                   norm=LogNorm(), mincnt=1, linewidths=0)
    tail = df.loc[st["tail_index"], [f1, f2]]
    ax.scatter(tail[f1], tail[f2], s=6, c="crimson", alpha=0.6, linewidths=0,
               label=f"linksonder (beide < {tail_z} SD): n={st['n_tail']} ({st['pct_tail']}%)")
    ax.axvline(tail_z, color="crimson", ls=":", lw=0.8)
    ax.axhline(tail_z, color="crimson", ls=":", lw=0.8)

    ax.set_xlabel(f"{f1} (z)", fontsize=8)
    ax.set_ylabel(f"{f2} (z)", fontsize=8)
    ax.set_title(
        f"r = {st['pearson_r']:.2f}   rho = {st['spearman_rho']:.2f}\n"
        f"zonder staart: r = {st['pearson_r_no_tail']:.2f}   rho = {st['spearman_rho_no_tail']:.2f}",
        fontsize=8,
    )
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    cb = plt.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label("aantal events (log)", fontsize=7)
    cb.ax.tick_params(labelsize=6)


def plot_grid(df: pd.DataFrame, stats: list[dict], tail_z: float, out_path: Path) -> None:
    n = len(stats)
    n_cols = min(N_COLS_GRID, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, st in zip(axes, stats):
        draw_pair(ax, df, st, tail_z)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Paren waar Pearson en Spearman afwijken -- geschaalde features", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Grid opgeslagen: {out_path}")


def plot_single(df: pd.DataFrame, st: dict, tail_z: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.2))
    draw_pair(ax, df, st, tail_z)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# SECTIE 4 — STAART-EVENTS
# =============================================================================

def tail_events_table(df: pd.DataFrame, stats: list[dict]) -> pd.DataFrame:
    """
    Alle events die in minstens één geplot paar in de staart vallen, met
    metadata, context-kolommen en in hoeveel paren ze in de staart zitten.
    """
    counts = pd.Series(0, index=df.index)
    for st in stats:
        counts.loc[st["tail_index"]] += 1
    idx = counts[counts > 0].index

    meta = [c for c in METADATA_COLS if c in df.columns]
    ctx = [c for c in CONTEXT_COLS if c in df.columns]
    feats = sorted({f for st in stats for f in (st["feature_1"], st["feature_2"])})

    out = df.loc[idx, meta + ctx + feats].copy()
    out.insert(0, "n_pairs_in_tail", counts.loc[idx])
    return out.sort_values("n_pairs_in_tail", ascending=False)


def tail_by_group(tail: pd.DataFrame, df: pd.DataFrame, key: list[str]) -> pd.DataFrame | None:
    """Aantal staart-events per subject/nacht, t.o.v. het totaal aantal events daar."""
    if any(k not in df.columns for k in key) or tail.empty:
        return None
    total = df.groupby(key).size().rename("n_events")
    in_tail = tail.groupby(key).size().rename("n_tail_events")
    out = pd.concat([total, in_tail], axis=1).fillna(0).astype(int)
    out["pct_tail"] = (100 * out["n_tail_events"] / out["n_events"]).round(2)
    out["share_of_all_tail"] = (100 * out["n_tail_events"] / len(tail)).round(2)
    return out.sort_values("n_tail_events", ascending=False)


# =============================================================================
# HOOFDLOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SCALED_PATH,
                        help="Pad naar arousal_feature_matrix_scaled.csv")
    parser.add_argument("--divergent", type=Path, default=DIVERGENT_PATH,
                        help="Pad naar pearson_vs_spearman_divergent.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT,
                        help="Aantal paren uit de divergent-CSV")
    parser.add_argument("--pair", nargs=2, metavar=("FEATURE_1", "FEATURE_2"),
                        help="Plot alleen dit ene paar")
    parser.add_argument("--tail-z", type=float, default=TAIL_Z,
                        help="Staart-grens in SD's (default -3.0)")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stap 1+2 ---
    df = read_csv_robust(args.input)
    print(f"Featurematrix geladen: {args.input}  shape={df.shape}")
    pairs = choose_pairs(args, df)

    # --- Stap 3: statistiek + plots ---
    stats = [pair_stats(df, f1, f2, args.tail_z) for f1, f2 in pairs]
    stats_df = pd.DataFrame([{k: v for k, v in s.items() if k != "tail_index"} for s in stats])
    stats_df.to_csv(out_dir / "pair_stats.csv", index=False)
    print("\nPer paar (staart = beide features < "
          f"{args.tail_z} SD):")
    print(stats_df.to_string(index=False))

    plot_grid(df, stats, args.tail_z, out_dir / "scatter_grid.png")
    for st in stats:
        plot_single(df, st, args.tail_z,
                    out_dir / f"scatter_{st['feature_1']}__{st['feature_2']}.png")

    # --- Stap 4: staart-events ---
    tail = tail_events_table(df, stats)
    tail.to_csv(out_dir / "tail_events.csv", index=False)
    print(f"\n{len(tail)} unieke staart-event(s) over alle geplotte paren "
          f"({100 * len(tail) / len(df):.2f}% van alle events).")

    # nacht = subject + night_id, voor het geval night_id per subject opnieuw begint
    for name, key in (("subject", ["subject_id"]), ("night", ["subject_id", "night_id"])):
        g = tail_by_group(tail, df, key)
        if g is not None:
            g.to_csv(out_dir / f"tail_by_{name}.csv")
            print(f"\nStaart-events per {name} (top 10):")
            print(g.head(10).to_string())

    if not tail.empty:
        ctx = [c for c in CONTEXT_COLS if c in df.columns]
        if ctx:
            cmp = pd.DataFrame({
                "mediaan_staart": tail[ctx].median(),
                "mediaan_rest": df.drop(index=tail.index)[ctx].median(),
            }).round(3)
            print("\nContext-features: staart vs rest (mediaan, z-scores):")
            print(cmp.to_string())

    print(f"\nAlles opgeslagen in: {out_dir}")
    print("  - scatter_grid.png + scatter_<f1>__<f2>.png")
    print("  - pair_stats.csv       <- r/rho met en zonder staart")
    print("  - tail_events.csv      <- staart-events met metadata, terug te zoeken in het EEG")
    print("  - tail_by_subject.csv / tail_by_night.csv  <- geclusterd in een paar nachten? -> artefact?")


if __name__ == "__main__":
    main()