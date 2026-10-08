"""
=============================================================================
4.1_hdbscan_plots.py

Vervolg op 4_hdbscan.py: plots maken van ÉÉN OF MEER specifieke HDBSCAN-
configuraties, volledig vanuit de terminal in te stellen. Er wordt geen grid
en geen stabiliteitscheck gedraaid -- alleen fitten + plotten + labels/summary.

Hergebruikt de functies uit 4_hdbscan.py (fitten, inladen, plotstijl), zodat
de figuren er precies hetzelfde uitzien. 4_hdbscan.py moet in dezelfde map
staan (of geef het pad mee met --hdbscan-script).

Per configuratie (submap OUTPUT_DIR/<tag>/plots/mcs<..>_ms<..>_<method>[_single]):
  hdbscan_final_labels.csv          -- label/probability/outlier_score per event
  hdbscan_final_cluster_summary.csv -- per cluster: n, %, n_subjects, grootste subject-aandeel
  fig_clusters_main.png/.pdf        -- 2 scatterplots (automatisch gekozen paren)
  fig_clusters_pairs.png/.pdf       -- corner plot (overslaan met --no-corner)
  fig_clusters_custom.png/.pdf      -- alleen met --xy: jouw eigen paren

Bij meer dan één configuratie bovendien (in OUTPUT_DIR/<tag>/plots/):
  fig_compare_<x>_<y>.png/.pdf      -- alle configuraties naast elkaar op hetzelfde
                                       assenpaar, om te laten zien hoe de uitkomst
                                       verandert met de parameters (appendix)

Gebruik (voorbeelden):
  # één configuratie
  python 4.1_hdbscan_plots.py --config 130 5 leaf

  # meerdere configuraties + vergelijkingsfiguur
  python 4.1_hdbscan_plots.py --config 130 5 leaf --config 650 10 eom --config 65 5 eom

  # ander inputbestand, single-cluster toegestaan, eigen tag
  python 4.1_hdbscan_plots.py --input "...\\pca_scores_3pc_unwhitened.csv" --tag unwhitened ^
         --allow-single-cluster --config 65 5 eom

  # eigen assenparen (mag vaker), corner plot overslaan
  python 4.1_hdbscan_plots.py --config 130 5 leaf --xy PC1 PC2 --xy PC2 PC3 --no-corner

  # bij de 8 features als input
  python 4.1_hdbscan_plots.py --input "...\\arousal_features_reduced_scaled.csv" ^
         --tag reduced_features --config 650 10 eom --xy mean_delta_ratio mean_beta_ratio
=============================================================================
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 4_hdbscan.py INLADEN (naam begint met een cijfer -> via importlib)
# =============================================================================

def load_hdbscan_module(path: Path):
    if not path.exists():
        sys.exit(f"{path} niet gevonden -- zet 4.1 in dezelfde map als 4_hdbscan.py "
                 "of geef het pad mee met --hdbscan-script.")
    spec = importlib.util.spec_from_file_location("hdbscan_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# PLOTS DIE NIET IN 4_hdbscan.py ZITTEN
# =============================================================================

def plot_custom(mod, df, pairs, out_dir: Path, config_txt: str) -> None:
    """Scatterplots op door de gebruiker gekozen assenparen (--xy)."""
    styles = mod._style_map(df["cluster"])
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.6))
    axes = np.atleast_1d(axes)
    for i, (ax, (x, y)) in enumerate(zip(axes, pairs)):
        mod._scatter_pair(ax, df, x, y, styles)
        ax.set_xlabel(mod._axis_label(x), fontsize=10, color=mod.TEXT_COLOR)
        ax.set_ylabel(mod._axis_label(y), fontsize=10, color=mod.TEXT_COLOR)
        if n > 1:
            ax.text(-0.12, 1.03, "ABCDEFGH"[i], transform=ax.transAxes, fontsize=14,
                    fontweight="bold", color=mod.TEXT_COLOR)
        mod._style_axes(ax)
    fig.legend(handles=mod._legend_handles(df["cluster"], styles), loc="lower center",
               ncol=min(4, len(styles) + 1), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.text(0.99, 0.99, f"HDBSCAN: {config_txt}", ha="right", va="top",
             fontsize=8, color=mod.MUTED_COLOR)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_clusters_custom.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figuur opgeslagen: {out_dir / 'fig_clusters_custom.png'} (+ .pdf)")


def plot_compare(mod, results: list[dict], x: str, y: str, out_dir: Path) -> None:
    """
    Alle configuraties naast elkaar (max 3 per rij) op hetzelfde assenpaar,
    met dezelfde aslimieten, zodat verschillen direct zichtbaar zijn.
    Elk paneel heeft zijn eigen legenda (clusternummers verschillen per run).
    """
    k = len(results)
    n_cols = min(3, k)
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 5.2 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()
    for i, (ax, res) in enumerate(zip(axes, results)):
        df = res["df"]
        styles = mod._style_map(df["cluster"])
        mod._scatter_pair(ax, df, x, y, styles, size=3)
        mod._style_axes(ax)
        n_cl = len(styles)
        noise = 100 * (df["cluster"] == -1).mean()
        ax.set_title(f"{res['config_txt']}\n{n_cl} cluster(s), noise {noise:.1f}%",
                     fontsize=9, color=mod.TEXT_COLOR)
        ax.text(-0.10, 1.06, "ABCDEFGHIJKL"[i], transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=mod.TEXT_COLOR)
        ax.legend(handles=mod._legend_handles(df["cluster"], styles), loc="upper right",
                  fontsize=6.5, frameon=True, framealpha=0.85, edgecolor="none")
        if i % n_cols == 0:
            ax.set_ylabel(mod._axis_label(y), fontsize=9, color=mod.TEXT_COLOR)
        if i >= k - n_cols:
            ax.set_xlabel(mod._axis_label(x), fontsize=9, color=mod.TEXT_COLOR)
    for ax in axes[k:]:
        ax.axis("off")
    fig.tight_layout()
    name = f"fig_compare_{x}_{y}"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVergelijkingsfiguur opgeslagen: {out_dir / (name + '.png')} (+ .pdf)")


# =============================================================================
# HOOFDLOOP
# =============================================================================

def parse_config(cfg: list[str]) -> tuple[int, int, str]:
    mcs, ms, method = cfg
    if method not in ("eom", "leaf"):
        raise SystemExit(f"Methode moet 'eom' of 'leaf' zijn, niet '{method}'.")
    return int(mcs), int(ms), method


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdbscan-script", type=Path, default=here / "4_hdbscan.py",
                        help="Pad naar 4_hdbscan.py (default: dezelfde map)")
    parser.add_argument("--input", type=Path, default=None,
                        help="Input-CSV (default: DEFAULT_INPUT uit 4_hdbscan.py)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: OUTPUT_DIR uit 4_hdbscan.py")
    parser.add_argument("--tag", default="whitened",
                        help="Submap (zelfde als bij 4_hdbscan.py); plots komen in <tag>/plots/")
    parser.add_argument("--config", nargs=3, action="append", required=True,
                        metavar=("MCS", "MS", "METHOD"),
                        help="min_cluster_size min_samples eom|leaf (mag vaker)")
    parser.add_argument("--allow-single-cluster", action="store_true")
    parser.add_argument("--xy", nargs=2, action="append", metavar=("X", "Y"),
                        help="Eigen assenpaar (mag vaker), bijv. --xy PC1 PC2")
    parser.add_argument("--no-corner", action="store_true",
                        help="Corner plot (alle paren) overslaan -- sneller")
    parser.add_argument("--no-compare", action="store_true",
                        help="Geen vergelijkingsfiguur bij meerdere configuraties")
    args = parser.parse_args()

    mod = load_hdbscan_module(args.hdbscan_script)
    if args.no_corner:
        mod.plot_pairs = lambda *a, **k: None      # corner plot uitschakelen

    input_path = args.input or mod.DEFAULT_INPUT
    out_root = (args.output_dir or mod.OUTPUT_DIR) / args.tag / "plots"
    out_root.mkdir(parents=True, exist_ok=True)

    df, cols = mod.load_scores(input_path)
    X = df[cols].to_numpy()

    if args.xy:
        bad = sorted({c for pair in args.xy for c in pair if c not in cols})
        if bad:
            raise SystemExit(f"--xy kolommen niet gevonden: {bad}. Beschikbaar: {cols}")
    pairs = [tuple(p) for p in args.xy] if args.xy else None

    results = []
    for cfg in args.config:
        mcs, ms, method = parse_config(cfg)
        suffix = "_single" if args.allow_single_cluster else ""
        cfg_dir = out_root / f"mcs{mcs}_ms{ms}_{method}{suffix}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 70)
        print(f"Configuratie: min_cluster_size={mcs}, min_samples={ms}, {method}"
              f"{' (allow_single_cluster)' if args.allow_single_cluster else ''}")
        print("=" * 70)

        # fit + labels + summary + standaardplots (uit 4_hdbscan.py)
        mod.final_fit(df, cols, X, mcs, ms, method, args.allow_single_cluster, cfg_dir)

        config_txt = f"min_cluster_size = {mcs}, min_samples = {ms}, {method}"
        df_lab = mod.read_csv_robust(cfg_dir / "hdbscan_final_labels.csv")
        if pairs:
            plot_custom(mod, df_lab, pairs, cfg_dir, config_txt)
        results.append({"df": df_lab,
                        "config_txt": f"mcs = {mcs}, ms = {ms}, {method}"})

    if len(results) > 1 and not args.no_compare:
        compare_pairs = pairs or [(cols[0], cols[1])]
        if not pairs and not cols[0].upper().startswith("PC"):
            first = mod.FEATURE_PLOT_PAIRS[0]
            if all(c in cols for c in first):
                compare_pairs = [first]
        for x, y in compare_pairs:
            plot_compare(mod, results, x, y, out_root)

    print(f"\nAlles opgeslagen in: {out_root}")


if __name__ == "__main__":
    main()