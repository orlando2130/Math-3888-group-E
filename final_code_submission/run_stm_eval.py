
"""
Run the STM algorithm on the yeast STRING network and evaluate it against CYC2008.

Useful options:
    --thresholds 0.05 0.1 0.15 0.3   STM merge thresholds to try
    --min-score 750                  STRING combined_score cut-off
    --exclude-essential              drop essential proteins from the network
    --largest-cc                     use only the largest connected component
"""

import argparse
import os
import sys
import time
from pathlib import Path

import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (anchored to the project, so the working directory doesn't matter)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # .../evaluation
ROOT = HERE.parent                                 # project root

DEFAULT_NETWORK = HERE / "protein.links.v12.0.txt.gz"
DEFAULT_ESSENTIAL = HERE / "EssentialProteins_YeastMine_cerevisiae.csv"
DEFAULT_COMPLEXES = HERE / "CYC2008.xls"
DEFAULT_OUTDIR = HERE / "stm_output"

import stm                                                   # noqa: E402
from evaluation import (load_cyc2008, filter_by_size,        # noqa: E402
                        evaluate, norm_id)
from reading_in import read_in_proteins                  # noqa: E402


def save_clusters(clusters, path):
    with open(path, "w") as fh:
        for c in sorted(clusters, key=len, reverse=True):
            fh.write(" ".join(sorted(c)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default=str(DEFAULT_NETWORK),
                    help="STRING protein.links file (default: data/raw/protein.links.v12.0.txt.gz)")
    ap.add_argument("--complexes", default=str(DEFAULT_COMPLEXES),
                    help="CYC2008 file (default: evaluation/CYC2008.xls)")
    ap.add_argument("--min-score", type=float, default=750,
                    help="STRING combined_score cut-off passed to read_in_proteins (default 750)")
    ap.add_argument("--exclude-essential", action="store_true",
                    help="remove essential proteins from the network")
    ap.add_argument("--essential-file", default=str(DEFAULT_ESSENTIAL))
    ap.add_argument("--largest-cc", action="store_true",
                    help="restrict the network to its largest connected component")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.35, 0.4, 0.45, 0.5],
                    help="STM merge thresholds to try (default: 0.05 0.15 0.3)")
    ap.add_argument("--match-threshold", type=float, default=0.2,
                    help="overlap threshold for precision/recall (default 0.2)")
    ap.add_argument("--score", choices=["na", "jaccard"], default="na")
    ap.add_argument("--min-ref-size", type=int, default=3)
    ap.add_argument("--min-pred-size", type=int, default=3)
    ap.add_argument("--no-restrict", action="store_true",
                    help="do NOT restrict reference complexes to proteins in the network")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--protein-col")
    ap.add_argument("--complex-col")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="folder for cluster files")
    ap.add_argument("--out", help="CSV file for the results table")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- network (your read_in_proteins, given absolute paths)
    G = read_in_proteins(args.network,
                         probability_threshold=args.min_score,
                         include_essential_proteins=False,
                         essential_protein_filepath=args.essential_file,
                         connected=True)


    # make IDs match the normalised CYC2008 IDs (upper-case, stripped)
    G = nx.relabel_nodes(G, {n: norm_id(n) for n in G.nodes()})
    print(f"[info] network: {G.number_of_nodes()} proteins, "
          f"{G.number_of_edges()} interactions (score >= {args.min_score})")

    # ---- gold standard
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    ref = list(load_cyc2008(args.complexes, sheet, args.protein_col,
                            args.complex_col).values())
    if not args.no_restrict:
        nodes = set(G.nodes())
        ref = [c & nodes for c in ref]
    ref = filter_by_size(ref, args.min_ref_size)
    print(f"[info] {len(ref)} reference complexes (size >= {args.min_ref_size})\n")
    if not ref:
        raise SystemExit("No reference complexes overlap your network. "
                         "Check that protein IDs use the same namespace (ORF names).")

    # ---- STM phases 1 and 2 (run once, reused for every threshold)
    t0 = time.time()
    print("Phase 1: computing signals (can be slow / memory hungry on large networks)...")
    signals = stm.compute_ppi_signals(G)
    print(f"  done in {time.time() - t0:.1f}s ({len(signals):,} signal entries)")
    print("Phase 2: building pre-clusters...")
    pre = stm.build_preclusters(G, signals)
    del signals
    print(f"  {len(pre)} pre-clusters\n")

    rows = []

    def record(name, clusters):
        clusters = [set(c) for c in clusters]
        save_clusters(clusters, os.path.join(args.outdir, f"{name}.txt"))
        pred = filter_by_size(clusters, args.min_pred_size)
        res = evaluate(ref, pred, args.match_threshold, args.score)
        try:
            res["modularity_Q"] = nx.community.modularity(G, clusters)
        except Exception:
            res["modularity_Q"] = float("nan")
        res["largest_cluster"] = max((len(c) for c in clusters), default=0)
        rows.append({"algorithm": name, **res})

    # baseline: pre-clusters alone (shows what the merge phase adds)
    record("stm_preclusters", pre.values())

    # ---- phase 3 for each threshold
    for thr in args.thresholds:
        print(f"Phase 3: merging with threshold {thr} ...")
        record(f"stm_t{thr}", stm.merge_clusters(G, pre, threshold=thr))

    # ---- report
    table = pd.DataFrame(rows).set_index("algorithm")
    cols = ["clusters", "mean_size", "largest_cluster", "precision", "recall",
            "f_measure", "mmr", "sn", "ppv", "acc", "modularity_Q"]
    table = table[cols]
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 200, "display.max_columns", 20):
        print(f"\nResults (match threshold={args.match_threshold}, score={args.score}; "
              f"clusters smaller than {args.min_pred_size} ignored in the metrics):")
        print(table)
    if args.out:
        table.to_csv(args.out)
        print(f"\n[info] table saved to {args.out}")
    print(f"[info] cluster files written to {args.outdir}/")


if __name__ == "__main__":
    main()