#!/usr/bin/env python3
"""
Evaluate the two Louvain-stability methods against CYC2008.

Run:

    python evaluation/run_louvain_eval.py
    python evaluation/run_louvain_eval.py --runs 100 --core-t 0.9 --resolution 1 5 10

What is evaluated
-----------------
Each method runs Louvain `--runs` times and reports, for every protein, its
dominant community and how often it landed there. Those results are turned into
clusters in two ways:
    *_all   every protein, grouped by its dominant community
    *_core  only "core" proteins (dominant_frequency >= --core-t), grouped by
            their dominant community
and each clustering is scored against CYC2008 (precision, recall, F, MMR,
Sn/PPV/Acc). Baselines: a single Louvain run, and the average over all runs.
"""

import argparse
import contextlib
import importlib
import io
import os
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT))

import load_cyc2008, filter_by_size, evaluate, norm_id   # noqa: E402
from reading_in import read_in_proteins                              # noqa: E402

DEFAULT_NETWORK = ROOT / "data" / "raw" / "protein.links.v12.0.txt.gz"
DEFAULT_ESSENTIAL = ROOT / "data" / "raw" / "EssentialProteins_YeastMine_cerevisiae.csv"
DEFAULT_COMPLEXES = HERE / "CYC2008.xls"
DEFAULT_OUTDIR = HERE / "louvain_output"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def partition_to_clusters(partition):
    """{node: label} -> list of sets."""
    groups = {}
    for node, label in partition.items():
        groups.setdefault(label, set()).add(node)
    return list(groups.values())


def clusters_from_df(df, core_only, core_t):
    sub = df[df["dominant_frequency"] >= core_t] if core_only else df
    return [set(g["protein"]) for _, g in sub.groupby("dominant_community")]


def adjusted_rand_index(a, b):
    """ARI between two {node: label} partitions over the same nodes."""
    nodes = list(a)
    comb2 = lambda x: x * (x - 1) / 2
    cont = Counter((a[n], b[n]) for n in nodes)
    ca = Counter(a[n] for n in nodes)
    cb = Counter(b[n] for n in nodes)
    sum_ij = sum(comb2(v) for v in cont.values())
    sum_a = sum(comb2(v) for v in ca.values())
    sum_b = sum(comb2(v) for v in cb.values())
    expected = sum_a * sum_b / comb2(len(nodes))
    maximum = (sum_a + sum_b) / 2
    return 1.0 if maximum == expected else (sum_ij - expected) / (maximum - expected)


def save_clusters(clusters, path):
    with open(path, "w") as fh:
        for c in sorted(clusters, key=len, reverse=True):
            fh.write(" ".join(sorted(c)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--louvain-module", default="louvain_methods",
                    help="importable module holding your two methods (default: louvain_methods)")
    ap.add_argument("--network", default=str(DEFAULT_NETWORK))
    ap.add_argument("--complexes", default=str(DEFAULT_COMPLEXES))
    ap.add_argument("--min-score", type=float, default=750)
    ap.add_argument("--exclude-essential", action="store_true")
    ap.add_argument("--essential-file", default=str(DEFAULT_ESSENTIAL))
    ap.add_argument("--largest-cc", action="store_true")
    # Louvain / stability settings
    ap.add_argument("--runs", type=int, default=100, help="number of Louvain runs (default 100)")
    ap.add_argument("--resolution", type=float, nargs="+", default=[125, 175],
                    help="Louvain resolution(s). Higher = smaller communities; default 1.0. "
                         "Several values run the whole evaluation once per value.")
    ap.add_argument("--core-t", type=float, default=0.9,
                    help="a protein is 'core' if its dominant frequency >= this (default 0.9)")
    ap.add_argument("--drifter-t", type=float, default=0.05, help="method 1 drifter threshold")
    ap.add_argument("--top-n", type=int, default=10,
                    help="method 2: signature size (high-degree proteins per community)")
    ap.add_argument("--match-t", type=float, default=0.3,
                    help="method 2: Jaccard threshold for matching communities across runs")
    # evaluation settings
    ap.add_argument("--match-threshold", type=float, default=0.2)
    ap.add_argument("--score", choices=["na", "jaccard"], default="na")
    ap.add_argument("--min-ref-size", type=int, default=3)
    ap.add_argument("--min-pred-size", type=int, default=3)
    ap.add_argument("--no-restrict", action="store_true")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--protein-col")
    ap.add_argument("--complex-col")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--out", help="CSV file for the results table")
    args = ap.parse_args()
    
    for name in ("network", "complexes", "essential_file", "outdir"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if args.out:
        args.out = str(Path(args.out).resolve())
    os.makedirs(args.outdir, exist_ok=True)

    # ---- network
    G = read_in_proteins(args.network,
                         probability_threshold=args.min_score,
                         include_essential_proteins=not args.exclude_essential,
                         essential_protein_filepath=args.essential_file,
                         connected=args.largest_cc)
    G = nx.relabel_nodes(G, {n: norm_id(n) for n in G.nodes()})
    print(f"[info] network: {G.number_of_nodes()} proteins, {G.number_of_edges()} interactions")

    # ---- reference complexes
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    ref = list(load_cyc2008(args.complexes, sheet, args.protein_col, args.complex_col).values())
    if not args.no_restrict:
        nodes = set(G.nodes())
        ref = [c & nodes for c in ref]
    ref = filter_by_size(ref, args.min_ref_size)
    ref = list(ref) 
    print(f"[info] {len(ref)} reference complexes (size >= {args.min_ref_size})")
    if not ref:
        raise SystemExit("No reference complexes overlap your network (check ID namespace).")

    # Calculate and print the average size
    avg_size = sum(len(c) for c in ref) / len(ref)
    print(f"[info] Average size of reference complexes: {avg_size:.2f}")
    
    # ---- your methods
    mod = importlib.import_module(args.louvain_module)

    # Both methods call run_louvain_n_times(G, n) with the same seeds, so cache it:
    # Louvain is then run once per resolution instead of twice.
    orig_run = mod.run_louvain_n_times
    cache = {}

    def cached_run(G_, n):
        key = (id(G_), n, current_resolution[0])
        if key not in cache:
            cache[key] = orig_run(G_, n)
        return cache[key]

    mod.run_louvain_n_times = cached_run

   
    current_resolution = [1.0]
    orig_louvain = nx.community.louvain_communities

    def louvain_with_resolution(G_, weight=None, seed=None, **kw):
        kw.setdefault("resolution", current_resolution[0])
        return orig_louvain(G_, weight=weight, seed=seed, **kw)

    nx.community.louvain_communities = louvain_with_resolution

    all_tables = []
    for res in args.resolution:
        current_resolution[0] = res
        tag = f"res{res:g}"
        print(f"\n{'=' * 70}\nLouvain resolution {res:g}: {args.runs} runs\n{'=' * 70}")
        t0 = time.time()
        partitions = mod.run_louvain_n_times(G, args.runs)          # progress printed once
        print(f"  Louvain done in {time.time() - t0:.0f}s")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res1 = mod.louvain_frequencies_using_all_members(
                G, args.runs, args.core_t, args.drifter_t)
        df1 = pd.DataFrame(res1)
        print(f"  method 1 done ({buf.getvalue().strip()})")

        with contextlib.redirect_stdout(io.StringIO()):               # it prints every community
            df2 = mod.louvain_frequencies_using_high_degree_proteins(
                G, args.runs, args.top_n, args.match_t)
        n_labels2 = len({lab for d in df2["community_frequencies"] for lab in d})
        print(f"  method 2 done ({n_labels2} labelled communities)")

        df1.to_csv(os.path.join(args.outdir, f"method1_{tag}.csv"), index=False)
        df2.to_csv(os.path.join(args.outdir, f"method2_{tag}.csv"), index=False)

        rows = []

        def add_row(name, clusters, stability=None, n_labels=None):
            clusters = [set(c) for c in clusters]
            save_clusters(clusters, os.path.join(args.outdir, f"{name}_{tag}.txt"))
            pred = filter_by_size(clusters, args.min_pred_size)
            r = evaluate(ref, pred, args.match_threshold, args.score)
            try:
                r["modularity_Q"] = nx.community.modularity(G, clusters)
            except Exception:                       # core-only rows are not full partitions
                r["modularity_Q"] = float("nan")
            r["largest_cluster"] = max((len(c) for c in clusters), default=0)
            r["coverage"] = sum(len(c) for c in pred) / G.number_of_nodes()
            r["stability"] = stability if stability is not None else float("nan")
            rows.append({"resolution": res, "clustering": name, **r})

        # baselines
        add_row("louvain_run0", partition_to_clusters(partitions[0]))
        per_run = [evaluate(ref, filter_by_size(partition_to_clusters(p), args.min_pred_size),
                            args.match_threshold, args.score) for p in partitions]
        mean_run = {k: float(np.nanmean([r[k] for r in per_run])) for k in per_run[0]}
        sd_f = float(np.nanstd([r["f_measure"] for r in per_run]))
        rows.append({"resolution": res, "clustering": "louvain_mean_of_runs", **mean_run})

        # the two methods
        for label, df in (("method1", df1), ("method2", df2)):
            for core_only in (False, True):
                name = f"{label}_{'core' if core_only else 'all'}"
                sub = df[df["dominant_frequency"] >= args.core_t] if core_only else df
                add_row(name, clusters_from_df(df, core_only, args.core_t),
                        stability=float(sub["dominant_frequency"].mean()) if len(sub) else float("nan"))

        table = pd.DataFrame(rows).set_index(["resolution", "clustering"])
        cols = ["clusters", "mean_size", "largest_cluster", "coverage", "stability",
                "precision", "recall", "f_measure", "mmr", "sn", "ppv", "acc", "modularity_Q"]
        table = table[[c for c in cols if c in table.columns]]
        with pd.option_context("display.float_format", "{:.3f}".format,
                               "display.width", 220, "display.max_columns", 30):
            print(f"\nResults vs CYC2008 (match threshold {args.match_threshold}, score {args.score}):")
            print(table)
        print(f"(single-run F-measure varies by sd = {sd_f:.3f} across the {args.runs} runs)")

        # stability / agreement summary
        lab1 = dict(zip(df1["protein"], df1["dominant_community"]))
        lab2 = dict(zip(df2["protein"], df2["dominant_community"]))
        print("\nStability summary:")
        for label, df in (("method1", df1), ("method2", df2)):
            f = df["dominant_frequency"]
            print(f"  {label}: mean dominant frequency {f.mean():.3f}, "
                  f"core (>= {args.core_t}) {100 * (f >= args.core_t).mean():.1f}% of proteins")
        print(f"  ARI method1 vs method2: {adjusted_rand_index(lab1, lab2):.3f}")
        print(f"  ARI method1 vs run0   : {adjusted_rand_index(lab1, partitions[0]):.3f}")
        print(f"  ARI method2 vs run0   : {adjusted_rand_index(lab2, partitions[0]):.3f}")
        all_tables.append(table)

    final = pd.concat(all_tables)
    if args.out:
        final.to_csv(args.out)
        print(f"\n[info] table saved to {args.out}")
    print(f"[info] per-protein results and cluster files written to {args.outdir}/")


if __name__ == "__main__":
    main()