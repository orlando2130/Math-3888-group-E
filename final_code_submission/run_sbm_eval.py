
"""
Run the SBM algorithm on the STRING network and evaluate it against CYC2008.

Useful options:
    --min-score 750                  STRING combined_score cut-off
    --exclude-essential              drop essential proteins from the network
    --largest-cc                     use only the largest connected component
"""

import argparse
import os
import time
import sys
import numpy as np
import pandas as pd
import graph_tool.all as gt

from pathlib import Path
# ---------------------------------------------------------------------------
# Paths (anchored to the project, so the working directory doesn't matter)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # .../evaluation
ROOT = HERE.parent                                 # project root

DEFAULT_NETWORK = HERE / "protein.links.v12.0.txt.gz"
DEFAULT_ESSENTIAL = HERE / "EssentialProteins_YeastMine_cerevisiae.csv"
DEFAULT_COMPLEXES = HERE / "CYC2008.xls"
DEFAULT_OUTDIR = HERE / "sbm_output"



from evaluation import (load_cyc2008, filter_by_size,        
                        evaluate, norm_id)
def _load_essential_proteins(essential_protein_filepath: str) -> pd.DataFrame:
    df = pd.read_csv(essential_protein_filepath)
    df.columns = ['counter', 'systematic name', 'organism', 'standard name', 'description']
    return df

def read_in_proteins(
    filepath: str,
    *,
    probability_threshold: int | float = 750,
    include_essential_proteins: bool = True,
    essential_protein_filepath: str | None = None,
    connected: bool = False,
) -> gt.Graph:
    """
    Read protein links into a graph-tool Graph.

    Returns
    -------
    g : graph_tool.Graph
        An undirected graph with vertex property 'protein_id'
        and edge property 'combined_score'.
    """

    # Read and filter edge list
    df = pd.read_csv(
        filepath,
        sep=" ",
        names=["protein1", "protein2", "combined_score"],
        header=0,
    )

    df = df[df["combined_score"] >= probability_threshold].copy()

    # Remove STRING organism prefix
    df["protein1"] = df["protein1"].str.replace(
        "4932.", "", regex=False
    )
    df["protein2"] = df["protein2"].str.replace(
        "4932.", "", regex=False
    )

    # Exclude essential proteins
    if not include_essential_proteins:
        if essential_protein_filepath is None:
            raise ValueError(
                "If include_essential_proteins is False, "
                "must give essential_protein_filepath"
            )

        df_essential = _load_essential_proteins(
            essential_protein_filepath
        )
        essential_set = set(df_essential["systematic name"])

        mask = (
            ~df["protein1"].isin(essential_set)
            & ~df["protein2"].isin(essential_set)
        )
        df = df.loc[mask].copy()

    # Map protein IDs to consecutive integer vertex indices
    proteins = pd.Index(
        pd.unique(df[["protein1", "protein2"]].to_numpy().ravel())
    )
    protein_to_idx = {
        protein: i for i, protein in enumerate(proteins)
    }

    src = df["protein1"].map(protein_to_idx).to_numpy()
    dst = df["protein2"].map(protein_to_idx).to_numpy()

    # Build graph-tool graph directly
    g = gt.Graph(directed=False)
    g.add_vertex(len(proteins))

    protein_id = g.new_vertex_property("string")
    for v, protein in zip(g.vertices(), proteins):
        protein_id[v] = str(protein)
    g.vp["protein_id"] = protein_id
    
    combined_score = g.new_edge_property("double")
    g.add_edge_list(
        __import__("numpy").column_stack((src, dst)),
        hashed=False,
    )

    # Populate edge scores in the same order as inserted edges
    for e, score in zip(g.edges(), df["combined_score"].to_numpy()):
        combined_score[e] = score

    g.ep["combined_score"] = combined_score

    # Keep only the largest connected component if requested
    if connected and g.num_vertices() > 0:
        comp, hist = gt.label_components(g)
        largest_component = int(hist.argmax())
        keep = comp.a == largest_component
        g = gt.GraphView(g, vfilt=keep).copy()

    return g
 

def fit_sbm(g, model, deg_corr, n_init, seed):
    """Run the SBM inference n_init times and keep the lowest-entropy fit."""
    gt.seed_rng(seed)
    np.random.seed(seed)
    best, best_S = None, np.inf
    for i in range(n_init):
        t0 = time.time()
        if model == "nested":
            state = gt.minimize_nested_blockmodel_dl(
                g, state_args=dict(deg_corr=deg_corr))
        else:
            state = gt.minimize_blockmodel_dl(
                g, state_args=dict(deg_corr=deg_corr))
        S = state.entropy()
        print(f"  run {i + 1}/{n_init}: description length = {S:.1f} "
              f"({time.time() - t0:.1f}s)")
        if S < best_S:
            best, best_S = state, S
    return best, best_S
 
 
def blocks_to_clusters(blocks_arr, ids):
    clusters = {}
    for pid, b in zip(ids, blocks_arr):
        clusters.setdefault(int(b), set()).add(pid)
    return list(clusters.values())
 

def save_clusters(clusters, path):
    with open(path, "w") as fh:
        for c in sorted(clusters, key=len, reverse=True):
            fh.write(" ".join(sorted(c)) + "\n")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default=str(DEFAULT_NETWORK),
                    help="STRING protein.links file")
    ap.add_argument("--complexes", default=str(DEFAULT_COMPLEXES),
                    help="CYC2008 file")
    ap.add_argument("--min-score", type=float, default=750,
                    help="STRING combined_score cut-off (default 750)")
    ap.add_argument("--include-essential", action="store_true",
                    help="keep essential proteins (default: they are removed, "
                         "as in the STM run)")
    ap.add_argument("--essential-file", default=str(DEFAULT_ESSENTIAL))
    ap.add_argument("--largest-cc", action="store_true",
                    help="restrict the network to its largest connected component")
    ap.add_argument("--model", choices=["flat", "nested"], default="flat",
                    help="flat: minimize_blockmodel_dl; nested: hierarchical "
                         "(every level is evaluated)")
    ap.add_argument("--no-deg-corr", action="store_true",
                    help="use the non-degree-corrected SBM (default: degree-corrected)")
    ap.add_argument("--n-init", type=int, default=5,
                    help="independent inference runs; lowest description length "
                         "is kept (default 5)")
    ap.add_argument("--seed", type=int, default=42)
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

    # ---- network (graph-tool)
    g = read_in_proteins(args.network,
                         probability_threshold=args.min_score,
                         include_essential_proteins=args.include_essential,
                         essential_protein_filepath=args.essential_file,
                         connected=args.largest_cc)
    pid_prop = g.vp["protein_id"]
    ids = [norm_id(pid_prop[v]) for v in g.vertices()]   # vertex-index order
    print(f"[info] network: {g.num_vertices()} proteins, "
          f"{g.num_edges()} interactions (score >= {args.min_score})")

    # ---- gold standard
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    ref = list(load_cyc2008(args.complexes, sheet, args.protein_col,
                            args.complex_col).values())
    if not args.no_restrict:
        nodes = set(ids)
        ref = [c & nodes for c in ref]
    ref = filter_by_size(ref, args.min_ref_size)
    print(f"[info] {len(ref)} reference complexes (size >= {args.min_ref_size})\n")
    if not ref:
        raise SystemExit("No reference complexes overlap your network. "
                         "Check that protein IDs use the same namespace (ORF names).")

    # ---- SBM inference
    deg_corr = not args.no_deg_corr
    print(f"Fitting {args.model} SBM (degree-corrected={deg_corr}), "
          f"{args.n_init} run(s)...")
    t0 = time.time()
    state, mdl = fit_sbm(g, args.model, deg_corr, args.n_init, args.seed)
    print(f"  done in {time.time() - t0:.1f}s, best description length = {mdl:.1f}\n")

    rows = []

    def record(name, blocks):
        """blocks: graph-tool vertex property map of block labels."""
        clusters = blocks_to_clusters(blocks.a, ids)
        save_clusters(clusters, os.path.join(args.outdir, f"{name}.txt"))
        pred = filter_by_size(clusters, args.min_pred_size)
        res = evaluate(ref, pred, args.match_threshold, args.score)
        try:
            res["modularity_Q"] = gt.modularity(g, blocks)
        except Exception:
            res["modularity_Q"] = float("nan")
        res["largest_cluster"] = max((len(c) for c in clusters), default=0)
        res["n_blocks"] = len(clusters)
        res["mdl"] = mdl
        rows.append({"algorithm": name, **res})

    if args.model == "nested":
        for lvl in range(len(state.get_levels())):
            b = state.project_level(lvl).get_blocks()
            n_blocks = len(set(b.a))
            if n_blocks <= 1:        # top of hierarchy: everything in one block
                continue
            print(f"Evaluating level {lvl} ({n_blocks} blocks)")
            record(f"sbm_nested_L{lvl}", b)
    else:
        print(f"Evaluating flat partition ({state.get_B()} blocks)")
        record("sbm_flat", state.get_blocks())

    # ---- report
    table = pd.DataFrame(rows).set_index("algorithm")
    cols = ["clusters", "n_blocks", "mean_size", "largest_cluster", "precision",
            "recall", "f_measure", "mmr", "sn", "ppv", "acc", "modularity_Q", "mdl"]
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