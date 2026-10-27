
"""
Evaluate PPI-network clustering results against the CYC2008 gold standard.
 
Metrics reported for each clustering file
-----------------------------------------
* Precision / Recall / F-measure at an overlap threshold
    - a predicted cluster "matches" a reference complex when their overlap
      score >= threshold (default: neighbourhood affinity NA >= 0.2)
    - precision = matched predicted clusters / predicted clusters
    - recall    = matched reference complexes / reference complexes
* MMR  (maximum matching ratio): one-to-one best matching
       of complexes to clusters (Hungarian algorithm), summed overlap divided
       by the number of reference complexes.
* Sn / PPV / Acc : sensitivity, positive predictive
       value, and their geometric mean (accuracy).
* Number of clusters, mean cluster size.
 
Usage
-----
 
  Optional: --network edges.txt   (restrict gold standard to proteins in the network)
            --min-ref-size 3  --min-pred-size 3  --threshold 0.2
            --score na|jaccard  --sweep  --out results.csv
 
Cluster files: plain text, ONE CLUSTER PER LINE, protein IDs separated by
whitespace, commas, tabs or semicolons. Lines starting with '#' are ignored.
IDs must use the same namespace as the CYC2008 ORF column (e.g. YAL001C).
 
Requirements: pip install pandas numpy scipy xlrd openpyxl
     
"""

import argparse
import math
import os
import re
import sys
 
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment

def norm_id(x):
    """Normalise a protein identifier."""
    return str(x).strip().upper()
 
 
def _find_col(columns, keywords):
    for kw in keywords:
        for c in columns:
            if kw in str(c).strip().lower():
                return c
    return None

def load_cyc2008(path, sheet=0, protein_col=None, complex_col=None):
    """
    Read CYC2008  and return {complex_name: set(protein_ids)}.
 
    The official file has one row per (protein, complex) pair, with columns SGD_id, ORF, Name, Complex, PubMed_id, Method, GO_id, GO_term, Jaccard.
    Column names are auto-detected; override with --protein-col / --complex-col.
    """
    ext = os.path.splitext(path)[1].lower()
    df = None
    for header_row in range(0, 6):  
        if ext in (".csv", ".txt", ".tsv", ".tab"):
            sep = "," if ext == ".csv" else "\t"
            df = pd.read_csv(path, sep=sep, header=header_row, dtype=str)
        else:
            df = pd.read_excel(path, sheet_name=sheet, header=header_row, dtype=str)
        p = protein_col or _find_col(df.columns, ["orf", "systematic", "protein", "gene"])
        c = complex_col or _find_col(df.columns, ["complex"])
        if p in df.columns and c in df.columns:
            protein_col, complex_col = p, c
            break
    else:
        sys.exit(
            "Could not detect the protein/complex columns. Columns found: "
            f"{list(df.columns)}\nUse --protein-col and --complex-col."
        )
 
    df = df[[protein_col, complex_col]].dropna()
    complexes = {}
    for prot, cplx in zip(df[protein_col], df[complex_col]):
        complexes.setdefault(str(cplx).strip(), set()).add(norm_id(prot))
    print(f"[info] CYC2008: {len(complexes)} complexes read "
          f"(protein column='{protein_col}', complex column='{complex_col}')")
    return complexes

def load_network_proteins(path):
    """Return the set of proteins appearing in an edge list (first two columns)."""
    prots = set()
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = re.split(r"[\s,;]+", line.strip())
            prots.update(norm_id(p) for p in parts[:2])
    print(f"[info] network: {len(prots)} proteins")
    return prots

def load_clusters(path):
    """One cluster per line -> list of sets."""
    clusters = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            members = {norm_id(p) for p in re.split(r"[\s,;]+", line) if p}
            if members:
                clusters.append(members)
    return clusters

def filter_by_size(groups, min_size):
    return [g for g in groups if len(g) >= min_size]

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def membership_matrix(groups, index):
    rows, cols = [], []
    for i, g in enumerate(groups):
        for p in g:
            rows.append(i)
            cols.append(index[p])
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(groups), len(index)))


def overlap_matrices(ref, pred):
    """Return (intersection, ref_sizes, pred_sizes) with shape (n_ref, n_pred)."""
    universe = sorted(set().union(*ref) | set().union(*pred))
    index = {p: i for i, p in enumerate(universe)}
    R = membership_matrix(ref, index)
    C = membership_matrix(pred, index)
    inter = (R @ C.T).toarray()
    rs = np.asarray(R.sum(axis=1)).ravel()
    cs = np.asarray(C.sum(axis=1)).ravel()
    return inter, rs, cs


def score_matrix(inter, rs, cs, kind="na"):
    if kind == "jaccard":
        union = rs[:, None] + cs[None, :] - inter
        return inter / union
    # neighbourhood affinity: |A∩B|^2 / (|A||B|)
    return (inter ** 2) / (rs[:, None] * cs[None, :])


def evaluate(ref, pred, threshold=0.2, kind="na"):
    """Compute all metrics for one clustering. ref/pred: lists of sets."""
    if not pred:
        return {k: float("nan") for k in
                ["clusters", "mean_size", "precision", "recall", "f_measure",
                 "mmr", "sn", "ppv", "acc"]}

    inter, rs, cs = overlap_matrices(ref, pred)
    S = score_matrix(inter, rs, cs, kind)
    match = S >= threshold

    precision = match.any(axis=0).mean()
    recall = match.any(axis=1).mean()
    f_measure = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0.0)

    # Maximum matching ratio (one-to-one assignment)
    r_idx, c_idx = linear_sum_assignment(S, maximize=True)
    mmr = S[r_idx, c_idx].sum() / len(ref)

    # Sensitivity / PPV / Accuracy
    sn = inter.max(axis=1).sum() / rs.sum()
    col_tot = inter.sum(axis=0)
    ppv = inter.max(axis=0).sum() / col_tot.sum() if col_tot.sum() > 0 else 0.0
    acc = math.sqrt(sn * ppv)

    return {
        "clusters": len(pred),
        "mean_size": float(np.mean([len(c) for c in pred])),
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "mmr": mmr,
        "sn": sn,
        "ppv": ppv,
        "acc": acc,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complexes", required=True, help="CYC2008 file (.xls/.xlsx/.csv/.tsv)")
    ap.add_argument("--clusters", required=True, nargs="+",
                    help="one or more cluster files (one cluster per line)")
    ap.add_argument("--network", help="optional edge list; restricts reference "
                                      "complexes to proteins present in your network")
    ap.add_argument("--sheet", default=0, help="sheet name/index for Excel files")
    ap.add_argument("--protein-col", help="override protein/ORF column name")
    ap.add_argument("--complex-col", help="override complex column name")
    ap.add_argument("--min-ref-size", type=int, default=3,
                    help="minimum size of reference complexes (default 3)")
    ap.add_argument("--min-pred-size", type=int, default=3,
                    help="minimum size of predicted clusters (default 3)")
    ap.add_argument("--threshold", type=float, default=0.2,
                    help="match threshold (default 0.2)")
    ap.add_argument("--score", choices=["na", "jaccard"], default="na",
                    help="overlap score used for matching (default: na)")
    ap.add_argument("--sweep", action="store_true",
                    help="also report precision/recall/F at thresholds 0.1..0.9")
    ap.add_argument("--out", help="write the results table to this CSV file")
    args = ap.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet

    # --- reference complexes
    complexes = load_cyc2008(args.complexes, sheet, args.protein_col, args.complex_col)
    ref = list(complexes.values())
    if args.network:
        net = load_network_proteins(args.network)
        ref = [c & net for c in ref]
    ref = filter_by_size(ref, args.min_ref_size)
    if not ref:
        sys.exit("No reference complexes left after filtering.")
    print(f"[info] {len(ref)} reference complexes after filtering "
          f"(size >= {args.min_ref_size})\n")

    # --- evaluate each clustering
    rows = []
    for path in args.clusters:
        name = os.path.splitext(os.path.basename(path))[0]
        pred = filter_by_size(load_clusters(path), args.min_pred_size)
        res = evaluate(ref, pred, args.threshold, args.score)
        res = {"algorithm": name, **res}
        rows.append(res)

        if args.sweep:
            print(f"Threshold sweep for {name} ({args.score}):")
            print(f"  {'thr':>5} {'prec':>7} {'rec':>7} {'F':>7}")
            for t in np.arange(0.1, 1.0, 0.1):
                r = evaluate(ref, pred, float(t), args.score)
                print(f"  {t:5.1f} {r['precision']:7.3f} {r['recall']:7.3f} "
                      f"{r['f_measure']:7.3f}")
            print()

    table = pd.DataFrame(rows).set_index("algorithm")
    table = table[["clusters", "mean_size", "precision", "recall", "f_measure",
                   "mmr", "sn", "ppv", "acc"]]
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 140):
        print(f"Results (threshold={args.threshold}, score={args.score}):")
        print(table)

    if args.out:
        table.to_csv(args.out)
        print(f"\n[info] results saved to {args.out}")


if __name__ == "__main__":
    main()