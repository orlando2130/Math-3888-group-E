import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pprint
import gzip
from pathlib import Path

while "src" not in os.listdir():
    os.chdir('..')

from src.reading_in import read_in_proteins

def prepare_expression_matrix(expression_df: pd.DataFrame) -> pd.DataFrame:
    """Numeric expression matrix, with protein IDs in the index and samples in columns."""
    expr = expression_df.copy()
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=1, how="all")
    expr = expr.groupby(level=0).mean()
    return expr


def expression_correlations(expression_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlations; requires at least 3 paired observations."""
    expr = prepare_expression_matrix(expression_df)
    return expr.T.corr(method="pearson", min_periods=3)


def _open_text(path):
    """Open a plain-text or .gz file as UTF-8 text."""
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def parse_go_obo(obo_path: str, aspect: str = "P") -> dict:
    """Parse GO terms from go-basic.obo and return child -> weighted parents.

    aspect: 'P' (biological process), 'F' (molecular function), or
            'C' (cellular component). Only terms in the requested aspect are
            retained. Parent relations are is_a (weight 0.8) and part_of (0.6).
    """
    aspect_to_namespace = {
        "P": "biological_process",
        "F": "molecular_function",
        "C": "cellular_component",
    }
    if aspect not in aspect_to_namespace:
        raise ValueError("aspect must be 'P', 'F', or 'C'")

    terms = {}
    current = None
    in_term = False
    with _open_text(obo_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                if in_term and current and current["id"]:
                    terms[current["id"]] = current
                current = {"id": None, "namespace": None, "parents": [], "obsolete": False}
                in_term = True
                continue
            if line.startswith("[") and line.endswith("]"):
                if in_term and current and current["id"]:
                    terms[current["id"]] = current
                current = None
                in_term = False
                continue
            if not in_term or not line or line.startswith("!"):
                continue
            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line[len("namespace: "):].strip()
            elif line.startswith("is_a: "):
                parent_id = line[len("is_a: "):].split()[0]
                current["parents"].append((parent_id, 0.8))
            elif line.startswith("relationship: part_of "):
                parent_id = line[len("relationship: part_of "):].split()[0]
                current["parents"].append((parent_id, 0.6))
            elif line == "is_obsolete: true":
                current["obsolete"] = True
        if in_term and current and current["id"]:
            terms[current["id"]] = current

    allowed = {
        go_id: term for go_id, term in terms.items()
        if term["namespace"] == aspect_to_namespace[aspect] and not term["obsolete"]
    }
    parents = {
        go_id: [(parent, weight) for parent, weight in term["parents"] if parent in allowed]
        for go_id, term in allowed.items()
    }
    return parents


def load_gaf_annotations(gaf_path: str, graph_nodes, aspect: str = "P",
                         include_evidence=None) -> dict:
    """Read GAF annotations and map them to graph nodes using ID/symbol/synonyms.

    Returns {graph_node: set(GO_ID)}. GAF aspect codes are P/F/C. Rows with
    the NOT qualifier are excluded. `include_evidence`, if supplied, is a set
    of evidence codes to retain (for example {'IDA', 'IMP'}); default keeps all.

    If a synonym maps to more than one graph node, that alias is treated as
    ambiguous and not used. Check the printed match summary before proceeding.
    """
    graph_nodes = {str(node) for node in graph_nodes}
    alias_to_nodes = {}
    for node in graph_nodes:
        alias_to_nodes.setdefault(node, set()).add(node)

    records = []
    with _open_text(gaf_path) as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith("!"):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            db_object_id = fields[1].strip()
            symbol = fields[2].strip()
            qualifier = fields[3].strip()
            go_id = fields[4].strip()
            evidence = fields[6].strip()
            row_aspect = fields[8].strip()
            synonyms = fields[10].split("|") if len(fields) > 10 and fields[10] else []
            if row_aspect != aspect or "NOT" in qualifier.split("|"):
                continue
            if include_evidence is not None and evidence not in include_evidence:
                continue
            aliases = {db_object_id, symbol, *[s.strip() for s in synonyms if s.strip()]}
            records.append((aliases, go_id))
            for alias in aliases:
                if alias in graph_nodes:
                    alias_to_nodes.setdefault(alias, set()).add(alias)

    # Build candidate mapping from any GAF alias to graph IDs. Only graph-node
    # names are used as matching aliases, avoiding accidental symbol-only remaps.
    alias_to_go = {node: set() for node in graph_nodes}
    matched_rows = 0
    for aliases, go_id in records:
        matches = aliases.intersection(graph_nodes)
        if len(matches) == 1:
            alias_to_go[next(iter(matches))].add(go_id)
            matched_rows += 1
        # If multiple graph IDs occur in one alias set, do not guess which one.

    matched = sum(bool(go_terms) for go_terms in alias_to_go.values())
    print(f"GAF annotation rows retained for aspect {aspect}: {len(records):,}")
    print(f"Graph nodes with at least one matched GO annotation: {matched:,}/{len(graph_nodes):,}")
    print(f"Matched annotation rows: {matched_rows:,}")
    return alias_to_go


def wang_s_values(go_id: str, go_parents: dict, cache=None) -> dict:
    """Return Wang S-values for a GO term and its ancestors.

    For each ancestor, retain the maximum product of relation weights along
    any path from the annotated term. Uses 0.8 for is_a and 0.6 for part_of.
    """
    if cache is not None and go_id in cache:
        return cache[go_id]
    if go_id not in go_parents:
        result = {}
        if cache is not None:
            cache[go_id] = result
        return result

    values = {go_id: 1.0}
    stack = [go_id]
    while stack:
        child = stack.pop()
        child_value = values[child]
        for parent, weight in go_parents.get(child, []):
            proposed = child_value * weight
            if proposed > values.get(parent, 0.0):
                values[parent] = proposed
                stack.append(parent)
    if cache is not None:
        cache[go_id] = values
    return values


def wang_go_term_similarity(go_a: str, go_b: str, go_parents: dict, s_value_cache=None) -> float:
    """Wang et al. (2007) semantic similarity between two GO terms."""
    if go_a == go_b:
        return 1.0 if go_a in go_parents else np.nan
    s_a = wang_s_values(go_a, go_parents, s_value_cache)
    s_b = wang_s_values(go_b, go_parents, s_value_cache)
    if not s_a or not s_b:
        return np.nan
    common = set(s_a).intersection(s_b)
    denominator = sum(s_a.values()) + sum(s_b.values())
    if denominator == 0:
        return np.nan
    return sum(s_a[t] + s_b[t] for t in common) / denominator


def wang_protein_similarity(go_terms_a, go_terms_b, go_parents: dict,
                            term_similarity_cache=None, s_value_cache=None) -> float:
    """Wang protein similarity using symmetric best-match averaging.

    For each term assigned to protein A, take its maximum similarity to any
    term assigned to B; do the reciprocal calculation and divide the combined
    sum by |A| + |B|. This is the set-wise aggregation described by Wang et al.
    Empty/unusable annotation sets return NaN rather than zero.
    """
    terms_a = sorted(set(go_terms_a).intersection(go_parents))
    terms_b = sorted(set(go_terms_b).intersection(go_parents))
    if not terms_a or not terms_b:
        return np.nan

    def term_sim(a, b):
        key = (a, b) if a <= b else (b, a)
        if term_similarity_cache is not None and key in term_similarity_cache:
            return term_similarity_cache[key]
        value = wang_go_term_similarity(a, b, go_parents, s_value_cache)
        if term_similarity_cache is not None:
            term_similarity_cache[key] = value
        return value

    a_to_b = []
    for a in terms_a:
        vals = [term_sim(a, b) for b in terms_b]
        vals = [v for v in vals if pd.notna(v)]
        if vals:
            a_to_b.append(max(vals))
    b_to_a = []
    for b in terms_b:
        vals = [term_sim(a, b) for a in terms_a]
        vals = [v for v in vals if pd.notna(v)]
        if vals:
            b_to_a.append(max(vals))
    if not a_to_b or not b_to_a:
        return np.nan
    return (sum(a_to_b) + sum(b_to_a)) / (len(terms_a) + len(terms_b))


def calculate_wang_edge_similarities(G: nx.Graph, protein_go_terms: dict) -> dict:
    """Calculate GO similarities only for graph edges; keys are frozenset pairs."""
    go_parents = protein_go_terms["go_parents"]
    annotations = protein_go_terms["annotations"]
    term_similarity_cache = {}
    s_value_cache = {}
    edge_similarities = {}
    for u, v in G.edges:
        edge_similarities[frozenset((u, v))] = wang_protein_similarity(
            annotations.get(str(u), set()), annotations.get(str(v), set()),
            go_parents, term_similarity_cache, s_value_cache
        )
    valid = sum(pd.notna(value) for value in edge_similarities.values())
    print(f"Calculated Wang GO similarities for {valid:,}/{G.number_of_edges():,} graph edges")
    return edge_similarities

def edge_clustering_coefficient_teo(G: nx.Graph, u, v) -> float:
    """TEO triangle-based ECC: common neighbours / min(deg(u)-1, deg(v)-1).

    An edge can only participate in triangles if both endpoints have degree >= 2.
    For edges with no possible triangle, return 0.0.
    """
    denominator = min(G.degree(u) - 1, G.degree(v) - 1)
    if denominator <= 0:
        return 0.0
    common = len(set(G.neighbors(u)).intersection(G.neighbors(v)))
    return common / denominator


def teo_scores(
    G: nx.Graph,
    expression_df: pd.DataFrame,
    go_similarity_df,
) -> pd.Series:
    """Calculate TEO(u) = sum_v ECC(u,v) * (GO_sim(u,v) + PCC(u,v)).

    expression_df: rows=protein IDs, columns=samples/conditions.
    go_similarity_df: either a square DataFrame or an edge dictionary keyed by frozenset({u, v}).
    """
    pcc = expression_correlations(expression_df)
    go_is_edge_dict = isinstance(go_similarity_df, dict)
    go = None if go_is_edge_dict else prepare_go_similarity(go_similarity_df)
    scores = {}

    for u in G.nodes:
        total = 0.0
        usable_edges = 0
        if u not in pcc.index or (not go_is_edge_dict and u not in go.index):
            scores[u] = np.nan
            continue

        for v in G.neighbors(u):
            if v not in pcc.index or v not in pcc.columns:
                continue
            if go_is_edge_dict:
                go_sim = go_similarity_df.get(frozenset((u, v)), np.nan)
            else:
                if v not in go.index or v not in go.columns:
                    continue
                go_sim = go.loc[u, v]
            corr = pcc.loc[u, v]
            if pd.isna(corr) or pd.isna(go_sim):
                continue
            ecc = edge_clustering_coefficient_teo(G, u, v)
            if pd.isna(ecc):
                continue
            total += float(ecc) * (float(go_sim) + float(corr))
            usable_edges += 1

        scores[u] = total if usable_edges else np.nan

    return pd.Series(scores, name="TEO").sort_values(ascending=False)



def calculate_TEO(G, expression_filepath, gaf_filepath, go_obo_filepath, go_aspect = "P"):
    GAF_FILEPATH = gaf_filepath
    GO_OBO_FILEPATH = go_obo_filepath
    EXPRESSION_FILEPATH = expression_filepath
    GO_ASPECT = go_aspect
    FILEPATH = "data/raw/protein.links.v12.0.txt.gz"
    G = read_in_proteins(include_essential_proteins=False, connected=True)

    # expression data
    expression_df = pd.read_csv(EXPRESSION_FILEPATH, index_col=0)
    expression_df = prepare_expression_matrix(expression_df)
    print(f"Expression profiles for {len(set(G.nodes) & set(expression_df.index)):,}/{G.number_of_nodes():,} graph nodes")

    #GO data
    go_parents = parse_go_obo(GO_OBO_FILEPATH, aspect=GO_ASPECT)
    annotations = load_gaf_annotations(GAF_FILEPATH, G.nodes, aspect=GO_ASPECT)
    protein_go_terms = {"go_parents": go_parents, "annotations": annotations}
    go_edge_similarity = calculate_wang_edge_similarities(G, protein_go_terms)

    # TEO calc
    teo = teo_scores(G, expression_df, go_edge_similarity)
    return teo
