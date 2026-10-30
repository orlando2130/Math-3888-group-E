import networkx as nx
import pandas as pd
import graph_tool.all as gt
import numpy as np

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


G = read_in_proteins("protein.links.v12.0.txt.gz", include_essential_proteins=False, essential_protein_filepath="essential_proteins.csv")
state=gt.minimize_blockmodel_dl(G)
blocks = state.get_blocks()
protein_id = G.vp["protein_id"]

communities = {}
for v in G.vertices():
    community = int(blocks[v])
    protein = protein_id[v]
    communities.setdefault(community,[]).append(protein)

# Get community assignments
blocks = state.get_blocks()
protein_id = G.vp["protein_id"]
