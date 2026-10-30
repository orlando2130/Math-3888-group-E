import networkx as nx
import pandas as pd
import graph_tool.all as gt
import numpy as np
from collections import defaultdict

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


# Get community assignments
G = read_in_proteins(
    "protein.links.v12.0.txt.gz",
    include_essential_proteins=False,
    essential_protein_filepath="essential_proteins.csv"
)

state = gt.minimize_blockmodel_dl(G)
blocks = state.get_blocks()
protein_id = G.vp["protein_id"]

# Group vertices by community
communities = {}

for v in G.vertices():
    community = int(blocks[v])
    communities.setdefault(community, []).append(v)

# Find the vertex corresponding to PEP5
pep5 = next(
    v for v in G.vertices()
    if protein_id[v] == "YMR231W"
)

# Community containing PEP5
pep5_community = int(blocks[pep5])
print("PEP5 community:", pep5_community)

# Find all communities connected to PEP5's community
neighbor_communities = set()

for v in communities[pep5_community]:
    for n in v.all_neighbors():
        community = int(blocks[n])

        if community != pep5_community:
            neighbor_communities.add(community)


# calculate within-community degree z-scores for the neighbouring communities
best_nodes = {}
for community in neighbor_communities:
    vertices = communities[community]
    # within-community degree for each node
    degrees = np.array([
        sum(1 for n in v.all_neighbors()
            if int(blocks[n])==community)
            for v in vertices
    ])
    mean_degree = degrees.mean()
    std_degree = degrees.std()

    #calculate z-scores
    if std_degree > 0:
        z_scores = (degrees - mean_degree) / std_degree
    else:
        z_scores = np.zeros(len(vertices))

    # find node with highest z-score
    best_idx = np.argmax(z_scores)
    best_vertex = vertices[best_idx]

    best_nodes[community]={
        "protein": protein_id[best_vertex],
        "z_score": z_scores[best_idx],
        "within_community_degree": degrees[best_idx]
    }

print(len(best_nodes))



# calculate betweenness centrality for the graph
vertex_bc, edge_bc = gt.betweenness(G)

# calculate shortest paths and identify highest betweenness nodes
selected_proteins = []
for community, result in best_nodes.items():
    target_protein = result["protein"]
    # find target vertex
    target_vertex = next(
        v for v in G.vertices()
        if protein_id[v] == target_protein
    )
    # find shortest path between PEP5 and best_node
    path, _ = gt.shortest_path(
        G,
        source=pep5,
        target=target_vertex
    )
    # Consider only internal nodes (exclude PEP5)
    internal_nodes = path[1:-1]
    if len(internal_nodes) == 0:
        continue

    # find node with highest betweenness centrality
    best_vertex = max(
        internal_nodes,
        key=lambda v: vertex_bc[v]
    )
    # calculate shortest distance from pep5
    distance = gt.shortest_distance(
        G,
        source=pep5,
        target=best_vertex
    )
    # exclude nodes 3 or fewer steps away from pep5
    if distance > 3:
        selected_proteins.append({
            "protein": protein_id[best_vertex],
            "community": community,
            "betweenness": vertex_bc[best_vertex],
            "distance_from_PEP5": int(distance)
        }
    )


# display results
for node in selected_proteins:
    print(
        f"Protein: {node['protein']}, "
        f"Community: {node['community']} "
        f"Betweenness: {node['betweenness']:.6f}, "
        f"Distance from PEP5: {node['distance_from_PEP5']}"
    )
