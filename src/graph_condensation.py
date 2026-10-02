"""
Wk 8 Toolbox:

Reducing a network into its communities by collapsing each community into a single node.
"""

import numpy as np
import networkx as nx
from src.community_finding import eliminate_small_communities, size_distribution_of_communities
from wk7_toolbox.wk7_subroutines import louvain_frequencies_using_all_members
from collections import Counter
import pandas as pd

def weights(G, communities) -> dict[tuple[int, int], float]:
    """
    Given a graph and its communitites, compute the weights of edges within this community.

    Parameters
    ---------
    G : nx.Graph
        The complete graph
    communities : list[set()]
        List of our communities. Each list contains a set of nodes corresponding to a community

    Returns
    --------
    normalised_weights : dict[tuple[int, int] : float]
        The weights for our edges. Each edge (i, j) is given a weight. Weights are computed as the
        number of edges between community i and j, divided by the number of nodes in community i 
        multiplied by its density plus the number of nodes in community j multiplied by its density.

    Notes
    -----
    The actual method for finding weights is something we should have a play around with.
    """
    sizes = np.array(size_distribution_of_communities(communities))
    n = len(communities)
    
    # precompute node -> community index lookup
    node_to_comm = {}
    for i, c in enumerate(communities):
        for node in c:
            node_to_comm[node] = i
    
    # precompute density per community
    densities = np.zeros(n)
    for i, c in enumerate(communities):
        subG = G.subgraph(c)
        densities[i] = nx.density(subG)
    
    edge_weights = {}
    
    for u, v in G.edges():
        ci, cj = node_to_comm[u], node_to_comm[v]
        if ci == cj:
            continue  # skip internal edges, only care about inter-community
        
        pair = tuple(sorted((ci, cj)))
        edge_weights[pair] = edge_weights.get(pair, 0) + 1
    
    # normalize each pair's raw connection count by density and size of both communities
    normalised_weights = {}
    for (i, j), count in edge_weights.items():
        denom = (densities[i] * sizes[i]) + (densities[j] * sizes[j])
        normalised_weights[(i, j)] = count / denom if denom > 0 else 0
    
    return normalised_weights

def reduce_network(
    G: nx.Graph,
    freqs_df: pd.DataFrame,
    *,
    collapse_drifters: bool = False,
    drifter_threshold: float = 0.8,
) -> nx.Graph:
    r"""Reduce a network by condensing each community into a node.

    Parameters
    ----------
    G : nx.Graph
        Graph to reduce.

    freqs_df : pd.DataFrame
        Output of `louvain_frequencies_using_high_degree_proteins`.
        Must contain the columns:
            - "protein"
            - "dominant_frequency"

    collapse_drifters : bool, default=False
        If False, drifter proteins are kept as individual nodes in the
        reduced graph. If True, drifters are collapsed into their
        respective communities like all other proteins.

    drifter_threshold : float, default=0.8
        Maximum dominant-community frequency for a protein to be considered
        a drifter.

        For example, with the default value:
            frequency > 0.8  -> not a drifter
            frequency <= 0.8 -> drifter

    Returns
    -------
    G_reduced : nx.Graph
        Reduced graph. Community nodes are labelled by integer community IDs.
        If `collapse_drifters=False`, drifter proteins remain as individual
        nodes labelled by their protein name.

    Notes
    -----
    Community-community edge weights are calculated using `weights()`.

    Edges involving retained drifter proteins use the sum of the original
    edge weights between the drifter and the target community/drifter.

    The reduction therefore looks conceptually like:

        Original:

            A--B--C
             \ | /
               D

            E--F--G

            C = drifter

        Reduced:

              1
             / \
            C   2

        where C remains an individual node rather than being absorbed
        into community 1.
    """

    if not 0 <= drifter_threshold <= 1:
        raise ValueError(
            "`drifter_threshold` must be between 0 and 1."
        )

    communities = eliminate_small_communities(
        G,
        threshold=3,
        seed=None,
        resolution=50,
    )

    # Identify drifters
    freq_lookup = (
        freqs_df
        .set_index("protein")["dominant_frequency"]
        .to_dict()
    )

    drifters = set()

    if not collapse_drifters:
        for community in communities:
            for protein in community:

                freq = freq_lookup.get(protein)

                if freq is not None and freq <= drifter_threshold:
                    drifters.add(protein)

    # Remove drifters from their communities before collapsing.
    working_communities = [
        community - drifters
        for community in communities
    ]

    # Remove any communities that became empty after removing drifters.
    working_communities = [
        community
        for community in working_communities
        if len(community) > 0
    ]

    # Give each surviving community an integer label.
    community_dict = {
        i: community
        for i, community in enumerate(
            working_communities,
            start=1,
        )
    }
    
    # Construct the subgraph containing only proteins that belong to a collapsed community.
    community_nodes = (
        set().union(*working_communities)
        if working_communities
        else set()
    )

    G_sub = G.subgraph(community_nodes)

    community_weights = weights(
        G_sub,
        working_communities,
    )

    # Create the reduced graph.
    G_reduced = nx.Graph()

    # Add community nodes.
    G_reduced.add_nodes_from(community_dict.keys())

    # Add edges between communities.
    for (i, j), weight in community_weights.items():
        G_reduced.add_edge(
            i,
            j,
            weight=weight,
        )

    # Map every retained protein to the node it represents in the reduced graph.
    node_to_label = {}

    for label, community in community_dict.items():
        for protein in community:
            node_to_label[protein] = label

    # Drifters retain their original identity.
    for drifter in drifters:
        node_to_label[drifter] = drifter

    # Add drifter nodes.
    G_reduced.add_nodes_from(drifters)

    # Add edges involving drifters.
    for drifter in drifters:

        edge_weights = Counter()

        for neighbor in G.neighbors(drifter):

            # Ignore self-loops.
            if neighbor == drifter:
                continue

            # Find the reduced node corresponding to this neighbour.
            target = node_to_label.get(neighbor)

            # Ignore proteins that were removed by
            # eliminate_small_communities().
            if target is None:
                continue

            # Preserve the original edge weight.
            edge_weights[target] += G[drifter][neighbor].get(
                "weight",
                1,
            )

        # Add the accumulated edges to the reduced graph.
        for target, total_weight in edge_weights.items():

            if G_reduced.has_edge(drifter, target):

                # This can happen if an edge between these nodes was
                # already created elsewhere.
                G_reduced[drifter][target]["weight"] += total_weight

            else:

                G_reduced.add_edge(
                    drifter,
                    target,
                    weight=total_weight,
                )

    return G_reduced