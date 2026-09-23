"""
Wk 8 Toolbox:

Reducing a network into its communities by collapsing each community into a single node.
"""

import numpy as np
import networkx as nx
from src.community_finding import eliminate_small_communities, size_distribution_of_communities

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


def reduce_network(G: nx.Graph) -> nx.Graph:
    r"""Reduce a network by condensing each of its communities into a node

    Parameters
    ----------
    G : nx.Graph
        Graph to reduce

    Returns
    -------
    G_reduced : nx.Graph
        Corresponding reduced graph

    Notes
    -----
    Weights are computed using the weights() method above, this method computes the weight
    of an edge between node i and j as the number of edges connecting community i and j divided
    by the size of i weighted

    $$
        w_{i,j} = \frac{1}{2}\frac{\sum_{k, l \text{ in } i,j} A_{k,l}}
        {\rho_i \cdot N_i + \rho_j \cdot N_j}
    $$
    """
    
    communities = eliminate_small_communities(G, threshold=3, seed=None, resolution=50)
    community_dict = {i: community for i, community in enumerate(communities, start=1)}
    
    subgraph_nodes = []
    for c in communities:
        for node in c:
            if node not in subgraph_nodes:
                subgraph_nodes.append(node)

    G_sub = nx.subgraph(G=G, nbunch=subgraph_nodes)
    w = weights(G, communities)  # dict like {(i, j): weight, ...}

    G_reduced = nx.Graph()
    G_reduced.add_nodes_from(community_dict.keys())

    for (i, j), weight in w.items():
        G_reduced.add_edge(i, j, weight=weight)

    return G_reduced
    