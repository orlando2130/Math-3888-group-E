import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp

"""
Toolbox wk 4: Centrality Measures

    - node degree
    - eigenvector centrality
    - katz centrality
    - page_rank
    - betweenness centrality
    - subgraph centrality
    - closeness centrality

    NOTES:
        - I read in Newman that atteniation factor for Katz centrality is better closer to the inverse of the leading
            eigenvalue, but this wasn't very well justified.
"""


def node_centrality_measures(G: nx.Graph, target: str) -> dict[str, float | int]:
    """
    Given a target protein, computes the centrality measures in toolbox 4

    Parameters
    ----------
    G : nx.Graph
        networkx graph to compute centrality measures on
    target : str
        systematic name of target protein
    
    Returns
    -------
    centrality_dict : dict[str, float | int]
        centrality dictionary holding the result of all centrality measures. Of the form
            {"degree" : degree,
            "eigenvector" : eigenvector,
            "katz" : katz,
            "page rank" : page_rank,
            "betweenness" : betweenness,
            "subgraph" : subgraph,
            "closeness" : closeness}

    Notes
    -----
        - This method should not be used for more than one node, it is terribly innefficient. Should take about the same time to 
            compute these for the entire graph and just put them all in a dictionary, will implement this method at some point TODO
        - I have not tested how the functions actually operate, just clicked the first function in the networkx 
            library that matches the name I am looking for. Some different algorithms to choose from in there
    """
    # check that protein is actually in our network
    graph_proteins = list(G.nodes)
    if not target in graph_proteins:
        raise ValueError(f"Protein '{target}' could not be found in the graph. Please enter a valid systematic name")

    #-- compute centrality measures --#
    degree = G.degree(target)
    
    # NOTE may want to play around with parameters in this
    eigenvector = nx.eigenvector_centrality(G)[target]
    
    # NOTE may want to play around with parameters in this
    katz = nx.katz_centrality(G)[target]

    page_rank = nx.pagerank(G)[target]
    betweenness = nx.betweenness_centrality(G)[target]
    subgraph = nx.subgraph_centrality(G)[target]
    closeness = nx.closeness_centrality(G, u=target)

    centrality_dict = {
        "degree" : degree,
        "eigenvector" : eigenvector,
        "katz" : katz,
        "page rank" : page_rank,
        "betweenness" : betweenness,
        "subgraph" : subgraph,
        "closeness" : closeness
    }
    return centrality_dict


def _get_alpha(G: nx.Graph, safety_factor: float = 0.9) -> float:
    """
    Automatically computes a safe alpha for katz_centrality based on the
    largest eigenvalue of the adjacency matrix (alpha must be < 1/lambda_max
    for the Katz centrality solution to converge / stay positive).

    Note: computing the full eigenvalue spectrum can be slow on large graphs.
    """
    eigenvalues = nx.adjacency_spectrum(G)
    lambda_max = max(eigenvalues.real)
    return safety_factor / lambda_max


def all_node_centrality_measures(G: nx.Graph,
                                 *,
                                 eigenvector_method: str = "power",
                                 katz_alpha: float | None = 1e-4,
                                 katz_method: str = "power",
                                 betweenness_k: int | None = None
                                 ) -> dict[str, dict[str, float | int]]:
    """
    Computes the centrality measures in toolbox 4 for every node in the graph.

    Parameters
    ----------
    G : nx.Graph
        networkx graph to compute centrality measures on
    eigenvector_method : str = "power" | "numpy"
        sets the method for computing katz_centrality. If "numpy" will use nx.eigenvector_centrality_numpy()
        to find exact solution. If "power" will use nx.eigenvector_centrality() to approximate solution.
    katz_alpha : float | None = 1e-4
        sets alpha used in nx.katz_centrality(). alpha must be less than the leading eigenvalue of 
        the adjacency matrix. If None, will call _get_alpha() method to compute automatically.
    katz_method : str =  "power" | "numpy"
        sets the method for computing katz_centrality. If "numpy" will use nx.katz_centrality_numpy()
        to find exact solution. If "power" will use nx.katz_centrality() to approximate solution.
    betweenness_k : int | None = None
        Sets the number of sampled nodes as sources for the considered paths. Networkx makes the
        appropriate adjustments. k closer to len(G) will result in a more accurate approximation
    
    
    Returns
    -------
    centrality_dict : dict[str, dict[str, float | int]]
        dictionary keyed by node systematic name, each value a dictionary holding the
        result of all centrality measures for that node. Of the form
            {node : {"degree" : degree,
                     "eigenvector" : eigenvector,
                     "katz" : katz,
                     "page rank" : page_rank,
                     "betweenness" : betweenness,
                     "subgraph" : subgraph,
                     "closeness" : closeness}}

    Notes
    -----
        - Unlike node_centrality_measures, this computes each centrality measure ONCE for
            the whole graph rather than once per node, which is why it's efficient to call
            this instead of looping node_centrality_measures over every node.
        - I have not tested how the functions actually operate, just clicked the first function
            in the networkx library that matches the name I am looking for. Some different
            algorithms to choose from in there
        - Automatically computing eigenvalues for alpha in katz centrality can take a long time!
    """
    #-- compute centrality measures for every node at once --#
    degree = dict(G.degree())

    if eigenvector_method == 'power':
        eigenvector = nx.eigenvector_centrality(G)
    elif eigenvector_method == 'numpy':
        eigenvector = nx.eigenvector_centrality_numpy(G)
    else:
        raise ValueError(f"eigenvector_method must be 'numpy' or 'power'. Not {katz_method}")
    
    if katz_alpha is None:
        katz_alpha = _get_alpha(G)

    if katz_method == 'power':
        katz = nx.katz_centrality(G, alpha=katz_alpha)
    elif katz_method == 'numpy':
        katz = nx.katz_centrality_numpy(G, alpha=katz_alpha)
    else:
        raise ValueError(f"katz_method must be 'numpy' or 'power'. Not {katz_method}")
    
    page_rank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(G, k=betweenness_k)
    subgraph = nx.subgraph_centrality(G)
    closeness = nx.closeness_centrality(G)
    centrality_dict = {
        node: {
            "degree": degree[node],
            "eigenvector": eigenvector[node],
            "katz": katz[node],
            "page rank": page_rank[node],
            "betweenness": betweenness[node],
            "subgraph": subgraph[node],
            "closeness": closeness[node],
        }
        for node in G.nodes
    }
    return centrality_dict
