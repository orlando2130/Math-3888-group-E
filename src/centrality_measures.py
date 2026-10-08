import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import warnings

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
    degree = G.degree(target) # type: ignore
    
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
    degree = dict(G.degree()) # type: ignore

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

def _unweighted_copy(G: nx.Graph) -> nx.Graph:
    """
    Returns a copy of G with all edge weights removed, so that functions which don't
    accept a `weight`/`distance` argument (and therefore implicitly use the graph's
    adjacency matrix as-is) treat every edge as weight 1.
    """
    H = G.copy()
    for _, _, data in H.edges(data=True):
        data.pop("weight", None)
    return H


def _eigenvector_centrality_per_component(G: nx.Graph,
                                           weight: str | None,
                                           method: str,
                                           max_iter: int,
                                           tol: float) -> dict:
    """
    Computes eigenvector centrality separately per connected component and merges
    the results into a single dict keyed by node.

    Necessary because eigenvector centrality is only mathematically well-defined
    (unique, meaningful) on a connected graph, per Perron-Frobenius theory. Running
    it on a disconnected graph directly can fail to converge, or silently return
    values that only reflect one dominant component while being meaningless
    (near-zero or arbitrary) for the rest.

    Note: values are NOT comparable across different components, since each
    component's eigenvector centrality is normalised independently.
    """
    result = {}
    for component_nodes in nx.connected_components(G):
        H = G.subgraph(component_nodes)
        if H.number_of_edges() == 0:
            # single isolated node or edgeless component: centrality is undefined/zero
            result.update({n: 0.0 for n in H.nodes})
            continue
        if method == 'power':
            try:
                sub_result = nx.eigenvector_centrality(H, weight=weight, max_iter=max_iter, tol=tol)
            except nx.PowerIterationFailedConvergence:
                sub_result = nx.eigenvector_centrality_numpy(H, weight=weight)
        else:
            sub_result = nx.eigenvector_centrality_numpy(H, weight=weight)
        result.update(sub_result)
    return result


def all_node_centrality_measures_weighted(G: nx.Graph,
                                 *,
                                 weight: str | None = "weight",
                                 eigenvector_method: str = "power",
                                 eigenvector_max_iter: int = 1000,
                                 eigenvector_tol: float = 1e-6,
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
    weight : str | None = "weight"
        Name of the edge attribute to use as weight. If None, all measures that support
        weighting are computed as if the graph were unweighted (i.e. every edge weight = 1).
        Applies to degree, eigenvector, katz, pagerank, betweenness, and closeness.
        nx.subgraph_centrality has no weight parameter of its own, so regardless of this
        argument's value it is always computed on an unweighted copy of G (see
        _unweighted_copy) to keep its behaviour consistent and predictable.
    eigenvector_method : str = "power" | "numpy"
        sets the method for computing eigenvector_centrality. If "numpy" will use nx.eigenvector_centrality_numpy()
        to find exact solution. If "power" will use nx.eigenvector_centrality() to approximate solution.
    eigenvector_max_iter : int = 1000
        max_iter passed to nx.eigenvector_centrality() when eigenvector_method == "power".
        Weighted graphs often need more than networkx's default of 100 to converge.
    eigenvector_tol : float = 1e-6
        tol passed to nx.eigenvector_centrality() when eigenvector_method == "power".
        Loosening this (e.g. to 1e-4) can help convergence at the cost of precision.
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
        - closeness_centrality treats weight as a *distance*: smaller weight = "closer" nodes.
            If your weights represent connection strength (bigger = stronger/closer), consider
            passing a graph with an inverted-weight attribute for that measure specifically.
        - subgraph_centrality is always computed on an unweighted copy of G, since networkx
            gives it no way to ignore edge weights otherwise.
        - If G is disconnected, eigenvector centrality is computed separately per connected
            component (see _eigenvector_centrality_per_component) since it is only well-defined
            on a connected graph. Values are not comparable across different components in
            that case. A RuntimeWarning is raised when this fallback is used.
    """
    #-- compute centrality measures for every node at once --#
    degree = dict(G.degree(weight=weight)) # type: ignore

    if nx.is_connected(G):
        if eigenvector_method == 'power':
            try:
                eigenvector = nx.eigenvector_centrality(
                    G, weight=weight, max_iter=eigenvector_max_iter, tol=eigenvector_tol
                )
            except nx.PowerIterationFailedConvergence:
                warnings.warn(
                    "eigenvector_centrality power iteration failed to converge "
                    f"within {eigenvector_max_iter} iterations; falling back to "
                    "eigenvector_centrality_numpy() for an exact solution instead.",
                    RuntimeWarning,
                )
                eigenvector = nx.eigenvector_centrality_numpy(G, weight=weight)
        elif eigenvector_method == 'numpy':
            eigenvector = nx.eigenvector_centrality_numpy(G, weight=weight)
        else:
            raise ValueError(f"eigenvector_method must be 'numpy' or 'power'. Not {eigenvector_method}")
    else:
        warnings.warn(
            "G is not connected: eigenvector centrality is computed separately per "
            "connected component, and values are not comparable across components.",
            RuntimeWarning,
        )
        eigenvector = _eigenvector_centrality_per_component(
            G, weight, eigenvector_method, eigenvector_max_iter, eigenvector_tol
        )

    if katz_alpha is None:
        katz_alpha = _get_alpha(G)

    if katz_method == 'power':
        katz = nx.katz_centrality(G, alpha=katz_alpha, weight=weight)
    elif katz_method == 'numpy':
        katz = nx.katz_centrality_numpy(G, alpha=katz_alpha, weight=weight)
    else:
        raise ValueError(f"katz_method must be 'numpy' or 'power'. Not {katz_method}")
    
    page_rank = nx.pagerank(G, weight=weight)
    betweenness = nx.betweenness_centrality(G, k=betweenness_k, weight=weight)
    subgraph = nx.subgraph_centrality(_unweighted_copy(G))
    closeness = nx.closeness_centrality(G, distance=weight)
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