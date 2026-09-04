import networkx as nx
import numpy as np
import scipy as sp

"""
Toolbox wk 6: Candidate Identification

General strategy:
    1. partition network into communitites
    2. Identify the community to which target protein belongs to 
    3. Identify adjacent communities to our target's community
    4. Locate high centrality nodes in each community (say top 3)
        -> these nodes characterise our communitites
    5. Hope is that these communities share a common functionality
        -> biochem can tell us which communtities we are more interested in
    6. Compute shortest path from a (high centrality) representative 
        of our interesting community and our target (pep5)

Subroutines:
    - find all shortest paths between high-centrality node and the initial target protein
    - sort all proteins appearing on shortest paths according to how frequently they occur
    - pick top three and if the shortest path from them to target is ~4, we are sweet.
            (if the candidates are less than 4 from our target, may need to adjust method)
"""

def all_shortest_paths(G: nx.Graph, source : str, target : str = "YMR231W", method: str = 'dijkstra') -> list[list]:
    """
    Find all shortest paths between source and target in G.

    Unlike nx.shortest_path, which returns a single shortest path, this returns
    every path of minimum length between source and target.

    Parameters
    ----------
    G : nx.Graph
        The networkx Graph object to search.
    source : str
        Starting node for the path, our chosen community representative.
    target : str = "YMR231W"
        Ending node for the path. Default is our target (pep5)
    method : str = 'dijkstra'
        The algorithm to use to compute the path: 'dijkstra' or 'bellman-ford'.

    Returns
    -------
    list[list]
        A list of shortest paths, where each path is a list of nodes from
        source to target, inclusive.

    Notes
    -----
    Unlike nx.shortest_path, this function requires both source and target to be
    specified — "all shortest paths" isn't well-defined when either is left as
    None (i.e., when computing paths to/from every other node in the graph).
    """
    return list(nx.all_shortest_paths(G, source=source, target=target, method=method))

def sort_proteins_by_frequency(paths: list[list], descending: bool = True) -> list[tuple]:
    """
    Sort all proteins appearing on a collection of shortest paths according to how
    frequently they occur.

    Parameters
    ----------
    paths : list[list]
        A list of paths, where each path is a list of nodes (proteins), as returned
        by all_shortest_paths(G, source, target).
    descending : bool = True
        descending=True (default) sorts from most frequent to least frequent.
        To do ascending order, set descending=False.

    Returns
    -------
    list[tuple]
        A list of (protein, count) tuples, sorted by count.

    Notes
    -----
        - A protein's count includes every occurrence across all paths, so a protein
            appearing multiple times within the same path is counted each time.
        - source and target nodes are included in the count, since they appear in
            every path passed in.
    """
    counts = {}
    for path in paths:
        for node in path:
            if node not in counts:
                counts[node] = 0
            counts[node] += 1

    return sorted(counts.items(), key=lambda item: item[1], reverse=descending)

