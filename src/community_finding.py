import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

"""
Toolbox wk 6: Community Finding

    - partiton a given connected graph into communities
    - sort the communitites by size, show the number of communities and their size distribution
    - eliminate communities below a specified size, record the number of communities and their size distributio
"""


def partition_graph(x: nx.Graph, seed: int | None = 42) -> list[set]:
    """
    Partition a given connected graph into its communitites using Louvain Community Detection Algorithm.

    Parameters
    ----------
    x : nx.Graph
        A connected networkx Graph object to partition.
    seed : int | None = 42
        A seed to control RNG in Louvain algorithm. If None, use the global RNG for the function's preferred package:
        https://networkx.org/documentation/stable/reference/randomness.html#randomness

    Returns
    -------
    final_communitites : list[set]
        A list of sets of nodes, each set holding the nodes belonging to one community.
    """

    final_communities = []
    communities = nx.community.louvain_communities(x, seed=seed)
    sorted_communities = sorted(communities, key = len)  
    for community in sorted_communities:
        if len(community)>1:
            final_communities.append(community)
    return final_communities

def partition_graph_size(x: nx.Graph, seed: int | None = 42) -> list[tuple[set, int]]:
    """
    Partition a given connected graph into its communities using the Louvain Community
    Detection Algorithm, returning each community alongside its size.

    Parameters
    ----------
    x : nx.Graph
        A connected networkx Graph object to partition.
    seed : int | None = 42
        A seed to control RNG in Louvain algorithm. If None, use the global RNG for the function's preferred package:
        https://networkx.org/documentation/stable/reference/randomness.html#randomness

    Returns
    -------
    list[tuple[set, int]]
        A list of (community, size) tuples, where each community is a set of nodes
        with more than one member.
    """
    final_communities = []
    communities = nx.community.louvain_communities(x, seed=seed)
    sorted_communities = sorted(communities, key=len)
    for community in sorted_communities:
        if len(community) > 1:
            final_communities.append(community)
    return [(community, len(community)) for community in final_communities]


def find_adjacent_communities(G: nx.Graph, communities: list[set], target_protien) -> list[set]:
    """
    Find all communities adjacent to a given target protein's community.

    A community is considered adjacent if any of the target protein's neighbors
    belong to it, excluding the target protein's own community.

    Parameters
    ----------
    G : nx.Graph
        The full networkx Graph containing the target protein and its neighbors.
    communities : list[set]
        A list of communities (sets of nodes), as produced by partition_graph(x).
    target_protien
        The node (protein) whose adjacent communities should be found.

    Returns
    -------
    list[set]
        A list of communities (sets of nodes) adjacent to the target protein's community.
    """
    neighbors = G.neighbors(target_protien)
    protien_communities = {}
    adjecent_communities = []
    for comunity in communities:
        for node in comunity:
            protien_communities[node] = comunity
    for node in neighbors:
        if protien_communities[node] not in adjecent_communities and protien_communities[node] != protien_communities[target_protien]:
            adjecent_communities.append(protien_communities[node])

    return adjecent_communities



def eliminate_small_communities(G: nx.Graph, threshold: int) -> list[set]:
    """
    Partition a graph into communities using the Louvain Community Detection Algorithm
    and remove any communities smaller than a given threshold.

    Parameters
    ----------
    G : nx.Graph
        A connected networkx Graph object to partition.
    threshold : int
        The minimum community size to keep. Communities with fewer nodes than this
        are discarded.

    Returns
    -------
    list[set]
        A list of communities (sets of nodes) with size >= threshold, sorted from
        smallest to largest.
    """
    communities_list = nx.algorithms.community.louvain_communities(G)
    communities_list.sort(key=len)
    filtered_list = [x for x in communities_list if len(x) >= threshold]
    return filtered_list




def size_distribution_of_communities(communities_list: list[set]) -> list[int]:
    """
    gets size of community for each community in the list

    Parameters
    ----------
    communitites_list: : list[set]
        list of communitites in our network
    
    Returns
    -------
    list of int
        list of sizes of communitites in communities_list
    """
    return [len(c) for c in communities_list] 

def sort_communities_by_size(communities_list: list[set], reverse: bool = True) -> list[set]:
    """
    sort communitites by size (duh)

    Parameters
    ----------
    communities_list: list[set]
        list of communitites to sort
    reverse: bool = True
        reverse=True (default) sorts from biggest to smallest. To do ascending order, set reverse=False.

    Notes
    -----
        - This function does not modify the input list. Normally, .sort() returns none and directly modifies a list, 
            hence why we copy before sorting it.
        - key=len sorts the communites by the size of the set, i.e., how many nodes there are.
    """

    copy = communities_list.copy()
    copy.sort(reverse=reverse, key=len)
    return copy

def histogram_of_community_size(communities_list: list[set]) -> None:
    """
    Plot a histogram showing the distribution of community sizes.

    Parameters
    ----------
    communities_list : list[set]
        Collection of communities, where each community is a set of nodes.
        The size of a community is the number of nodes it contains.

    Returns
    -------
    None
        Displays a histogram of the community sizes.

    Notes
    -----
    The histogram is constructed by first determining the number of nodes
    in each community and then plotting the resulting distribution.
    """

    Size_list = size_distribution_of_communities(communities_list)
    # plots Histogram
    counts, bins = np.histogram(Size_list)
    plt.stairs(counts, bins)
    plt.plot()
    return None