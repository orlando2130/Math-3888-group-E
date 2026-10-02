import networkx as nx
import markov_clustering as mc

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from src.reading_in import read_in_proteins
from src.community_finding import partition_graph, partition_graph_size, eliminate_small_communities, get_adjacent_communities

def run_louvain_n_times(G, n):
    partitions = []
    for run in range(n):
        communities = nx.community.louvain_communities(G, weight="weight", seed=run)
        partition = {}
        for community_id, community in enumerate(communities):
            for protein in community:
                partition[protein] = community_id
        partitions.append(partition)

        if (run + 1) % 50 == 0:
            print(f"Completed {run + 1}/{n} runs")
    return partitions

def to_list_of_sets(partition):
    """Accepts either a dict {node: label} or an already-list-of-sets
    partition, and always returns a list of sets."""
    if isinstance(partition, dict):
        groups = defaultdict(set)
        for node, label in partition.items():
            groups[label].add(node)
        return list(groups.values())
    return list(partition)  # already list-of-sets; just make sure it's a list

def community_signature(G, members, n):
    degree_view = G.degree()
    degrees = { node: degree_view[node] for node in members}
    ranked = sorted(degrees, key=lambda node: degrees[node], reverse=True)
    return set(ranked[:n])

def jaccard(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0

    return intersection / union

def louvain_frequencies_using_high_degree_proteins(G, n, top_n, match_t):
    """
    Estimate community stability by running Louvain multiple times and tracking
    each protein's community assignment frequency.

    Since Louvain community detection is stochastic, running it repeatedly on the
    same graph can produce different partitions. This function runs Louvain `n`
    times, uses each community's highest-degree proteins as a "signature" to match
    communities across runs (since raw community IDs aren't consistent from run to
    run), and then tracks, for each protein, how often it lands in each labelled
    community across all runs.

    Parameters
    ----------
    G : nx.Graph
        Graph to partition.
    n : int
        Number of times to run the Louvain algorithm.
    top_n : int
        Number of highest-degree proteins to use as each community's signature
        (passed to `community_signature`), used to match communities across runs.
    match_t : float
        Minimum Jaccard similarity between signatures required to match a community
        in the current run to a previously labelled community. If no existing
        community meets this threshold, the community is assigned a new label.

    Returns
    -------
    results_df : pd.DataFrame
        One row per protein in `G`, with columns:

        - ``protein`` : the protein/node.
        - ``dominant_community`` : label of the community the protein was most
          frequently assigned to across the `n` runs.
        - ``dominant_frequency`` : fraction of runs (out of `n`) in which the
          protein was assigned to its dominant community.
        - ``number_of_communities`` : number of distinct labelled communities the
          protein was assigned to across all runs.
        - ``community_frequencies`` : dict mapping each community label the
          protein was assigned to, to the fraction of runs it appeared in that
          community.

    Notes
    -----
    Community labels are only stable within a single call to this function, since
    labelling is built up incrementally as runs are processed and new communities
    are encountered.
    """
    raw_partitions = run_louvain_n_times(G, n)
    partitions = [to_list_of_sets(p) for p in raw_partitions]

    labelled_communities = {}
    first_p = partitions[0]
    for community_id, community in enumerate(first_p):
        signature = community_signature(G, community, top_n)
        labelled_communities[community_id] = {"signature": signature, "members": community}

    next_label = len(labelled_communities)
    labelled_partitions = []
    for run_number, partition in enumerate(partitions):
        if run_number == 0:
            labelled_partition = {}
            for label, data in labelled_communities.items():
                for protein in data["members"]:
                    labelled_partition[protein] = label
            labelled_partitions.append(labelled_partition)
            continue

        current_communities = []
        for community in partition:
            signature = community_signature(G, community, top_n)
            current_communities.append((community, signature))
        labelled_partition = {}
        used_labels = set()

        for community, signature in current_communities:
            best_label = None
            best_similarity = 0
            for label, data in labelled_communities.items():
                if label in used_labels:
                    continue
                similarity = jaccard(signature, data["signature"])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_label = label

            if best_label is not None and best_similarity >= match_t:
                label = best_label
                used_labels.add(label)
            else:
                label = next_label
                next_label += 1
                labelled_communities[label] = {"signature": signature, "members": set()}
                used_labels.add(label)

            for protein in community:
                labelled_partition[protein] = label

            labelled_communities[label]["members"].update(community)
        labelled_partitions.append(labelled_partition)

    results = []
    for protein in G.nodes():
        assignments = [partition[protein] for partition in labelled_partitions if protein in partition]
        counts = Counter(assignments)
        frequencies = {community: count / n for community, count in counts.items()}
        ranked = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        dominant_label = ranked[0][0]
        dominant_frequency = ranked[0][1]
        results.append({
            "protein": protein,
            "dominant_community": dominant_label,
            "dominant_frequency": dominant_frequency,
            "number_of_communities": len(frequencies),
            "community_frequencies": frequencies,
        })

    results_df = pd.DataFrame(results)
    return results_df