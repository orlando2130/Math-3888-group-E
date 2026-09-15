import os
if not "src" in os.listdir(): # I use this to change my directory to access the src and data folders from in here
    os.chdir("..") 
# print(os.listdir()) # this should say something like ['README.md', 'wk5_toolbox', '.git', 'wk4_toolbox']

import networkx as nx
import markov_clustering as mc

import numpy as np
import scipy as sp
import pandas as pd
from collections import Counter, defaultdict

import mygene # allow us to convert from systematic to standard

import matplotlib.pyplot as plt

from src.reading_in import read_in_proteins
from src.community_finding import partition_graph, partition_graph_size, eliminate_small_communities, get_adjacent_communities


def align_partition(partition, reference):
    reference_ids = sorted(set(reference.values()))
    partition_ids = sorted(set(partition.values()))

    # construct overlap matrix
    # overlap[i,j]=#proteins in ref community i and curr community j
    overlap = np.zeros((len(reference_ids), len(partition_ids)), dtype=int)
    ref_index = {community: i for i, community in enumerate(reference_ids)}
    part_index = {community: j for j, community in enumerate(partition_ids)}

    for protein in reference:
        r = ref_index[reference[protein]]
        p = part_index[partition[protein]]
        overlap[r,p] += 1

    row_ind, col_ind = sp.optimize.linear_sum_assignment(-overlap)

    mapping = {}
    for r, p in zip(row_ind, col_ind):
        mapping[partition_ids[p]] = reference_ids[r]
    next_id = max(reference_ids) + 1
    for community in partition_ids:
        if community not in mapping:
            mapping[community] = next_id
            next_id += 1

    aligned_partition = {protein: mapping[community] for protein, community in partition.items()}
    return aligned_partition

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


def louvain_frequencies_using_all_members(G, n, core_t, drifter_t):
    partitions = run_louvain_n_times(G, n)
    reference = partitions[0]
    reference_communities = sorted(set(reference.values()))
    print("\nNumber of communities in reference:",len(reference_communities))
    aligned_partitions = []
    for partition in partitions:
        aligned = align_partition(partition, reference)
        aligned_partitions.append(aligned)

    protein_results = []

    proteins = list(G.nodes())

    for protein in proteins:
        assignments = [ partition[protein] for partition in aligned_partitions]
        counts = Counter(assignments)

        # Convert counts to frequencies
        frequencies = { community: count / n for community, count in counts.items()}
        # Sort communities by frequency
        sorted_frequencies = sorted(
            frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )

        dominant_community = sorted_frequencies[0][0]
        dominant_frequency = sorted_frequencies[0][1]

        # Number of communities the protein was observed in
        num_communities = len(sorted_frequencies)

        is_core = (dominant_frequency >= core_t)
        is_drifter = not is_core

         #2-community drifter
        # Look at communities occurring at least 5% of the time.
        # If approximately 50/50 between two communities,
        # this is a strong candidate for a bridge/linking protein.

        significant = [
            (community, frequency)
            for community, frequency in sorted_frequencies
            if frequency >= drifter_t
        ]

        is_two_community_drifter = False

        community_a = None
        community_b = None
        frequency_a = None
        frequency_b = None

        if len(significant) == 2:
            community_a, frequency_a = significant[0]
            community_b, frequency_b = significant[1]

            # Difference between the two frequencies
            frequency_difference = abs(
                frequency_a - frequency_b
            )

            #  "roughly equal" if within 10 percentage points of each other (parameter may need to be tuned)
            if frequency_difference <= 0.10:
                is_two_community_drifter = True

        protein_results.append({

            "protein": protein,

            "dominant_community": dominant_community,

            "dominant_frequency": dominant_frequency,

            "num_communities": num_communities,

            "is_core": is_core,

            "is_drifter": is_drifter,

            "is_two_community_drifter":
                is_two_community_drifter,

            "community_A": community_a,

            "frequency_A": frequency_a,

            "community_B": community_b,

            "frequency_B": frequency_b
        })
    return protein_results

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

def to_list_of_sets(partition):
    """Accepts either a dict {node: label} or an already-list-of-sets
    partition, and always returns a list of sets."""
    if isinstance(partition, dict):
        groups = defaultdict(set)
        for node, label in partition.items():
            groups[label].add(node)
        return list(groups.values())
    return list(partition)  # already list-of-sets; just make sure it's a list

def louvain_frequencies_using_high_degree_proteins(G, n, top_n, match_t):
    raw_partitions = run_louvain_n_times(G, n)
    partitions = [to_list_of_sets(p) for p in raw_partitions] 
    print("finished louvain")
    labelled_communities = {}
    first_p = partitions[0]
    for community_id, community in enumerate(first_p):
        signature = community_signature(G, community, top_n)
        labelled_communities[community_id] = { "signature": signature, "members": community}

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

            if (best_label is not None and best_similarity >= match_t):
                label = best_label
                used_labels.add(label)
            else:
                label = next_label
                next_label += 1
                labelled_communities[label] = {"signature": signature, "members":set()}
                used_labels.add(label)

            for protein in community:
                labelled_partition[protein] = label

            labelled_communities[label]["members"].update(community)
        labelled_partitions.append(labelled_partition)
    print("\nNumber of labelled communities:", len(labelled_communities))
    for label, data in labelled_communities.items():
        signature = community_signature(G, data["members"], top_n)
        print(f"\nCommmunity {label}")
        print("High-degree proteins:", sorted(signature))

    results = []
    for protein in G.nodes():
        assignments = [partition[protein] for partition in labelled_partitions if protein in partition]
        counts = Counter(assignments)
        frequencies = { community: count/n for community, count in counts.items()}
        ranked = sorted(frequencies.items(), key=lambda x:x[1], reverse=True)
        dominant_label = ranked[0][0]
        dominant_frequency = ranked[0][1]
        results.append({
            "protein": protein,

            "dominant_community":
                dominant_label,

            "dominant_frequency":
                dominant_frequency,

            "number_of_communities":
                len(frequencies),

            "community_frequencies":
                frequencies
        })
    results_df = pd.DataFrame(results)
    return results_df


def summarise_louvain_results_method1(protein_results):
    results = pd.DataFrame(protein_results)

    results.to_csv(
        "protein_community_stability.csv",
        index=False
    )
    cores = results[
        results["is_core"]
    ]

    drifters = results[
        results["is_drifter"]
    ]

    bridge_candidates = results[
        results["is_two_community_drifter"]
    ]

    print("\n========================================")
    print("RESULTS")
    print("========================================")

    print(
        "Core proteins:",
        len(cores)
    )

    print(
        "Drifter proteins:",
        len(drifters)
    )

    print(
        "Two-community drifter candidates:",
        len(bridge_candidates)
    )

    print("\nPotential community-linking proteins:")
    print("----------------------------------------")

    bridge_candidates = bridge_candidates.sort_values(
        "frequency_A"
    )

    for _, row in bridge_candidates.iterrows():

        print(
            f"{row['protein']}: "
            f"Community {int(row['community_A'])} "
            f"{row['frequency_A']:.1%}, "
            f"Community {int(row['community_B'])} "
            f"{row['frequency_B']:.1%}"
        )

def summarise_louvain_results_method2(df):
    cores = df[df["dominant_frequency"] <= 0.99]
    print("Core proteins:", len(cores))
    drifters = df[df["dominant_frequency"]<0.99] 
    print("Number of drifters:", len(drifters))

    bridge_candidates = []

    for _, row in df.iterrows():

        frequencies = row["community_frequencies"]

        # Communities that occur at least 10% of the time
        significant = sorted(
            [
                (community, frequency)
                for community, frequency in frequencies.items()
                if frequency >= 0.10
            ],
            key=lambda x: x[1],
            reverse=True
        )

        if len(significant) == 2:

            community_a, freq_a = significant[0]
            community_b, freq_b = significant[1]

            # Approximately equal frequency
            if abs(freq_a - freq_b) <= 0.10:

                bridge_candidates.append({

                    "protein": row["protein"],

                    "community_A":
                        community_a,

                    "frequency_A":
                        freq_a,

                    "community_B":
                        community_b,

                    "frequency_B":
                        freq_b
                })


    bridge_df = pd.DataFrame(
        bridge_candidates
    )

    print("\nPotential community-linking proteins:")
    print(bridge_df)
