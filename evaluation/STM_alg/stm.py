import networkx as nx
import math
import heapq


def calculate_erlang_f(c):
    """Calculates F(c) using the Erlang CDF formula at x/b = 1."""
    if c <= 0:
        return 0.0
    series_sum = sum(1/math.factorial(k) for k in range(c))
    return 1 - (math.exp(-1)*series_sum)

def compute_ppi_signals(G):
    """
    Implements the Signal Transduction Model loop across all node pairs.
    
    Parameters:
    G (nx.Graph)
    Returns:
    dict: A dictionary mapped as {(v, w): signal_value}
    """
    signals = {}
    degrees = dict(G.degree())
    f_c_cache = {c: calculate_erlang_f(c) for c in range(50)}


    for v in G.nodes():
        d_v = G.degree(v)
        paths_from_v = nx.single_source_shortest_path(G, v)
        for w, path in paths_from_v.items():
            if v == w:
                continue
            c = len(path) - 1
            f_c = f_c_cache.get(c, 0.0)
            if f_c == 0.0:
                continue

            inter_nodes = path[1:-1]
            denom_prod = 1.0
            for inter_node in inter_nodes:
                denom_prod *= degrees[inter_node]

            signal_v_w = (d_v / denom_prod) * f_c
            signals[(v,w)] = signal_v_w
    return signals

def build_preclusters(G, network_signals):
    """
    Implements the STM Pre-clustering loop based on the highest signal scores.
    
    Parameters:
    G (nx.Graph): 
    network_signals (dict): {(v, w): signal_value} output from the previous algorithm step.
    
    Returns:
    dict: A mapping of representative nodes to their set of cluster members {representative: {node1, node2, ...}}
    """
    pre_clusters = {}

    for v in G.nodes():
        best_score = -float("inf")
        representative_w = None

        for w in G.nodes():
            if v == w:
                continue
            score = network_signals.get((w,v), 0.0)
            key = (score, G.degree(w))
            if best_key is None or key > best_key:
                best_key, representative_w = key, w
        # Handle isolated nodes with no valid signal relationships
        if representative_w is None or best_score == 0.0:
            # Self-assign to ensure the node isn't dropped entirely from the network partitions
            representative_w = v
        # Check if cluster_w already exists, create it on demand, and add node v
        if representative_w not in pre_clusters:
            pre_clusters[representative_w] = set()
        pre_clusters[representative_w].add(v)

    return pre_clusters

def calculate_interconnections(G, cluster_c, cluster_k):
    """
    Calculates the interconnection strength between cluster_c and cluster_k.
    In PPI networks, this is typically the count of edges in G that bridge 
    a node in cluster_c to a node in cluster_k.
    """
    int_count = 0
    for node_u in cluster_c:
        for node_v in cluster_k:
            if G.has_edge(node_u, node_v):
                int_count += 1
    return float(int_count)

def find_max_pair(G, cluster_list):
    """
    Finds the pair of clusters with the maximum interconnection value.
    
    Parameters:
    G (nx.Graph): The underlying PPI network.
    cluster_list (list of sets): The current list of clusters (C).
    
    Returns:
    tuple: ((index_c, index_k), max_value)
    """
    max_value = -float("inf")
    max_pair = (None, None)
    n_clusters = len(cluster_list)
    if n_clusters < 2:
        return (None, None), 0.0

    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            val = calculate_interconnections(G, cluster_list[i], cluster_list[j])
            if val > max_value:
                max_value = val
                max_pair = (i, j)
    return max_pair, max_value

def merge_clusters(G, preclusters, threshold=0.15):
    """
    DENSITY-NORMALIZED Phase 3: Merges clusters based on their Jaccard connection ratio 
    instead of raw edge counts. Prevents a giant component from taking over.
    
    Recommended threshold range: 0.05 to 0.30
    """
    cluster_lookup = {}
    current_clusters = {}
    cluster_internal_edges = {}

    # 1. Map every initial pre-cluster to a unique ID
    for cluster_id, members in enumerate(preclusters.values()):
        if not members:
            continue
        cluster_set = set(members)
        current_clusters[cluster_id] = cluster_set
        for node in cluster_set:
            cluster_lookup[node] = cluster_id
            
        # Track the inner edge footprint of this cluster
        sub_g = G.subgraph(cluster_set)
        cluster_internal_edges[cluster_id] = sub_g.number_of_edges()

    # 2. Build a local inter-cluster weight tracking map
    interconnections = {}
    for u, v in G.edges():
        c_u = cluster_lookup.get(u)
        c_v = cluster_lookup.get(v)
        
        if c_u is not None and c_v is not None and c_u != c_v:
            c_min, c_max = min(c_u, c_v), max(c_u, c_v)
            if c_min not in interconnections:
                interconnections[c_min] = {}
            interconnections[c_min][c_max] = interconnections[c_min].get(c_max, 0) + 1
            
    # 3. Push initial valid pairs onto the Max-Heap using a Jaccard ratio
    heap = []
    for c_i in interconnections:
        for c_j, shared_w in interconnections[c_i].items():
            # Total distinct edges touching either cluster
            total_footprint = cluster_internal_edges[c_i] + cluster_internal_edges[c_j] + shared_w
            jaccard_ratio = shared_w / total_footprint if total_footprint > 0 else 0.0
            
            if jaccard_ratio >= threshold:
                heapq.heappush(heap, (-jaccard_ratio, c_i, c_j))
                
    active_cluster_ids = set(current_clusters.keys())

    # 4. Agglomerative Merging Loop
    while heap:
        neg_ratio, c_c, c_k = heapq.heappop(heap)
        ratio = -neg_ratio
        
        if c_c not in active_cluster_ids or c_k not in active_cluster_ids:
            continue
            
        c_c, c_k = min(c_c, c_k), max(c_c, c_k)
        shared_w = interconnections.get(c_c, {}).get(c_k, 0)
        if shared_w == 0:
            continue  # stale entry: pair no longer connected
        total = cluster_internal_edges[c_c] + cluster_internal_edges[c_k] + shared_w
        current = shared_w / total
        if current < threshold or abs(current - ratio) > 1e-12:
            continue  # stale entry: a fresh one is already in the heap
        if ratio < threshold:
            break
        current_clusters[c_c].update(current_clusters[c_k])
        
        # Update the combined internal edge tracking math
        cluster_internal_edges[c_c] = cluster_internal_edges[c_c] + cluster_internal_edges[c_k] + shared_w
        
        del current_clusters[c_k]
        del cluster_internal_edges[c_k]
        active_cluster_ids.remove(c_k)
        
        # Track updated neighbor relationships locally
        updated_neighbors = {}
        if c_c in interconnections:
            for c_neigh, w in interconnections[c_c].items():
                if c_neigh in active_cluster_ids and c_neigh != c_k:
                    updated_neighbors[c_neigh] = updated_neighbors.get(c_neigh, 0) + w
            del interconnections[c_c]
            
        if c_k in interconnections:
            for c_neigh, w in interconnections[c_k].items():
                if c_neigh in active_cluster_ids and c_neigh != c_c:
                    updated_neighbors[c_neigh] = updated_neighbors.get(c_neigh, 0) + w
            del interconnections[c_k]

        for c_source in list(interconnections.keys()):
            if c_source in active_cluster_ids:
                if c_c in interconnections[c_source]:
                    updated_neighbors[c_source] = updated_neighbors.get(c_source, 0) + interconnections[c_source][c_c]
                    del interconnections[c_source][c_c]
                if c_k in interconnections[c_source]:
                    updated_neighbors[c_source] = updated_neighbors.get(c_source, 0) + interconnections[c_source][c_k]
                    del interconnections[c_source][c_k]

        # Save updated ratios back to our map and priority queue
        for c_neigh, new_w in updated_neighbors.items():
            c_first, c_second = min(c_c, c_neigh), max(c_c, c_neigh)
            
            total_footprint = cluster_internal_edges[c_first] + cluster_internal_edges[c_second] + new_w
            new_ratio = new_w / total_footprint if total_footprint > 0 else 0.0
            
            interconnections.setdefault(c_first, {})[c_second] = new_w
            if new_ratio >= threshold:
                heapq.heappush(heap, (-new_ratio, c_first, c_second))

    # Return remaining valid clusters
    all_modules = [current_clusters[cid] for cid in active_cluster_ids]

    return all_modules

def run_stm_algorithm(G, threshold=0.05):
    print(f"Network Stats: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
    print("Executing Phase 1: Computing Signal Transduction Matrices...")
    signals = compute_ppi_signals(G)
    print("Executing Phase 2: Generating Core Pre-clusters...")
    preclusters = build_preclusters(G, signals)
    print(f"Executing Phase 3: Merging overlapping structures (Threshold >= {threshold})...")
    final_modules = merge_clusters(G, preclusters, threshold=threshold)

    print("\n=======================================================")
    print(f"SUCCESS: Extracted {len(final_modules)} Total Functional Modules:")
    print("=======================================================")
    for index, module in enumerate(final_modules, 1):
        print(f"Module #{index} ({len(module)} proteins): {sorted(list(module))}")
        
    return final_modules


def evaluate_network_modules(G, raw_modules):
    """
    Computes Network Modularity (Q) and Conductance metrics.
    """
    print("\n=======================================================")
    print("EXECUTING EVALUATION LAYER")
    print("=======================================================")

    try:
        q_score = nx.community.modularity(G, raw_modules)
        print(f"Network Modularity (Q-Score): {q_score:.4f}")
        if q_score > 0.3:
            print("Result: Strong modular structure detected (Q > 0.3).")
        else:
            print("Result: Weak structural separation.")
    except Exception as e:
        print(f"Modularity calculation skipped: {e}")
        
    # 3. Apply the size filter for clean biological module report
    final_modules = [m for m in raw_modules if len(m) >= 4]
    
    print("\n=======================================================")
    print(f"SUCCESS: Extracted {len(final_modules)} Cleaned Functional Modules (Size >= 4):")
    print("=======================================================")
    
    # Quick structural diagnosis printout
    sizes = [len(m) for m in final_modules]
    if sizes:
        print(f"Max Module Size: {max(sizes)}")
        print(f"Average Module Size: {sum(sizes)/len(sizes):.1f}")
    else:
        print("No modules met the size >= 4 constraint.")