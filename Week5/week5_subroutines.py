#importing the important libraries 

import pandas as pd 
import numpy as np  
import networkx as nx 
from networkx.algorithms.community import louvain_communities

#building the graph from the protien list. Not sure why but I had to name it v11 not v12?

G0 = nx.read_weighted_edgelist("/course/data/4932_protein_links_v11_0.txt",comments="#",nodetype=str)

#this step prunes the graph to keep the important points. Potentially this is not needed for this week, let me know what you guys think?

list_edges = []
threshold_score = 750
for edge in G0.edges():
    weights = list(G0.get_edge_data(edge[0],edge[1]).values())
    if (weights[0]<=threshold_score):
        list_edges.append(edge)

for edge in list_edges:
    G0.remove_edge(edge[0],edge[1])  
        
#The question asks to use the algotithm of a connected component so i took the largest

largest_cc = max(nx.connected_components(G0), key = len)
G = G0.subgraph(largest_cc)

# The next two functions do almost the same thing excpet the first one just returns the communities and the next one retuns their size.

def partition_graph(x):
    final_communities = []
    communities = nx.community.louvain_communities(x, seed = 42)
    sorted_communities = sorted(communities, key = len)  
    for community in sorted_communities:
        if len(community)>1:
            final_communities.append(community)
    return final_communities

def partition_graph_size(x):
    final_communities = []
    communities = nx.community.louvain_communities(x, seed = 42)
    sorted_communities = sorted(communities, key = len)  
    for community in sorted_communities:
        if len(community)>1:
            final_communities.append(community)
    return [(community,len(community)) for community in final_communities]

#This one find alls the adjacent communities to a given taget protien_communities
#the input of communities into the function is the partition_graph(x) function which is why I made two

def find_adjacent_communities(G, communities, target_protien):
    neighbors = G.neighbors(target_protien)
    protien_communities = {}
    adjecent_communities = []
    for comunity in communities:
        for node in comunity:
            protien_communities[node] = comunity
    for node in neighbors:
        if protien_communities[node] not in adjecent_communities and protien_communities[node]!= protien_communities[target_protien]:
            adjecent_communities.append(protien_communities[node])

    return adjecent_communities

        

#print(find_adjacent_communities(G, partition_graph(G), '4932.YDR227W')) we can choose the protien the bio chem people want

# print(partition_graph_size(G))
                        
# result = partition_graph(G)
# sizes = [size for community, size in result]
# print(sizes)
# print("Numer of communities", len(sizes))
