import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp

"""
Toolbox wk 3: Reading in and basic tools
    
    read_in_proteins()
        - read in PPI network data with 4932 prefixes deleted
        - has the optiion to include or excluede all essential proteins
        - constructrs a networkX graph of the PPI network that 
            i. is unweighted
            ii. only considers links above a certain threshold (default 750)
            iii. is connected

    largest_connected_component()
        - Given a graph, what is the largest connected component, and
            how many nodes are in it and how many got discarded?

    is_essential_protein(s)()
        - Given a protein, determine if it is an essential protein

    NOTE have not got a dedicated shortest path length function as networkx already has a function for that of the form
    nx.shortest_path_length(G, source="A", target="F")
"""

def _load_essential_proteins(essential_protein_filepath: str) -> pd.DataFrame:
    df = pd.read_csv(essential_protein_filepath)
    df.columns = ['counter', 'systematic name', 'organism', 'standard name', 'description']
    return df

def read_in_proteins(filepath: str, 
                     *, 
                     probability_threshold: int | float = 750,
                     include_essential_proteins: bool = True,
                     essential_protein_filepath: str | None = None,
                     connected: bool = False
                     ) -> nx.Graph:
    """
    Given filepath, read in proteins as a networkx Graph object.
    
    Parameters
    ----------
    filepath: str
        filepath to protein links data
    probability_threshold: int | float = 750
        method automatically removes all edges with combined_score less than this value
    include_essential_proteins: bool = True
        If False, removes essential proteins from network
    essential_protein_filepath: str | None = None
        filepath to essential protein data
    connected: bool = False
        if True, return the graph of only the largest connected component
    
    Returns
    -------
    G : nx.Graph
        nx.Graph object generated from our data

    Notes
    -----
    - This method removes essential nodes by removing all edges from the dataframe which are connected to at least one essential node.
        This will also remove proteins which are only connected to the graph via an essential node. 
        This is okay as any such nodes would by the removal of essential nodes anyway and will not change any of our connectivity measures.
    - This method doesn't worry about 'making' the graph unweighted. We operate with the understanding that the edge_attr 'combined_score'
        is not a weight but a representation of probability that there exists an edge.
    """
    #-- read file to dataframe --#
    df = pd.read_csv(filepath, sep=" ")
    df.columns = ['protein1', 'protein2', 'combined_score']
    
    #-- probability threshold --#
    df = df[df['combined_score'] >= probability_threshold].copy()
    
    #-- prefix --#
    # '.' is treated as placeholder for any character
        # regex=False ensures we treat it as '.'        
    df['protein1'] = df['protein1'].str.replace("4932.", "", regex=False) 
    df['protein2'] = df['protein2'].str.replace("4932.", "", regex=False)    

    #-- essential proteins --#
    if not include_essential_proteins:
        if essential_protein_filepath is None:
            raise ValueError("If include_essential_proteins is False, must give essential_protein_filepath")
        
        df_essential = _load_essential_proteins(essential_protein_filepath)
        
        # I got this from claude: removes an edge if either protein1 or protein 2 is an essential protein
        essential_set = set(df_essential['systematic name'])
        mask = ~df['protein1'].isin(essential_set) & ~df['protein2'].isin(essential_set)
        df = df[mask]

    G = nx.from_pandas_edgelist(df, source='protein1', target='protein2', edge_attr='combined_score')

    #-- connected --#
    if connected:
        # get graph of largest connected component
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc_nodes).copy()
        return G_connected

    return G

def largest_connected_component(G: nx.Graph, stats: bool = True) \
                -> tuple[nx.Graph, dict[str, int]] | nx.Graph:
    """
    Given a networkx Graph instance, return the largest connected component and stats

    Parameters
    ----------
    G: nx.Graph
        Graph instance to compute largest connected component of
    stats: bool = True
        If True, return dictionary with relevant stats
    
    Returns
    -------
    G_connected : nx.Graph
        Largest connected component of G
    stats_dict : dict[str, int]
        dictrionary recording the 'number_connected_nodes' and 'number_nodes_discarded'

    Notes
    -----
    I haven't properly debugged this
    """
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    G_connected = G.subgraph(largest_cc_nodes).copy()

    if stats:
        number_connected_nodes = len(largest_cc_nodes)
        number_nodes_discarded = G.number_of_nodes() - number_connected_nodes
        stats_dict = {
            "number_connected_nodes" : number_connected_nodes,
            "number_nodes_discarded" : number_nodes_discarded
        }

        return (G_connected, stats_dict)
    return G_connected

def is_essential_protein(protein: str,
                          essential_protein_filepath: str,
                          nametype: str = 'systematic') -> bool:
    """
    Checks if 'protein' is an essential protein.

    Parameters
    ----------
    protein : str
        systematic or standard name of protein
    essential_protein_filepath : str
        filepath to essential protein data
    nametype : str = 'systematic'
        either 'systematic' or 'standard'. Specifies which nametype to match against

    Returns
    -------
    is_essential : bool
        pretty self explanitory...

    Notes
    -----
    - This method isnt optimised for useage on large numbers of nodes as it reads in the data fresh every time it is called.
        Depending on useage, may change this later to allow user to pass in a vector of names and return a vector of booleans.
    """
    if nametype not in {'systematic', 'standard'}:
        raise ValueError(f"nametype must be 'systematic' or 'standard', got {nametype!r}")

    df_essential = _load_essential_proteins(essential_protein_filepath)
    essential_names = df_essential[nametype + " name"]
    return protein in essential_names.values

def are_essential_proteins(proteins: list[str],
                            essential_protein_filepath: str,
                            nametype: str = 'systematic') -> list[bool]:
    """
    Vectorized version of is_essential_protein: checks many proteins in one file read.

    Parameters
    ----------
    proteins : list[str]
        systematic or standard names of proteins
    essential_protein_filepath : str
        filepath to essential protein data
    nametype : str = 'systematic'
        either 'systematic' or 'standard'. Specifies which nametype to match against

    Returns
    -------
    is_essential : list[bool]
        one boolean per entry in `proteins`, in the same order
    """
    if nametype not in {'systematic', 'standard'}:
        raise ValueError(f"nametype must be 'systematic' or 'standard', got {nametype!r}")

    df_essential = _load_essential_proteins(essential_protein_filepath)
    essential_set = set(df_essential[nametype + " name"])  # O(1) lookups

    return [p in essential_set for p in proteins]
