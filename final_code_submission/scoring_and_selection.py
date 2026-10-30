import networkx as nx
import numpy as np
import mygene
from wk6_toolbox.wk6_subroutines import all_shortest_paths, sort_proteins_by_frequency

def text_selection(protein_name: str, absolute_threshold: int = 0, absolute_warning_threshold: int = 1000, relative_threshold: float = 0, return_bool: bool = True, searched_string: str = 'ribosom'):
    """
    Looks through the "term" descriptor for the list of dictionaries "go" in mygene, and counts instances where the searched string is included.
    Currently this function doesn't work well and is error prone. Needs to be improved a lot.
    Main errors come from the way mygene stores information is very varied, and also uncertainty about the specific significance of particular parts 
    of the mygene output.

    Parameters
    ----------
    protein_name: str
        Systematic name of the protein
    absolute_threshold: int = 0
        Threshold that, if the absolute count of "ribosome" instances is above (but not equal to), function can evaluate to False.
    absolute_warning_threshold: int = 1000
        If the absolute count is above (but not equal) to this, then a warning will be printed about this protein's absolute count.
    relative_threshold: float = 0
        Threshold that, if the relative count of "ribosome" insttances is above (but not equal to), function can evaluate to False.
    searched_string: str = 'ribosom'
        Increments the count if it is included in the descriptor.
    return_bool: bool = True
        If True, a bool is returned corresponding to whether both thresholds are met (True if they are, False if at least one isn't).
        If False, a Tuple, (absolute_count, relative_count), is returned

    Returns
    -------
    If "return_bool" True, a bool is returned corresponding to whether both values are below thresholds (True if they are, False if at least one isn't).
    If "return_bool" False, a Tuple, (absolute_count, relative_count), is returned
    """

    mg = mygene.MyGeneInfo()
    important_categories = list(mg.getgene(protein_name)['go'].items())
    absolute_count = 0
    total_count = 0
    for dictionaries in important_categories:
        try:
            for entry in dictionaries[1]:
                total_count += 1
                if searched_string in entry['term']:
                    absolute_count += 1
        except:
            print(dictionaries)
    relative_count = absolute_count/total_count
    if return_bool:
        if absolute_count <= absolute_threshold:
            if relative_count <= relative_threshold:
                if absolute_count > absolute_warning_threshold:
                    print(f'Warning: {protein_name} has {absolute_count} instances.')
                return True
        return False
    return (absolute_count, relative_count)

def selecting_high_frequency_intermediary_nodes(G : nx.Graph, adjacent_communities : tuple, centrality_measure_dictionary : dict, path_length: int = 4, upwards_leeway: int = 0, downwards_leeway: int = 0, nodes_per_community: int = 3, target : str = "YMR231W", print_bool: bool = False, exclude_representatives: bool = False) -> list[list]:
    """
    Takes in a centrality measure, a network, and the adjacent communities, and produces a list of nodes of interest.
    Nodes of interest are nodes which appear with a high frequency in the shortest paths connecting to a high centrality node in an adjacent community.

    Parameters
    ----------
    G : nx.Graph
        The networkx Graph object to search.
    source : str
        Starting node for the path, our chosen community representative.
    adjacent_communities : tuple 
        Output of get_adjacent_communities
    centrality_measure_dictionary : dict
        The dictionary obtained from networkx centrality measure functions.
    path_length: int = 4
        Distance the returned nodes have to be from the target protein.
    upwards_leeway: int = 0
    downwards_leeway: int = 0
        Allowed deviation from path_length, so that the distance has to be within [path_length - downwards_leeway, path_length + upwards_leeway].
    nodes_per_community: int = 3
        The number of nodes of interest returned per adjacent community.
    target : str = "YMR231W"
        Ending node for the path. Default is our target (pep5).
    print_bool : bool = True
        If True, prints out the nodes with their frequencies.
    exclude_representatives: bool = False
        If True, representatives themselves will not be included in their own path, so they won't show up as important nodes in the returned array
        This does not affect the nodes_per_community. X nodes will still be selected, it just alters whether the represenative is one of the X nodes.

    Returns
    -------
    list[list]
        A list of lists, each list containing the source (high centrality representative of an adjacent community) and, as a set,
        the top #nodes_per_community with the highest shortest_path_frequency (each entry being a tuple of (protein, frequency count))
    """
    allowable_lengths = range(path_length - downwards_leeway, path_length + upwards_leeway + 1)

    adjacent_communities = adjacent_communities[1]
    adjacent_representatives = [
        max(community, key=lambda n: centrality_measure_dictionary[n])
        for community in adjacent_communities
    ]

    total_array = []
    for source in adjacent_representatives:
        source_array = [source]
        paths = all_shortest_paths(G, str(source), target=target)  # list of paths (each a list of nodes)
        shortest_path_frequencies = sort_proteins_by_frequency(paths)
        i_set = set()
        for i in range(exclude_representatives, len(shortest_path_frequencies)-1):
            if nx.shortest_path_length(G, shortest_path_frequencies[i][0], target) in allowable_lengths:
                i_set.add(shortest_path_frequencies[i])
                if len(i_set) > nodes_per_community:
                    break 
            
        source_array.append(i_set)
        if print_bool:
            print(source_array)
        total_array.append(source_array)
    return total_array

def compress_important_nodes(list_of_selected_lists : list, exclude_empty_sets : bool = True) -> dict[str, set[tuple[str,int]]]:
    """
    To be used after selecting_high_frequency_intermediary_nodes. 
    Takes a list of multiple returned selecting_high_frequency_intermediary_nodes, and compresses them into a dictionary.
    The dictionary stores the source protein as a key, and with that key stores the set of tuples of important nodes as the value.

    Parameters
    ----------
    list_of_selected_lists : list
        A list of multiple returned objects from multiple selecting_high_frequency_intermediary_nodes calls.
    exclude_empty_sets : bool = True
        If True, will remove key entries which only have an empty set as a value.

    Returns
    -------
    dict[str, set[tuple[[str,int]]]]
        A dictionary with the protein name as keys. The set of important high frequency proteins on that protein's shortest path, as determined with
        selecting_high_frequency_intermediary_nodes, are stored as the keys in a tuple with their name and their frequency.
    """

    dictionary = {}
    for lists in list_of_selected_lists:
        for things in lists:
            if things[0] in dictionary.keys() == True:
                dictionary[things[0]].update(things[1])
            else:
                dictionary.update({things[0] : things[1]})       
    if exclude_empty_sets:
        dictionary = {k: v for k, v in dictionary.items() if len(v) != 0} # If len=0, i.e., if empty set, then remove.
    return dictionary

def simple_text_selection(protein_name: str, absolute_threshold: int = 0, absolute_warning_threshold: int = 1000, relative_threshold: float = 0, return_bool: bool = True, searched_string: str = 'ribosom'):
    """
    Turns the dictionary from mygene for the given protein into a string.
    Then, counts the absolute number of instances of the searched string in words in the string, and total number of words (rather dumbly & simplistically). 
    Relative is measured by absolute/total.

    Parameters
    ----------
    protein_name: str
        Systematic name of the protein
    absolute_threshold: int = 0
        Threshold that, if the absolute count of "ribosome" instances is above (but not equal to), function can evaluate to False.
    absolute_warning_threshold: int = 1000
        If the absolute count is above (but not equal) to this, then a warning will be printed about this protein's absolute count.
    relative_threshold: float = 0
        Threshold that, if the relative count of "ribosome" insttances is above (but not equal to), function can evaluate to False.
    searched_string: str = 'ribosom'
        Increments the count if it is included in the descriptor.
    return_bool: bool = True
        If True, a bool is returned corresponding to whether both thresholds are met (True if they are, False if at least one isn't).
        If False, a Tuple, (absolute_count, relative_count), is returned

    Returns
    -------
    If "return_bool" True, a bool is returned corresponding to whether both values are below thresholds (True if they are, False if at least one isn't).
    If "return_bool" False, a Tuple, (absolute_count, relative_count), is returned
    """

    mg = mygene.MyGeneInfo()
    strings = np.array(str(mg.getgene(protein_name)).lower().split(' '))
    total_count = strings.size
    searched_string = searched_string.lower()
    strings = np.array([s for s in strings if searched_string in s])
    absolute_count = strings.size
    relative_count = absolute_count/total_count
    if return_bool:
        if absolute_count <= absolute_threshold:
            if relative_count <= relative_threshold:
                if absolute_count > absolute_warning_threshold:
                    print(f'Warning: {protein_name} has {absolute_count} instances.')
                return True
        return False
    return (absolute_count, relative_count)

def trim_nodes_dictionary(dictionary : dict, selection_function, exclude_empty_sets : bool = True) -> dict:
    """
    Trims dictionary based on some True/False exclusion criteria based upon analysing the value-proteins in the dictionary. 
    Exclusion criteria given by the input function

    Parameters
    ----------
    dictionary : dict
        Dictionary in the same format as produced by compress_important_nodes().
    selection_function
        Function  which evaluates to true or false based on the input of a protein.
        E.g., could use as input:
            lambda protein : simple_text_selection(protein, absolute_threshold=1, relative_threshold=1)
    exclude_empty_sets : bool = True
        If True, protein keys with an empty set will be removed before returning. 

    Returns
    -------
    dict
        A dictionary containing the key:value pairs of input dictionary for value-proteins that pass the selection criteria. 
    
    """
    trimmed_nodes_list = {
        key: {t for t in tuples_set if selection_function(t[0])}
        for key, tuples_set in dictionary.items()
    }
    if exclude_empty_sets:
        trimmed_nodes_list = {k: v for k, v in trimmed_nodes_list.items() if len(v) != 0} # If len=0, i.e., if empty set, then remove.

    return trimmed_nodes_list

