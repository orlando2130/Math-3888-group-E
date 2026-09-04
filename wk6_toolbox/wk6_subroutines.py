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


