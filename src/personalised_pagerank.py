
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pandas as pd

from src.reading_in import read_in_proteins, _load_essential_proteins


PEP5 = "YMR231W"
MIN_DISTANCE = 4
MAX_DISTANCE = np.inf
PERSONALISATION_VECTOR = {PEP5 : 1}

def pagerank(G: nx.Graph, )