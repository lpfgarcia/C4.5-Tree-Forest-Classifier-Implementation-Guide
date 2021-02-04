import pandas as pd
import numpy as np
import math
def set_entropy(labels:pd.Series)->float:
    """
    * `labels:pd.Series`: The column corresponding to the labels on your dataset.
    Calculates the entropy of a set of data points using their labels.
    """
    # Firstly we extract the relative frequencies of each possible class
    freqs = labels.value_counts(normalize=True)
    # And return the opposite of the sum of the products of each frequency by its log2
    return -sum([math.log2(pr)*pr for pr in freqs])