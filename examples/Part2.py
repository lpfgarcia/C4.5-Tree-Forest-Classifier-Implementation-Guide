import pandas as pd
import numpy as np
import math
from Part1 import set_entropy
def attribute_entropy(data:pd.DataFrame,label_column:str,attribute:str)->float:
    """
    * `data:pd.DataFrame`: The data set we're using as our reference.
    * `label_column:string`: The column we'll use as our labels
    * `attribute:string`: The column we'll use as our subset-generating attribute
    Calculates the Attribute Entropy for labelled set
    """
    if data[attribute].dtype!=object:
        return None
    #Firstly, we list all possible values for the attribute
    possible_values = {value for value in data[attribute]}
    #Then we separate the subsets
    subsets = [data[data[attribute]==value] for value in possible_values]
    #Then we calculate each set's entropy
    entropies = [set_entropy(subset[label_column]) for subset in subsets]
    #And we return the weighted sum of all entropies
    return sum([entropies[i]*(len(subsets[i])/len(data)) for i in range(len(subsets))])