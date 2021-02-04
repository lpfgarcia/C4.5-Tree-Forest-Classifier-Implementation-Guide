import pandas as pd
import numpy as np
import math
from Part1 import set_entropy
from Part2 import attribute_entropy
from Part3 import minimum_num_attribute_entropy
def best_split(data:pd.DataFrame,label_column:str)->tuple:
    """
    * `data:pd.DataFrame`: The data set we're using as our reference.
    * `label_column:str`: The column we'll use as our labels
    Calculates the best possible split. Returns (attribute,threshold/None,gain)
    """
    #We calculate the set entropy
    entropy = set_entropy(data[label_column])
    #And the attribute entropies
    attributes = [column for column in data.columns if column!=label_column]
    entropies = [(attribute_entropy(data,label_column,attribute),None) if data[attribute].dtype == object else minimum_num_attribute_entropy(data,label_column,attribute) for attribute in attributes]
    #Then the gains
    gains = [(entropy-ent[0],i) for i,ent in enumerate(entropies)]
    #Then we return the minimum
    minimum = sorted(gains,reverse=True)[0]
    return (attributes[minimum[1]],entropies[minimum[1]][1],minimum[0])