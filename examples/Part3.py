import pandas as pd
import numpy as np
import math
from Part1 import set_entropy
def num_attribute_entropy(data:pd.DataFrame,label_column:str,attribute:str,threshold:float)->float:
    """
    * `data:pd.DataFrame`: The data set we're using as our reference.
    * `label_column:str`: The column we'll use as our labels
    * `attribute:str`: The column we'll use as our subset-generating attribute
    * `threshold:float`: The threshold we'll use for separating the subsets.
    Calculates the attribute entropy for a numerical attribute
    """
    if not data[attribute].dtype in ['float','int']:
        return None
    #We separate the subsets
    lessereq = data[data[attribute]<=threshold]
    greater = data[data[attribute]>threshold]
    #If any set is empty, we return the whole set entropy
    if len(lessereq)==0 or len(greater)==0:
        return set_entropy(data[label_column])
    #Then calculate their entropies and make the weighted average
    return set_entropy(lessereq[label_column])*(len(lessereq)/len(data))+\
           set_entropy(greater[label_column])*(len(greater)/len(data))
def minimum_num_attribute_entropy(data:pd.DataFrame,label_column:str,attribute:str)->float:
    """
    * `data:pd.DataFrame`: The data set we're using as our reference.
    * `label_column:str`: The column we'll use as our labels
    * `attribute:str`: The column we'll use as our subset-generating attribute
    Calculates the minimum attribute entropy for a numerial attribute
    """
    """
    I'll just bruteforce all possible points since it's the most reliable
    way of finding the absolute extrema and we'll only be dealing with small sets
    """
    #Firstly we list all posible points
    possible_thresh = {point for point in data[attribute]}
    #Then we calculate each threshold's entropy
    entropies = [(num_attribute_entropy(data,label_column,attribute,thr),thr) for thr in possible_thresh]
    #And find the minimum
    mi = min(entropies)
    return (mi[0],mi[1])