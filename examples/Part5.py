import pandas as pd
import numpy as np
import math
from Part4 import best_split
def generate_tree(data:pd.DataFrame,label_column:str,mindepth:int,info_thresh:float,level:int=0)->tuple:
    """
    * `data:pd.DataFrame`: The data set we're using as our reference.
    * `label_column:str`: The column we'll use as our labels
    * `mindepth:int`: The minimum depth for the tree (up until which it ignores the minimum information gain threshold).
    * `info_trhesh:float`: The minimum information gain threshold. After the minimum depth, if the next split's gain is smaller than this, it'll stop splitting
    Generates a decision tree based on the given parameters
    .
    """
    #If there's nothing there, we can't do much
    if len(data) == 0:
        return (None,None,None)
    #Finds the best split
    bs = best_split(data,label_column)
    #If it's below the minimum depth and the split isn't worth it
    if level>=mindepth and bs[2]<info_thresh:
        #Return the most common label and become a leaf
        return (data[label_column].mode()[0],None,None)
    
    children = dict()
    #If it's a categorical attribute
    if data[bs[0]].dtype == object:
        #Find all possible values
        values = {value for value in data[bs[0]]}
        #Generate a tree for each one
        children = {value:generate_tree(data[data[bs[0]]==value],label_column,mindepth,info_thresh,level+1) for value in values}
    else:
        #Generates the tree for the lesser/equal subset
        children['lessereq'] = generate_tree(data[data[bs[0]]<=bs[1]],label_column,mindepth,info_thresh,level+1)
        #Generates the tree for the greater subset
        children['greater'] = generate_tree(data[data[bs[0]]>bs[1]],label_column,mindepth,info_thresh,level+1)
    return (bs[0],bs[1],children)
def classify_point(point:pd.Series,tree:tuple)->str:
    """
    * `point:pd.Series`: A data row
    * `tree:tuple`: A tree
    Classifies a point using a decision tree
    """
    # If there's no children, it's a leaf.
    if not tree[2]:
        return tree[0]
    # If the attribute doesn't exist in the data point, something's wrong
    if not tree[0] in point.index:
        return None
    else:
        # If it's numerical
        if tree[1]:
            if point[tree[0]]<=tree[1]:
                return classify_point(point,tree[2]['lessereq'])
            else:
                return classify_point(point,tree[2]['greater'])
        else:
            return classify_point(point,tree[2][point[tree[0]]])