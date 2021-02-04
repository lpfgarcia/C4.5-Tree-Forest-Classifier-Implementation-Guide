import pandas as pd
import numpy as np
import math

# Imports from the main guide
# Part 1
from Part1 import set_entropy
# Part 2
from Part2 import attribute_entropy
# Part 3
from Part3 import num_attribute_entropy,minimum_num_attribute_entropy
# Part 5
from Part5 import classify_point

def tree_score(tree:tuple,data:pd.DataFrame,label_column:str)->float:
    return sum([classify_point(point,tree)==point[label_column] for i,point in data.iterrows()])/len(data)

def prune_tree_score(tree:tuple,data:pd.DataFrame,label_column:str,subset:pd.DataFrame=None,root_tree:tuple=None,parent:tuple=None,childname:str=None)->tuple:
    # If we have no data, something went wrong
    if len(data)==0 or ((type(subset)!=type(None)) and len(subset)==0):
        return None
    # If we're a leaf, just return ourselves
    if not tree[2]:
        return tree
    # If all children are leaves and the same
    if all([tree[2][child][2]==None for child in tree[2]]) and len({tree[2][child][0] for child in tree[2]})==1:
        # Become a leaf
        return (tree[2][next(iter(tree[2]))][0],None,None)
    # Unless we are in the root, we should try pruning this node off
    if root_tree:
        score = tree_score(root_tree,data,label_column)
        #We define our parent node's child at our position as a leaf with our most frequent class
        parent[2][childname] = (subset[label_column].mode()[0],None,None)
        newscore = tree_score(root_tree,data,label_column)

        if newscore>score:
            #This branch stays a leaf
            return (subset[label_column].mode()[0],None,None)
        else:
            #We go back to what we were before
            parent[2][childname] = tree
    else:
        root_tree=tree
        subset = data
    # We'll now prune each child

    # If it's a numerical branch
    if tree[1] != None:
        #Prune the "lesser" tree
        tree[2]['lessereq']=prune_tree_score(tree[2]['lessereq'],data,label_column,data[data[tree[0]]<=tree[1]],root_tree,tree,'lessereq')
        tree[2]['greater']=prune_tree_score(tree[2]['greater'],data,label_column,data[data[tree[0]]>tree[1]],root_tree,tree,'greater')
        if not tree[2]['lessereq'] or not tree[2]['greater']:
                return (subset[label_column].mode()[0],None,None)
    else:
        for child in tree[2]:
            nsubset = data[data[tree[0]]==child]
            # If one of the children is empty
            if len(nsubset) == 0:
                return (subset[label_column].mode()[0],None,None)
            tree[2][child]=prune_tree_score(tree[2][child],data,label_column,nsubset,root_tree,tree,child)
    return tree
def prune_tree_entropy(tree:tuple,data:pd.DataFrame,label_column:str,threshold:float=1):
    # If we're a leaf, just return ourselves
    if not tree[2]:
        return tree
    # If all children are leaves and the same
    if all([tree[2][child][2]==None for child in tree[2]]) and len({tree[2][child][0] for child in tree[2]})==1:
        return (tree[2][next(iter(tree[2]))][0],None,None)
    # Check our information gain
    # If it's a numerical attribute
    entropy = set_entropy(data[label_column])
    if tree[1] != None:
        attentropy = num_attribute_entropy(data,label_column,tree[0],tree[1])
    else:
        attentropy = attribute_entropy(data,label_column,tree[0])
    
    #If it's smaller than the threshold, become a leaf
    if entropy-attentropy<threshold:
        return (data[label_column].mode()[0],None,None)
    else:
        #Prune our children
        #If it's a numerical attribute
        if tree[1]!=None:
            tree[2]['lessereq']=prune_tree_entropy(tree[2]['lessereq'],data[data[tree[0]]<=tree[1]],label_column,threshold)
            tree[2]['greater']=prune_tree_entropy(tree[2]['greater'],data[data[tree[0]]>tree[1]],label_column,threshold)
            if not tree[2]['lessereq'] or not tree[2]['greater']:
                return (data[label_column].mode()[0],None,None)
        else:
            for child in tree[2]:
                nsubset = data[data[tree[0]]==child]
                # If one of the children is empty
                if len(nsubset) == 0:
                    return (data[label_column].mode()[0],None,None)
                tree[2][child]=prune_tree_entropy(tree[2][child],nsubset,label_column,threshold)
    return tree

def prune_tree(tree:tuple,data:pd.DataFrame,label_column:str,method='entropy',threshold:float=1)->tuple:
    """
    * `tree:tuple`: A tree
    * `data:pd.DataFrame`: The data set we're using as our reference
    * `label_column:str`: The column we'll use as our labels
    Prunes a tree using the selected method
    """
    if method == 'score':
        tree = prune_tree_score(tree,data,label_column)
        return tree
    else:
        tree = prune_tree_entropy(tree,data,label_column,threshold)
        return tree