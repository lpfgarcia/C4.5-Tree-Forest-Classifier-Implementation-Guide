import pandas as pd
import numpy as np
import math

# Imports from the main guide
# Part 5
from Part5 import generate_tree,classify_point

# Imports from the extra guide
from ExtraPart1 import prune_tree,tree_score

def forest_score(forest:list,data:pd.DataFrame,label_column:str)->float:
    return sum([forest_classify(forest,point)==point[label_column] for i,point in data.iterrows()])/len(data)

def generate_forest(data:pd.DataFrame,label_column:str,forest_size:int,threshold_deviation:float,bagsize:int=0,maxiterations:int=20)->tuple:
    """
    * `data:pd.DataFrame`: The data set we're using as our reference
    * `label_column:str`: The column we'll use as our labels
    * `forest_size:int`: The amount of weak classifiers to be generated
    * `threshold_variance:float`: The maximum deviation from the maximum score we'll tolerate within our ensemble
    Generates a forest classifier for the given training set.
    """
    #If there's no specified bagsize, we'll use 25% of the set size
    bagsize = math.ceil(len(data)/4)
    #Test subsets
    test = [data.loc[data.index[np.random.choice(len(data),bagsize)]] for i in range(forest_size)]
    #First we generate an empty forest
    forest = np.array([None for i in range(forest_size)])
    #And iterate until it's full
    #This variable is for keeping average tree scores for each iteration
    average = []
    i=0
    while None in forest:
        #First we generate our subsets
        train = [data.loc[data.index[np.random.choice(len(data),bagsize)]] for f in forest]
        print(f"Forest generation: creating {len(forest[forest==None])} new trees")
        #Populate the empty slots with new trees
        tforest = np.empty((forest_size,),dtype=object)
        tforest[:] = [tree if type(tree)!=type(None) else prune_tree(generate_tree(train[i],label_column,2,0.05),test[i],label_column,'score') for i,tree in enumerate(forest)]
        forest=tforest
        #Calculate their estimated scores
        scores = np.array([tree_score(tree,test[i],label_column) for i,tree in enumerate(forest)])
        #Find the maximum and calculate the deviations
        m = np.max(scores)
        dev = np.abs(scores - m)
        #Eliminate the ones with deviation above the threshold
        print("Eliminating trees")
        forest[dev>threshold_deviation] = None
        average.append((np.mean(scores),m,np.min(scores),forest_score(forest,data,label_column)))
        i+=1
        if i>maxiterations:
            break
    return (forest,average)
def forest_classify(forest:list,point:pd.Series)->str:
    """
    * `forest:list`: A forest
    * `point:pd.Series`: A data point
    Classifies a point using a forest
    """
    #Process all tree predictions
    predictions = [classify_point(point,tree) for tree in forest if type(tree)!=type(None)]
    #Return the most frequent
    return max(predictions, key = predictions.count)