"""
This is an implementation of the extra section of my guide available at
https://github.com/m3101/C4.5-Tree-Forest-Classifier-Implementation-Guide
Feel free to analyse and improve upon it!

Sorry (again) for the mess :p

Copyright (C) 2021 Amélia O. F. da S. (a.mellifluous.one@gmail.com)

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
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
# Part 4
from Part4 import best_split
# Part 5
from Part5 import generate_tree,classify_point

# Tree-printing
from Utility import print_tree

# Imports from the extra guide's specific parts' implementations
from ExtraPart1 import prune_tree,tree_score
from ExtraPart2 import forest_classify,generate_forest,forest_score

# For comparing trees
def splits(tree:tuple)->int:
    if not tree[2]:
        return 0
    return sum([splits(tree[2][child]) for child in tree[2]])+1

# Part 1
print("Part 1")
iris = pd.read_csv('iris.csv')

np.random.seed(31)
random = np.random.rand(len(iris))<0.25
train = iris[random]
test = iris[~random]

tree = generate_tree(train,'class',2,0.1)
print(f"The original tree has {splits(tree)} splits and score {tree_score(tree,test,'class')}.")
print_tree(tree)

ptree_score = prune_tree(tree,test,'class','score')
print(f"Pruning it by score yields a tree with {splits(ptree_score)} splits and score {tree_score(ptree_score,test,'class')}")
print_tree(ptree_score)

tree = generate_tree(train,'class',2,0.1)
ptree_entropy = prune_tree(tree,test,'class','entropy',0.1)
print(f"Pruning it by entropy yields a tree with {splits(ptree_entropy)} splits and score {tree_score(tree,test,'class')}")
print_tree(ptree_entropy)

#Part 2
print(f"Part 2")
accents = pd.read_csv('accent-recognition-mfcc--1/accent-mfcc-data-1.csv')

random = np.random.rand(len(accents))<0.30
atrain = accents[random]
atest = accents[~random]

#For generating the figures for the guide
#If there's any argument on the commandline, it'll generate figures
import sys
genfigures = len(sys.argv)>1
if genfigures:
    import matplotlib.pyplot as plt
    import seaborn as sns
    palette = sns.color_palette(['#FFA500','#0000FF'])

print(f"Accent base:")
print(f"Using {len(atrain)} items for training, {len(atest)} for testing")
n_trees = 25
forest,averages = generate_forest(atrain,'language',n_trees,.05)
print(averages)
print(f"Forest score for {n_trees} trees, accent dataset: {forest_score(forest,atest,'language')}")
if genfigures:
    d=pd.DataFrame(averages,columns=['Mean tree score','Maximum tree score','Minimum tree score','Ensemble score'])
    d['Iteration'] = list(range(len(d)))
    sns.lineplot(data=pd.melt(d,'Iteration',var_name="Variable",value_name="Score"),x="Iteration",y="Score",hue="Variable")
    plt.show()
def score_accent():
    random = np.random.rand(len(accents))<0.075
    train = accents[random]
    test = accents[~random]
    tree = generate_tree(train,'language',2,0.1)
    return 100*tree_score(tree,test,'language')
n=20
print(f"Calculating single-tree average score for the accent recognition database (training with 7.5% of the set, just like the ensemble trees):")
avg = round(sum([score_accent() for i in range(n)])/n,2)
print(f"Single tree average score: {avg}%")


print(f"Iris base:")
print(f"Using {len(train)} items for training, {len(test)} for testing")
n_trees = 10
forest,averages = generate_forest(train,'class',n_trees,.05)
print(averages)
print(f"Forest score for {n_trees} trees, iris dataset: {forest_score(forest,test,'class')}")
if genfigures:
    d=pd.DataFrame(averages,columns=['Mean tree score','Maximum tree score','Minimum tree score','Ensemble score'])
    d['Iteration'] = list(range(len(d)))
    sns.lineplot(data=pd.melt(d,'Iteration',var_name="Variable",value_name="Score"),x="Iteration",y="Score",hue="Variable")
    plt.show()