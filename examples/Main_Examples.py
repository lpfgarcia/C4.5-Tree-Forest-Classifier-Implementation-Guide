"""
This is an implementation of my guide available at
https://github.com/m3101/C4.5-Tree-Forest-Classifier-Implementation-Guide
Feel free to analyse and improve upon it!

Sorry for the mess :p

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

#For generating the figures for the guide
#If there's any argument on the commandline, it'll generate figures
import sys
genfigures = len(sys.argv)>1
if genfigures:
    import matplotlib.pyplot as plt
    import seaborn as sns
    palette = sns.color_palette(['#FFA500','#0000FF'])

#Imports from the specific parts' implementations
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

# Part 1 - Entropy
train = pd.read_csv("train.csv")
print(f"Part 1:")
print(f"\tEntropy = {set_entropy(train['colour'])}")

# Part 2 - Attribute Entropy
apples = pd.read_csv("apples.csv")
print(f"Part 2:")
print(f"\tEntropy (Taste as label) = {set_entropy(apples['taste'])}")
print(f"\tEntropy (Chile) = {set_entropy(apples[apples['country']=='Chile']['taste'])}")
print(f"\tEntropy (Green) = {set_entropy(apples[apples['colour']=='green']['taste'])}")
print(f"\tEntropy (Red) = {set_entropy(apples[apples['colour']=='red']['taste'])}")
print(f"\tAttribute Entropy (Country): {attribute_entropy(apples,'taste','country')}")
print(f"\tAttribute Entropy (Colour): {attribute_entropy(apples,'taste','colour')}")

# Part 3 - Numerical Attribute Entropy

print(f"Part 3:")
print(f"\tNum. Attribute Entropy (x at 0): {num_attribute_entropy(train,'colour','x',0)}")
print(f"\tNum. Attribute Entropy (y at 0): {num_attribute_entropy(train,'colour','y',0)}")
print(f"\tOutput of minimum_num_attribute_entropy for x: {minimum_num_attribute_entropy(train,'colour','x')}")
print(f"\tOutput of minimum_num_attribute_entropy for y: {minimum_num_attribute_entropy(train,'colour','y')}")

# Part 4 - Splitting
trainq = pd.read_csv("trainq.csv")
print(f"Part 4:")
bs = best_split(trainq,'colour')
print(f"Best split: {bs}")
if genfigures:
    sns.pairplot(trainq,x_vars='x',y_vars='y',hue='colour',palette=palette)
    if bs[0]=='x':
        plt.axvline(bs[1])
    else:
        plt.axhline(bs[1])
    plt.show()
# Part 5 - Trees
print("Part 5:")
if genfigures:
    sns.pairplot(trainq,x_vars='x',y_vars='y',hue='colour',palette=palette)

print("Generating tree...")
tree = generate_tree(trainq,'colour',2,0.1)
print_tree(tree,genfigures=True)

if genfigures:
    plt.show()
print("Done!")
np.random.seed(31)
iris = pd.read_csv("iris.csv")
def test():
    random = np.random.rand(len(iris))<0.25
    train = iris[random]
    test = iris[~random]
    #print(f"\tTraining with {len(train)} items, testing with {len(test)}")
    #Disable figure generation (it's hardcoded for trainq.csv)
    tree = generate_tree(train,'class',2,0.1)

    #print_tree(tree)

    results = [(classify_point(point,tree),point['class']) for i,point in test.iterrows()]
    score = round(sum([a[0]==a[1] for a in results])/len(test)*100,2)
    #print(f"\tScore: {score}%")
    return score
n=20
print(f"Calculating average score for {n} tests using the iris dataset")
avg = sum([test() for i in range(n)])/n
print(f"{round(avg,2)}%")