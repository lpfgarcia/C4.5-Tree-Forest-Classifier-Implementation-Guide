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

# Part 1
def set_entropy(labels:pd.Series)->float:
    """
    * `labels:pd.Series`: The column corresponding to the labels on your dataset.
    Calculates the entropy of a set of data points using their labels.
    """
    # Firstly we extract the relative frequencies of each possible class
    freqs = labels.value_counts(normalize=True)
    # And return the opposite of the sum of the products of each frequency by its log2
    return -sum([math.log2(pr)*pr for pr in freqs])
# Part 2
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

# Part 3
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
# Part 4
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
# Part 5
def generate_tree(data:pd.DataFrame,label_column:str,mindepth:int,info_thresh:float,level:int=0,x=(0,1),y=(0,1))->tuple:
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
        #This first part is only used for visualisation and can be ignored
        if genfigures:
            plt.text(((x[0]+x[1])/2)*20-10,((y[0]+y[1])/2)*20-10,data[label_column].mode()[0],bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=data[label_column].mode()[0],))

        #Print the leaf point
        print("\t"*level+f" Label = {data[label_column].mode()[0]}")
        #Return the most common label and become a leaf
        return (data[label_column].mode()[0],None,None)
    #Prints the current tree node
    print("\t"*level+f"Split at {bs[0]}"+(" = "+str(round(bs[1],4)) if bs[1] else ""))
    #This part is only used for visualisation and can be ignored
    gx=lx=x
    gy=ly=y
    if genfigures:
        if bs[0]=='x':
            plt.axvline(bs[1],y[0],y[1])
            lx = (x[0],(bs[1]+10)/20)
            gx = ((bs[1]+10)/20,x[1])
        elif bs[0]=='y':
            plt.axhline(bs[1],x[0],x[1])
            ly = (y[0],(bs[1]+10)/20)
            gy = ((bs[1]+10)/20,y[1])
    
    children = dict()
    #If it's a categorical attribute
    if data[bs[0]].dtype == object:
        #Find all possible values
        values = {value for value in data[bs[0]]}
        #Generate a tree for each one
        children = {value:(generate_tree(data[data[bs[0]]==value],label_column,mindepth,info_thresh,level+1),print("\t"*level+f"When {value}"))[0] for value in values}
    else:
        #Generates the tree for the lesser/equal subset
        print("\t"*level+f"If lesser than or equal to {round(bs[1],4)}")
        children['lessereq'] = generate_tree(data[data[bs[0]]<=bs[1]],label_column,mindepth,info_thresh,level+1,lx,ly)
        print("\t"*level+"Otherwise:")
        #Generates the tree for the greater subset
        children['greater'] = generate_tree(data[data[bs[0]]>bs[1]],label_column,mindepth,info_thresh,level+1,gx,gy)
    return (bs[0],bs[1],children)
def classify_point(point:pd.Series,tree:tuple)->str:
    if not tree[2]:
        return tree[0]
    if not tree[0] in point:
        return None
    else:
        if tree[1]:
            if point[tree[0]]<=tree[1]:
                return classify_point(point,tree[2]['lessereq'])
            else:
                return classify_point(point,tree[2]['greater'])
        else:
            return classify_point(point,tree[2][point[tree[0]]])

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
tree = generate_tree(trainq,'colour',2,0.1)
if genfigures:
    plt.show()
print("Done!")

iris = pd.read_csv("iris.csv")
np.random.seed(31)
random = np.random.rand(len(iris))<0.25
train = iris[random]
test = iris[~random]
print(f"Training with {len(train)} items, testing with {len(test)}")
#Disable figure generation (it's hardcoded for trainq.csv)
genfigures=False
tree = generate_tree(train,'class',2,0.1)
results = [(classify_point(point,tree),point['class']) for i,point in test.iterrows()]
print(f"Score: {round(sum([a[0]==a[1] for a in results])/len(test)*100,2)}%")