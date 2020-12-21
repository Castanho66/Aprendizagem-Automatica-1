#Import the necessary libraries
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
import numpy as np 
import math

#código decision tree
"""class myDecisionTreeREPrune():"""

def class_counts(rows, index):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.    difference between list = ['one','two',3,4] and dict = {'one: 1, two:2}
    #indice = 0
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[index]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1

    return counts


def gini(dataset,index):
    """
    Calculate the Gini Impurity for a list of rows.
    """
    label_counts = class_counts(dataset,index)
    numberOfRows = float(len(dataset))  #conta o numero de linhas do ficheiro
    impurity = 1

    for label in label_counts:
        numberOfLabelOccurence = label_counts[label]
        probability_of_label = float(numberOfLabelOccurence / numberOfRows)
        impurity -= (probability_of_label**2)
    return impurity

def choose_gini(dataset):
    """
    Choose the best gini option from all the attributes we have
    """
    columns_number = len(dataset[0])

    aux = 0

    array = []  #will put all the gini values from all the attributes here

    for x in range(columns_number):
        print(x)
        class_counts_aux = class_counts(dataset,x)
        print(class_counts_aux)

        gini_aux = gini(dataset,x)
        print(gini_aux)

        array.append(gini_aux)

    return array

def best_gini(dataset, array):

    best_gini =  min(array)

    return best_gini

def entropy(rows,index):

    """"Calculate the entropy for a list of rows.
	"""
    entropy = 0
    counts = class_counts(rows,index)   #{'sunny': 5, 'overcast': 4, 'rainy': 5} isto é o count, o for vai iterar isto mesmo
    for label in counts:
        prob_of_label = counts[label] / float(len(rows))
        entropy -= prob_of_label * math.log(prob_of_label, 2) 
       
    return entropy

def choose_entropy(dataset):
    """
    Choose the best entropy option from all the attributes we have
    """
    columns_number = len(dataset[0])

    aux = 0

    array = []  #will put all the gini values from all the attributes here

    for x in range(columns_number):
        print(x)
        class_counts_aux = class_counts(dataset,x)
        print(class_counts_aux)

        entropy_aux = entropy(dataset,x)
        print(entropy_aux)

        array.append(entropy_aux)

    return array

def best_entropy(dataset, array):

    best_entropy =  min(array)

    return best_entropy

#READ DATA
data=np.genfromtxt("weather.nominal.csv", delimiter=",", dtype=None, encoding=None)
xdata=data[1:,0:-1]    #  dados: da segunda à ultima linha, da primeira à penúltima coluna  
ydata=data[1:,-1]      # classe: da segunda à ultima linha, só última coluna

x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, random_state=0)

print(xdata)
print("\n")
print(xdata[0:,0])
print("\n")
print(ydata)
print("\n\n")
print((len(data)))
print("\n\n")
label_counts2 = class_counts(data,0)
print(label_counts2)
print("\n\n")
gini_array = choose_gini(xdata)
print("Our data has ", gini_array, " attributes.")

auxiliar = best_gini(xdata, gini_array)
print("Best gini value: ", auxiliar)

size = float(len(xdata))
print(size)

ounts = class_counts(xdata,0)
print(ounts)
print("\n\n\n")

entropy_array = choose_entropy(xdata)
print("Our data has ", entropy_array, " attributes.")

auxiliar2 = best_entropy(xdata, entropy_array)
print("Best gini value: ", auxiliar2)


""""row_counts = class_counts(xdata)
print(row_counts)"""
""""gini_imp = gini(data)
entropy_imp = entropy(data)
print(gini_imp)
print(entropy_imp)"""


"""classifier = myDecisionTreeREPrune(gini, True)
classifier.fit(x_data, y_data)
result = classifier.score(x_test, y_test)
print("Percentagem de casos corretamente classificados {:2.2%}".format(result)"""