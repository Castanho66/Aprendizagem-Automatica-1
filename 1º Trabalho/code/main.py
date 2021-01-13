#Import the necessary libraries
from sklearn.model_selection import train_test_split
import numpy as np
import math
from collections import Counter
from treeNode import treeNode
from DecisionTreeREPrune import DecisionTreeREPrune

#Load and prepare data from the file
data=np.genfromtxt("weather.nominal.csv", delimiter=",", dtype=None, encoding=None)
xdata=data[1:,0:-1]    #  dados: da segunda à ultima linha, da primeira à penúltima coluna
ydata=data[1:,-1]      # classe: da segunda à ultima linha, só última coluna

header = data[0:1,0:]   #array with all the headers from the file

classifier = DecisionTreeREPrune('gini', False, header)

x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=.75, random_state=1)

#print("X train -> " , x_train,"\n")
#print("Y train -> " , y_train,"\n")

classifier.fit(x_train, y_train)
#classifier.prune(x_train, y_train)

print("---- Decision Tree ----\n")
#print("Decision tree to long to image it.")
print(classifier, "\n")
#print("----------------------\n")

#print("X test -> " , x_test,"\n")
#print("Y test -> " , y_test,"\n")

result = classifier.score(x_test, y_test)

print("Percentagem de casos corretamente classificados {:2.2%}".format(result))