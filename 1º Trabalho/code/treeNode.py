from sklearn.model_selection import train_test_split
import numpy as np
import math
from collections import Counter

class treeNode():
    '''
    A node class for a decision tree.
    '''
    def __init__(self):
        self.column = None  # index of feature to split on
        self.value = None   # value of the feature to split on
        self.name = None    #name of feature      
        self.left = None    # (TreeNode) left child
        self.right = None   # (TreeNode) right child
        self.leaf = False   # (bool)   true if node is a leaf, false otherwise
        self.header = None  # array of headers from the data (just to use in the print function)
        self.classes = Counter()  # (Counter) only necessary for leaf node:
                                  #           key is class name and value is
                                  #           count of the count of data points
                                  #           that terminate at this leaf
                    

    def predict_one(self, x):  #return the predicted label 

        if self.leaf:
            return self.name

        col_value = x[self.column]   #x[index] of the feature we want to compare later

        if col_value == self.value:
            #right_evaluated+=1
            return self.left.predict_one(x)   #return the value from the self.left prediction
        else:
            #bad_evaluated+=1
            return self.right.predict_one(x)  #return the value from the self.right prediction

    def as_string(self, level=0, prefix=""):    #Return a string representation of the tree with a node as a root

        result = ""
        if prefix:
            indent = "  |   " * (level - 1) + "  |-> "
            result += indent + prefix + "\n"
        indent = "  |   " * level

        if (str(self.name) == 'None'):
            result += indent + " Is " + str(self.column) + ": " + self.header + "  equal to:  " + "\n"
        else:
            result += indent + " Predict ->  " + str(self.name) + " -> reached a leaf \n"
        
        if not self.leaf:
            left_key = str(self.value)
         
            right_key = "no " + str(self.value)

            result += self.left.as_string(level + 1, left_key + ":")
            result += self.right.as_string(level + 1, right_key + ":")

        return result

    def __repr__(self):
        return self.as_string().strip()     #the strip() method removes any leading (spaces at the beginning) and trailing (spaces at the end) 
                                            #characters (space is the default leading character to remove)