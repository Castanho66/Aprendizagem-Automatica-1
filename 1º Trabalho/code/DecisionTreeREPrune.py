from sklearn.model_selection import train_test_split
import numpy as np
import math
from collections import Counter
from treeNode import treeNode

class DecisionTreeREPrune:

    def __init__(self, criterion, prune, array_header):

        self.prune = prune
        self.array_header = array_header      #array_header is an 2d array that contains all the "column titles", 
                                              #example for weather.nominal =  [['outlook' 'temperature' 'humidity' 'windy' 'play']] , 
                                              #where array[0][0] = outlook, array[0][1] = temperature, basically it will be used when printing the tree
        self.root = None
     
        if criterion == 'gini' and prune == True:
           self.criterion = self._gini_
        if criterion == 'gini' and prune == False:
            self.criterion = self._gini_
        if criterion == 'entropy' and prune == True:
           self.criterion = self._entropy_
        if criterion == 'entropy' and prune == False:
            self.criterion = self._entropy_
        if criterion == 'erro' and prune == True:
           print("Erro")
        if criterion == 'erro' and prune == False:
            print("Erro")

    """def prune(self, X, y, node=None):   #Post-prune tree by merging leaves using error rate.  Recursively checks for leaves and compares error rate before and after
                                           #merging the leaves.  If merged improves error rate, merge leaves.         
        if node is None:
            node = self.root

        if not node.left.leaf:
            self.prune(X, y, node.left)

        if not node.right.leaf:
            self.prune(X, y, node.right)

        if node.left.leaf and node.right.leaf:
            leaf_y = self._predict(X, node)
            merged_classes = node.left.classes + node.right.classes
            merged_name = merged_classes.most_common(1)[0][0]
            merged_y = np.array([merged_name] * y.shape[0])
            leaf_score = sum(leaf_y == y) / float(y.shape[0])
            merged_score = sum(merged_y == y) / float(y.shape[0])

            if merged_score >= leaf_score:bool
               node.name = merged_name
               node.left = None
               node.right = None

        return node"""

    def _entropy_(self, y):    #returns the entropy impurity value from the y target values
        
        size= y.shape[0]   #amount of rows with y values
        impurity = 0
        for values in np.unique(y):  #np unique has ['no' 'yes'] when y is array with samples
            prob = sum(y == values) / float(size)
            impurity += prob * np.log2(prob)
        return -impurity


    def _gini_(self, y):

        size = y.shape[0]  #amount of rows with y values
        impurity = 0  
        for values in np.unique(y):    #np.unique is an array with all the unique values from the y input (will be the array with the target values)
            prob = sum(y == values) / float(size)
            impurity += prob**2
        return 1 - impurity

    def _information_gain(self, y, y1, y2):   #Return the information gain of making the given split.
        size = y.shape[0]
        child_inf = 0

        for index in (y1, y2):

            child_inf += self.criterion(index) * index.shape[0] / float(size)

        return self.criterion(y) - child_inf 

    def _make_split(self, x, y, split_index, split_value):  #Return the subsets of the dataset for the given split index & value.

        #Make split function receives 4 parameteres. X is a 2d array, and Y is a 1d array.
        #split_index is the index from the column of the features, so if it is 0 it represents the first column of attributes (int because it is a range)
        #split_valie represents the value from the column[i] of the features. if column[0] is outlook, so split_value[0] is overcast

        idx = x[:, split_index] == split_value   #idx its an array equal to x[:, split_index], 
                                                #if split index is 0 = ['overcast' 'rainy' 'rainy' 'sunny' 'sunny' 'sunny' 'rainy' 'sunny''rainy' 'overcast']
                                                #when we do the == equality operator, if the split value is equal to the attribute in the array, it returns true
                                                #for index 0, and split value = overcast it idx is -> [ True False False False False False False False False  True]

        #x[idx] = [['overcast' 'hot' 'high' 'FALSE']['overcast' 'hot' 'normal' 'FALSE']], represents the rows with overcast
        #y[idx] = ['yes' 'yes'] reprensents the value from the target when overcast is the split_value.... when the two arrays from above are in the data, Play is yes
        #x[idx == False] = [['rainy' 'mild' 'high' 'TRUE']
                           # ['rainy' 'mild' 'normal' 'FALSE']
                           #['sunny' 'hot' 'high' 'TRUE']
                           #['sunny' 'mild' 'high' 'FALSE']
                           # ['sunny' 'mild' 'normal' 'TRUE']
                           # ['rainy' 'mild' 'high' 'FALSE']
                           #['sunny' 'hot' 'high' 'FALSE']
                           # ['rainy' 'cool' 'normal' 'TRUE']]  represents the rest of the rows without the overcast ones
        #y[idx == False],['no' 'yes' 'no' 'no' 'yes' 'yes' 'no' 'no'], represents the rows in target without the ones from attribute overcast

        return x[idx], y[idx], x[idx == False], y[idx == False]

    def _choose_split_index(self, x, y):  #Return the index and value of the feature to split on. Determine which feature and value to split on. Return the index and
                                          # value of the optimal split along with the split of the dataset.
                                          # Return None, None, None if there is no split which improves information gain
        split_index, split_value, splits = None, None, None      #split in the beginning dont have any values
        gain = 0

        for i in range(x.shape[1]):     #index from 0 to range(x.shape[1]) -> x.shape[1] represents the size of the second part of x array, which is NUMBER OF FEATURES
            values = np.unique(x[:, i]) #values represent the unic values from feature 0 to range(x.shape[1])  example from weather.nominal.csv i = 0 ['overcast' 'rainy' 'sunny'] -> Outlook
                                                                                                                                                #i = 1 ['cool' 'hot' 'mild'] -> Temperature
                                                                                                                                                #i = 2 ['high' 'normal'] -> Humidity
                                                                                                                                                #i = 3 ['FALSE' 'TRUE'] -> Windy
            if len(values) < 1:  #if the array of the unic values is less thatn 1 we continue the for cicle
                continue

            for value in values:  #value = ['overcast' 'rainy' 'sunny'], so value[0] = overcast, value[1] = rainy, value[2] = sunny

                x1, y1, x2, y2 = self._make_split(x, y, i, value)   #i represents the index of the feature to split on, value represents the value of that feature
                                                                    #example i = 0 ['overcast' 'rainy' 'sunny']  and value = 0 is [overcast], so it is going to divide the attribute overcast 
                                                                    # in is target splits.  in weather.nominal.csv it will be overcast[yes] = 4 and overcast[no] = 0

                new_gain = self._information_gain(y, y1, y2)     #new_gain parameteres y1 = target values wherer rows from x1 happens // y2 = rows where x1 dont happen
                                                                 #with y1 = ['yes' 'yes'] and y2 = ['no' 'yes' 'no' 'no' 'yes' 'yes' 'no' 'no'], gain is = 0.125

                if new_gain > gain:                   
                    split_index = i              #split index equals i so we know the index of the column we pretend to split
                    split_value = value          #value of the split we are going to do -> value[0] = overcast, value[1] = rainy, value[2] = sunny from outlook
                    splits = (x1, y1, x2, y2)    #splits saves all the new array with the values
                    gain = new_gain              #gain is now the bigger gain that was calculated (maybe not the one from the first attibute, it can be the last)
        

        return split_index, split_value, splits

    def predict(self, x):   #Return an array of predictions for the array x -> ['yes' 'no' 'yes' 'yes' 'yes' 'yes' 'yes']

        return np.array([self.root.predict_one(row) for row in x])  

    def _build_tree(self, x, y): #Build the decision tree recursively

        node = treeNode()   #initialize a node

        index, value, splits = self._choose_split_index(x, y)    #uses the choose split function to obtain the index of the column we want so split, the value of the split(attribute)
                                                                 # and  the splits that are the new array values from the make split done inside the choose.

        if index is None or len(np.unique(y)) == 1:    #len when only exists one value of target in the y array (so its a leaf and predict the value) 
            node.leaf = True                           # give the tree node leaf attribute the boolean value True so we know that its this node is a Leaf
            node.classes = Counter(y)                  #Node classes  Counter({'no': 4})  Node classes  Counter({'yes': 3})            
            node.name = node.classes.most_common(1)[0][0]  #node.name has the value of the most  common value in the node.classes counter

        else:
            X1, y1, X2, y2 = splits                     #arrays with the values calcualted in the choose split from above
            node.column = index                         #give the node.column the value from the index of the column we want to split
            node.value = value                          #give the node.value the value from the attribute we will split
            node.header = self.array_header[0][index]   #gives the node.header the value from the column (only use this in the print time)
            node.left = self._build_tree(X1, y1)        #recursively do two new branches to do three (new trees)based on the new array values calculated aboce
            node.right = self._build_tree(X2, y2)
            
        return node

    def __str__(self):  #return string representation of the decision tree

        return str(self.root)

    def score(self, x, y):  #return accuracy of the test dates x and y from the original data

        self.x = np.array(x)
        self.y = np.array(y)
        N = self.x.shape[0]       #N = number of rows in the test file x
        y_pred = self.predict(x)     #ypred ->  ['yes' 'no' 'yes' 'yes' 'yes' 'yes' 'yes']
        accuracy = (np.sum(y == y_pred)) / N     #Array with comparison with the predict values and the train ones: y==y_pred -> [False  True  True  True False  True  True]
                                                 #np.sum = 5, because there are 5 values that are 5 values in the predict that are the same as the test ones
                                                 #accuracy = 5 / 7(x.shape[0] = number of rows on the data test)
        return accuracy

    def fit(self, x, y):   #generates the decision tree based on the train data (75% in the normal, and 25% with prune)
 
       n_samples = np.size(x, 0)     #number of samples in the file (number of rows in the x array)
       n_features = np.size(x, 1)    #number of features in the file (number of columns in the x array)

       self.x = x[:n_samples, :n_features]
       self.y = y[:n_samples]

       self.root = self._build_tree(self.x, self.y)