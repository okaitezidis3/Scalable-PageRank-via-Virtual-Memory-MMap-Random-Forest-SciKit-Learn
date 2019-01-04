from util import entropy, information_gain, partition_classes
import numpy as np 
import ast
import json

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}


    def is_number(self,s):
    #Function to calculate is a string is a number
    #https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
        try:
            float(s)
            return True
        except ValueError:
            return False


    def choose_attr_value(self, X, y):
        """Choose the attribute and value to split the tree that maximized the information gain criterion"""
        #https://stackoverflow.com/questions/44360162/how-to-access-a-column-in-a-list-of-lists-in-python


        max_split_attribute,max_split_val, max_information_gain = -1,-1,-1

        #The range is the number of attributes in the dataset
        for split_attribute_temp in range(len(X[0])):
            #For numbers
            if self.is_number(X[0][split_attribute_temp]):
                #print("numeric",X[0][split_attribute_temp])
                #If split_value is numerical then assign average value of the column
                column_item = [float(row[split_attribute_temp]) for row in X]
                split_val_temp = np.mean(column_item)
            #For strings
            else:
                #print("string",X[0][split_attribute_temp])
                #If split_value is string then assign the mode of the column
                #https://stackoverflow.com/questions/16330831/most-efficient-way-to-find-mode-in-numpy-array
                column_item = [row[split_attribute_temp] for row in X]
                (_, idx, counts) = np.unique(column_item, return_index=True, return_counts=True)
                index = idx[np.argmax(counts)]
                split_val_temp = column_item[index]
                
            
            X_left_, X_right_, y_left_, y_right_  = partition_classes(X,y,split_attribute_temp,split_val_temp)
            temp_information_gain = information_gain(y,[y_left_,y_right_])
            #Store the split with the best information gain
            if temp_information_gain > max_information_gain:
                max_split_attribute,max_split_val, max_information_gain = split_attribute_temp,split_val_temp,temp_information_gain
                best_X_left,best_X_right, best_y_left,best_y_right = X_left_, X_right_, y_left_, y_right_

        return max_split_attribute,max_split_val,best_X_left,best_X_right, best_y_left,best_y_right
        
        


    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        
        
        #Check minimum number of records in the leaf to avoid overfitting
        MIN_NODES = 0.05*len(y)
        MAX_DEPTH = 5
        self.tree = self.build_tree(X,y,1,MIN_NODES,MAX_DEPTH)

    # Build a decision tree
    # Reference https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
    # https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
    # https://www.youtube.com/watch?v=y6DmpG_PtN0&list=PLPOTBrypY74xS3WD0G_uzqPjCQfU6IRK-
    def build_tree(self, X, y, depth,MIN_NODES,MAX_DEPTH):
        #print("Depth",depth)
        #Check minimum node records and Entropy as a stopping Criterion
        node = {}
        if (len(y)<MIN_NODES or depth>MAX_DEPTH):
            node['split_attrib'] = 'leaf'
            node['classification'] = int(y.count('1') > y.count('0'))
            return node

        #Call choose_attr_value(self, X, y)
        split_attribute,split_val,X_left,X_right, y_left,y_right = self.choose_attr_value(X, y)
        node['split_val'] = split_val
        node['split_attrib'] = split_attribute

        #Right child Processing
        node['right'] = self.build_tree(X_right,y_right,depth+1,MIN_NODES,MAX_DEPTH)

        #Left child Processing
        node['left'] = self.build_tree(X_left,y_left,depth+1,MIN_NODES,MAX_DEPTH)
        return node

    def classify_helper(self, record,node):
        # TODO: classify the record using self.tree and return the predicted label
        #Check if node is leaf or not

        if (node['split_attrib'] != 'leaf'):
            #If split value is numeric
            if(self.is_number(node['split_val'])):
                #Enter left child for split_val <= split_val
                if (float(record[node['split_attrib']]) <= float((node['split_val']))):
                    return self.classify_helper(record,node['left'])
                #Enter right child for split_val > split_val
                else:
                    return self.classify_helper(record,node['right'])            
            #Else split value is String
            else:
                #Enter right child for split_val <> split_val
                if (str(record[node['split_attrib']]) != str(node['split_val'])):
                    return self.classify_helper(record,node['right'])
                #Enter left child for split_val == split_val
                else:
                    return self.classify_helper(record,node['left'])  
 
        #Reached leaf
        else:   
            return node['classification']


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        #Check if node is leaf or not
        node = self.tree
        value_for_class = self.classify_helper(record,node)
        return value_for_class