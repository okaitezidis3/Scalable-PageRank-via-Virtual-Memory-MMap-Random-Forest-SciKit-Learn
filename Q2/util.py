from scipy import stats
import numpy as np

# This method computes entropy for information gain
def entropy(class_y):
    # Input:
    #   class_y         : list of class labels (0's and 1's)

    # TODO: Compute the entropy for a list of classes
    #
    # Example:
    #    entropy([0,0,0,1,1,1,1,1,1]) = 0.92


    #Manual Calculation Slower than Numpy
    probabilities_zero = class_y.count("0")/len(class_y) if class_y.count("0")!=0 else 1
    probabilities_one = class_y.count("1")/len(class_y) if class_y.count("1")!=0 else 1
    entropy = -probabilities_zero* np.log2(probabilities_zero)-probabilities_one*np.log2(probabilities_one)

    #Reference https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    # _,counts = np.unique(class_y, return_counts=True)
    # return stats.entropy(counts, base=2)

    return entropy


#Function to check if string is a number
#https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False




def partition_classes(X, y, split_attribute, split_val):
    # Inputs:
    #   X               : data containing all attributes
    #   y               : labels
    #   split_attribute : column index of the attribute to split on
    #   split_val       : either a numerical or categorical value to divide the split_attribute

    # TODO: Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
    #
    # You will have to first check if the split attribute is numerical or categorical
    # If the split attribute is numeric, split_val should be a numerical value
    # For example, your split_val could be the mean of the values of split_attribute
    # If the split attribute is categorical, split_val should be one of the categories.
    #
    # You can perform the partition in the following way
    # Numeric Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all
    #   the rows where the split attribute is less than or equal to the split value, and the
    #   second list has all the rows where the split attribute is greater than the split
    #   value. Also create two lists(y_left and y_right) with the corresponding y labels.
    #
    # Categorical Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all
    #   the rows where the split attribute is equal to the split value, and the second list
    #   has all the rows where the split attribute is not equal to the split value.
    #   Also create two lists(y_left and y_right) with the corresponding y labels.

    '''
    Example:

    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]

    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.

    Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
    Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.

    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]

    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.

    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]

    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]

    '''

    #Attempt to make it quicker

    # #Check if split value is string
    # if(isinstance(split_val, str)):
    #     #Indexes of rows in X that row[split_attribute]==split_val
    #     index_of_eq = [i for (i,row) in enumerate(X) if row[split_attribute]==split_val]
    #     #The remaining indexes
    #     diff = list(set(range(len(X))) - set(index_of_eq))

    #     X = np.array(X)
    #     y = np.array(y)
    #     #Split to left and right
    #     X_left = X[index_of_eq,:]
    #     y_left = y[index_of_eq]

    #     X_right = X[diff,:]
    #     y_right = y[diff]

    # #Numeric
    # else:
    #     #Indexes of rows in X that row[split_attribute]==split_val
    #     index_of_eq = [i for (i,row) in enumerate(X) if row[split_attribute] <= split_val]
    #     #The remaining indexes
    #     diff = list(set(range(len(X))) - set(index_of_eq))

    #     #Split to left and right
    #     X = np.array(X)
    #     y = np.array(y)
    #     X_left = X[index_of_eq,:]
    #     y_left = y[index_of_eq]

    #     X_right = X[diff,:]
    #     y_right = y[diff]


    # return (X_left, X_right, y_left, y_right)


    X_left = []
    X_right = []

    y_left = []
    y_right = []


    if is_number(split_val):
        #Numeric Values
        #print("Number",split_val)

        #Check if the element at the split attribute column is less than the split value
        for i,row in enumerate(X):
            if float(row[split_attribute]) <= float(split_val):
                X_left.append(row)
                y_left.append(y[i])
            else:
                X_right.append(row)
                y_right.append(y[i])
    else:
        #String Values
        #print("String", split_val)

        #Check if the element at the split attribute column is equal to the split value
        for i,row in enumerate(X):
            if str(row[split_attribute]) == str(split_val):
                X_left.append(row)
                y_left.append(y[i])
            else:
                X_right.append(row)
                y_right.append(y[i])

    return (X_left, X_right, y_left, y_right)


def information_gain(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels (0's and 1's)
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value

    # TODO: Compute and return the information gain from partitioning the previous_y labels
    # into the current_y labels.
    # You will need to use the entropy function above to compute information gain
    # Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf

    """
    Example:

    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]

    info_gain = 0.45915
    """
    number_item = sum([len(list_) for list_ in current_y])
    sums = 0
    for i in current_y:
        sums += -len(i)/number_item*entropy(i)
    info_gain = entropy(previous_y) + sums

    return info_gain


