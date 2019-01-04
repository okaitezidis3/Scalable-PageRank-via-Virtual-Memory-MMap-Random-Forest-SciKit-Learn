## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect seizure

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize

######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
data = pd.read_csv('seizure_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the paramater 'shuffle' set to true and the 'random_state' set to 100.
# XXX
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=random_state, shuffle = True)


# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX
start_Linear_Regression = time.time()
reg = LinearRegression().fit(x_train, y_train)
end_Linear_Regression = time.time()
print("The fitting time of the Linear Regression is",end_Linear_Regression - start_Linear_Regression)

# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Use y_predict.round() to get 1 or 0 as the output.
# XXX

reg_y_train_pred = reg.predict(x_train)
reg_y_train_pred = reg_y_train_pred.round()
print("The accuracy of the linear regression for the train set is {0:.0%}".format(accuracy_score(y_train, reg_y_train_pred)))

reg_y_test_pred = reg.predict(x_test)
reg_y_test_pred = reg_y_test_pred.round()
print("The accuracy of the linear regression for the test set is {0:.0%}".format(accuracy_score(y_test, reg_y_test_pred)))


# ############################################### Multi Layer Perceptron #################################################
# XXX
# TODO: Create an MLPClassifier and train it.
# XXX


# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
start_MLP_Classifier = time.time()
MLP = MLPClassifier().fit(x_train, y_train)
end_MLP_Classifier = time.time()
print("The fitting time of the MLP Classifier is",end_MLP_Classifier - start_MLP_Classifier)

MLP_y_train_pred = MLP.predict(x_train)
MLP_y_train_pred = MLP_y_train_pred.round()
print("The accuracy of the MLP Classifier for the train set is {0:.0%}".format(accuracy_score(y_train, MLP_y_train_pred)))

MLP_y_test_pred = MLP.predict(x_test)
MLP_y_test_pred = MLP_y_test_pred.round()
print("The accuracy of the MLP Classifier for the test set is {0:.0%}".format(accuracy_score(y_test, MLP_y_test_pred)))




# ############################################### Random Forest Classifier ##############################################
# TODO: Create a RandomForestClassifier and train it.
RFR = RandomForestClassifier().fit(x_train, y_train)

# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# RFR_y_train_pred = RFR.predict(x_train)
RFR_y_train_pred = RFR.predict(x_train)
print("The accuracy of the Random Forest Classifier for the train set is {0:.0%}".format(accuracy_score(y_train, RFR_y_train_pred)))


RFR_y_test_pred = RFR.predict(x_test)
RFR_y_test_pred = RFR_y_test_pred.round()
print("The accuracy of the Random Forest Classifier for the test set is {0:.0%}".format(accuracy_score(y_test, RFR_y_test_pred)))


# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.

n_estimators = [int(x) for x in np.linspace(start = 1, stop = 150, num = 3)] #1,75,150
max_depth = [int(x) for x in np.linspace(start = 3, stop = 15, num = 3)] #3,9,15

# Random grid
random_grid = {'n_estimators': n_estimators,'max_depth': max_depth}

RFR = RandomForestClassifier()
RF_random = GridSearchCV(RFR, random_grid, cv = 10, return_train_score= True)
RF_random.fit(x_data, y_data)
display_results_RF = pd.DataFrame.from_dict(RF_random.cv_results_)

print("The best parameters of the hyper tuned Random Forest Classifier are ", RF_random.best_params_)
print("The best score of the hyper tuned Random Forest Classifier is {0:.0%}".format(RF_random.best_score_))
print("The results for the RF cross validation are")
print("")
print(display_results_RF[["mean_fit_time","mean_train_score","mean_test_score","param_n_estimators","param_max_depth"]])



# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX
#Scaling of the x_train and x_test data by first fitting the x_train
scaler = StandardScaler()
scaler.fit(x_train)
scaled_x_train = scaler.transform(x_train)
scaled_x_test = scaler.transform(x_test)
# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
SVM_obj = SVC(gamma='auto')
SVM = SVM_obj.fit(scaled_x_train, y_train)
SVC_train_pred = SVM.predict(scaled_x_train)
SVC_test_pred = SVM.predict(scaled_x_test)

print("The accuracy score of the training set with SVM is {0:.0%}".format((accuracy_score(y_train, SVC_train_pred))))
print("The accuracy score of the test set with SVM is {0:.0%}".format((accuracy_score(y_test, SVC_test_pred))))

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
c_param_range = [10**ex for ex in range(-2,1)] #0.01, 0.1, 1
param_grid = {'C': c_param_range,'kernel': ['rbf','linear']}

gs = GridSearchCV(estimator=SVM_obj,param_grid=param_grid,cv=10, return_train_score= True)
scaled_x_data = scaler.transform(x_data)
gs = gs.fit(scaled_x_data, y_data)
display_results_SVM = pd.DataFrame.from_dict(gs.cv_results_)
print("The best parameters of the hyper-tuned SVM are ", gs.best_params_)
print("The best score of the hyper-tuned SVM is {0:.0%}".format((gs.best_score_)))
print("The results for the SVM cross validation are")
print("")
print(display_results_SVM[["mean_fit_time","mean_train_score","mean_test_score","param_kernel","param_C"]])
