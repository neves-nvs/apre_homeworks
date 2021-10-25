#!/usr/local/bin/python3

import pandas as pd
import numpy as np

from scipy.io import arff
from scipy.sparse.construct import random

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import mean_absolute_error, mean_squared_error

GROUPN = 39

# Extract Data
data = pd.DataFrame( arff.loadarff( "breast.w.arff" )[0] )
# Elements array
X = data.drop(columns=data.columns[-1]).to_numpy().astype(int)
# Results array binarized
Y = data[data.columns[-1]].replace(b'benign', 0).replace(b'malignant', 1)

def q5_1():
     
    train_errors_list, test_errors_list = list(), list()  
        
    for k_val in [1, 3, 5, 9]:

        KBest = SelectKBest(mutual_info_classif, k=k_val).fit_transform(X, Y)
        
        X_train, X_test, Y_train, Y_test = train_test_split(KBest, Y, random_state=GROUPN)
        
        clf = DecisionTreeClassifier()

        clf.fit(X_train, Y_train)

        Y_train_predict = clf.predict(X_train)
        Y_test_predict = clf.predict(X_test)

        train_error = mean_absolute_error(Y_train, Y_train_predict)
        test_error = mean_absolute_error(Y_test, Y_test_predict)

        train_errors_list.append(train_error)
        test_errors_list.append(test_error)

        err = mean_squared_error(train_errors_list, test_errors_list)    
        print(f"Mean square error: {err} (k = {k_val})")

    print(train_errors_list, test_errors_list)
        
    

def q5_2():

    train_errors_list, test_errors_list = list(), list()  

    for k_val in [1, 3, 5, 9]:
        
        KBest = SelectKBest(mutual_info_classif, k=k_val).fit_transform(X, Y)
        
        X_train, X_test, Y_train, Y_test = train_test_split(KBest, Y, random_state=GROUPN)
        
        clf = DecisionTreeClassifier(max_depth=k_val)

        clf.fit(X_train, Y_train)

        Y_train_predict = clf.predict(X_train)
        Y_test_predict = clf.predict(X_test)

        train_error = mean_absolute_error(Y_train, Y_train_predict)
        test_error = mean_absolute_error(Y_test, Y_test_predict)

        train_errors_list.append(train_error)
        test_errors_list.append(test_error)

        err = mean_squared_error(train_errors_list, test_errors_list)    
        print(f"Mean square error: {err} (k = {k_val})")

    print(train_errors_list, test_errors_list)

q5_2()