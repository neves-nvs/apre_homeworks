from scipy.io import arff
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

#KNN SEED
GROUPN = 39
# Extract Data
data = pd.DataFrame( arff.loadarff( "breast.w.arff" )[0] )
# Elements array
X = data.drop(columns=data.columns[-1]).to_numpy().astype(int)
# Results array
Y = data[data.columns[-1]].replace(b'benign', 0).replace(b'malignant', 1)

knn = StratifiedKFold(n_splits=10, random_state=GROUPN, shuffle=True)

def quest6():

    for k in [3, 5, 7]:
        knn_model = KNeighborsClassifier(n_neighbors=k)  # p=2, weights='uniform', metric='minkowski' are already defaults

        train_errors, test_errors = list(), list()

        for train_index, test_index in knn.split(X, Y):
            X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

            knn_model.fit(X_train, Y_train)

            Y_train_predict = knn_model.predict(X_train)
            Y_test_predict = knn_model.predict(X_test)

            fold_train_error = mean_absolute_error(Y_train, Y_train_predict)
            fold_test_error = mean_absolute_error(Y_test, Y_test_predict)

            train_errors.append(fold_train_error)
            test_errors.append(fold_test_error)

        # Calculate mse for each k, lower value -> less risk of overfitting  
        err = mean_squared_error(train_errors, test_errors)    
        print(f"Mean square error: {err} (k = {k})")


def quest7():
    
    knn_model = KNeighborsClassifier(n_neighbors=3) # p=2, weights='uniform', metric='minkowski' are already defaults
    nbayes_model = MultinomialNB()

    knnScore, nbayesScore = list(), list()

    for train_index, test_index in knn.split(X, Y):
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

        knn_model.fit(X_train, Y_train); nbayes_model.fit(X_train, Y_train)

        knnScore.append(knn_model.score(X_test, Y_test)); nbayesScore.append(nbayes_model.score(X_test, Y_test))

    stat, p_value = ttest_rel(knnScore, nbayesScore, alternative='greater'); print(f"p Value: {p_value}")

quest6()
quest7()