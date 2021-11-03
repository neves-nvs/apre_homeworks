
import pandas as pd

import matplotlib as plt
import numpy as np

from scipy.io import arff

from sklearn.neural_network import MLPClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

# LEARNING_RATE = 0.001
GROUPN = 95

knn = KFold(n_splits=5, random_state=GROUPN, shuffle=True)


def quest2():
    # Extract Data
    D_breast = pd.DataFrame( arff.loadarff( "breast.w.arff" )[0] )
    # Elements array
    X = D_breast.drop(columns=D_breast.columns[-1]).to_numpy().astype(int)
    # Results array binarized
    Y = D_breast[D_breast.columns[-1]].replace(b'benign', 0).replace(b'malignant', 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=GROUPN)

    clf = MLPClassifier(random_state=GROUPN)
    clf.fit(X_train, Y_train)

    Y_test_predict = clf.predict(X_test)

    conf_matrix = confusion_matrix(Y_test, Y_test_predict)
    
    print(Y_test_predict)
    print(conf_matrix)




def quest3():
    # Extract Data
    D_kin = pd.DataFrame( arff.loadarff( "kin8nm.arff" )[0] )
    # Elements array
    X = D_kin.drop(columns=D_kin.columns[-1]).to_numpy()
    # Results array binarized
    Y = D_kin[D_kin.columns[-1]].to_numpy()

    print(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=GROUPN)

    clf = MLPClassifier(random_state=GROUPN)
    clf.fit(X_train, Y_train)

    # Y_test_predict = clf.predict(X_test)

    # residuals = np.subtract(Y_test, Y_test_predict)
    # print(residuals, Y_test_predict)
    # plt.scatter(residuals, Y_test_predict)

# quest2()
quest3()