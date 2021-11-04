
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from scipy.io import arff

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

GROUPN = 0

def quest2():
    # Extract Data
    D_breast = pd.DataFrame( arff.loadarff( "breast.w.arff" )[0] )
    # Elements array
    X = D_breast.drop(columns=D_breast.columns[-1]).to_numpy().astype(int)
    # Results array binarized
    Y = D_breast[D_breast.columns[-1]].replace(b'benign', 0).replace(b'malignant', 1)

    stratifiedk_splits = StratifiedKFold(n_splits=5, random_state=GROUPN, shuffle=True)

    clf = MLPClassifier(hidden_layer_sizes=(3, 2), random_state=GROUPN)
    Y_pred = cross_val_predict(clf, X, Y, cv=stratifiedk_splits)
    conf_matrix = confusion_matrix(Y, Y_pred)
    
    clf_es = MLPClassifier(hidden_layer_sizes=(3, 2), random_state=GROUPN, early_stopping=True)
    Y_es_pred = cross_val_predict(clf_es, X, Y, cv=stratifiedk_splits)
    conf_matrix_es = confusion_matrix(Y, Y_es_pred)

    print("Confusion matrix")
    print(conf_matrix)
    print("Confusion matrix - Early Stopping")
    print(conf_matrix_es)


def quest3():
    # Extract Data
    D_kin = pd.DataFrame( arff.loadarff( "kin8nm.arff" )[0] )
    # Elements array
    X = D_kin.drop(columns=D_kin.columns[-1]).to_numpy()

    Y = D_kin[D_kin.columns[-1]].to_numpy()

    k_splits = KFold(n_splits=5, random_state=GROUPN, shuffle=True)

    clf = MLPRegressor(alpha=0.1, random_state=GROUPN)
    Y_pred = cross_val_predict(clf, X, Y, cv=k_splits)
    residuals = np.subtract(Y, Y_pred)
    
    clf_reg = MLPRegressor(alpha=0, random_state=GROUPN)
    Y_reg_pred = cross_val_predict(clf_reg, X, Y, cv=k_splits)
    residuals_reg = np.subtract(Y, Y_reg_pred)
    
    plt.boxplot([residuals, residuals_reg], labels=("Not Regularized", "Regularized"))
    plt.savefig("graph_ex3")

quest2()
quest3()