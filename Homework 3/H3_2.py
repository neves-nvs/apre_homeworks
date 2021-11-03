

import matplotlib.pyplot as plt

import pandas as pd
from scipy.io import arff


GROUPN = 95


def quest2():
    # Extract Data
    D_breast = pd.DataFrame( arff.loadarff( "breast.w.arff" )[0] )
    # Elements array
    X = D_breast.drop(columns=D_breast.columns[-1]).to_numpy().astype(int)
    # Results array binarized
    Y = D_breast[D_breast.columns[-1]].replace(b'benign', 0).replace(b'malignant', 1)

def quest3():
    # Extract Data
    D_kin = pd.DataFrame( arff.loadarff( "kin8nm.arff" )[0] )
    # Elements array
    X = D_kin.drop(columns=D_kin.columns[-1]).to_numpy().astype(int)
    # Results array binarized
    # Y = D_kin[D_kin.columns[-1]].replace(b'benign', 0).replace(b'malignant', 1)

    print(D_kin)
    print("--------------------")
    print(X)

quest2()
quest3()