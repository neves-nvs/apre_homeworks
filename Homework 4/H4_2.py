
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from scipy.io import arff

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
# from sklearn.metrics import #ECR

SEED = 95

def quest4():
    # Extract Data
    D_breast = pd.DataFrame( arff.loadarff( "breast.w.arff" )[0] )
    # Elements array
    X = D_breast.drop(columns=D_breast.columns[-1]).to_numpy().astype(int)
    # Results array binarized
    Y = D_breast[D_breast.columns[-1]].replace(b'benign', 0).replace(b'malignant', 1)

    kmeans_2 = KMeans(n_clusters=2, random_state=SEED)
    kmeans_2 = kmeans_2.fit(X)
    Y2_pred = kmeans_2.predict(X)

    sil_score2 = silhouette_score(X, Y2_pred, random_state=SEED)
    print(sil_score2)
    
    
    kmeans_3 = KMeans(n_clusters=2, random_state=SEED)
    kmeans_3 = kmeans_3.fit(X)



def quest5():
    pass

def quest6():
    pass

quest4()
quest5()
quest6()
