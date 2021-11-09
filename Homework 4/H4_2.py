
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from scipy.io import arff

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.feature_selection import mutual_info_classif

from sklearn.feature_selection import SelectKBest

SEED = 95

# Extract Data
D_breast = pd.DataFrame( arff.loadarff( "breast.w.arff" )[0] )
# Elements array
X = D_breast.drop(columns=D_breast.columns[-1]).to_numpy().astype(int)
# Results array binarized
Y = D_breast[D_breast.columns[-1]].replace(b'benign', 0).replace(b'malignant', 1)

def quest4():

    def Cluster(k, Y_pred):
        cluster = np.where(Y_pred == k) [0]
        return cluster

    def ECR(k, Y, Y_pred):
        soma = 0

        for i in range(k):
            cluster = Cluster(i, Y_pred)
            total = [0]*k

            for ind in cluster:
                total[Y[ind]] += 1
            phi = max(total)

            soma += (len(cluster) - phi)

        return (1/k) * soma

    for k in [2, 3]:
        # Train model
        kmeans = KMeans(n_clusters=k, random_state=SEED).fit(X)
        Y_pred = kmeans.predict(X)

        print()
        print(f"------------ K = {k} ------------")
        # Error classification rate (external measure)
        ecr = ECR(k, Y, Y_pred)
        print(f"       ECR = {ecr}")
        # Silhouette coefficient (internal measure)
        sil_score = silhouette_score(X, Y_pred, random_state=SEED)
        print(f"silhouette = {sil_score}")

def quest5():
    kmeans = KMeans(n_clusters=3, random_state=SEED)
    kmeans = kmeans.fit(X)
    Y_pred = kmeans.predict(X)

    best_feat = SelectKBest(mutual_info_classif, k=2).fit_transform(X, Y)
    print(best_feat)

    # pprint(dir(best_feat))
    # print(best_feat)

    # for feature in range(2):
    #     plt.scatter(feature, Y2_pred)

    plt.savefig("graph5")

quest4()
# quest5()
