
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import numpy as np

from scipy.io import arff

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

import pprint


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
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans = kmeans.fit(X)
    Y_pred = kmeans.predict(X)

    # Get elements with only their 2 most meaningful features
    Kbest = SelectKBest(mutual_info_classif, k=2)
    Kbest = Kbest.fit_transform(X, Y)
    # Separate KBest by their output class

    def marker(i):
        if Y_pred[i] == 0: return "o"
        elif Y_pred[i] == 1: return "^"
        elif Y_pred[i] == 2: return "D"
        else: print("Bad classification")

    def color(i):
        if Y[i] == 0: return "green"
        elif Y[i] == 1: return "red"
        else: print("Data missing")
   
    for i, point in enumerate(Kbest):
        plt.scatter(point[0], point[1], alpha=0.2, marker=marker(i), color=color(i))

    cluster_centers = kmeans.cluster_centers_
    
    print(cluster_centers)
    


    plt.title("K = 3 Clustering Solution based on top 2 features")
    plt.xlabel("Cell_Size_Uniformity")
    plt.ylabel("Cell_Shape_Uniformity")

    green = mpatches.Patch(color='green', label='Malignant')
    red = mpatches.Patch(color='red', label='Benign')

    cl_1 = mlines.Line2D([], [], marker='D', color="black", linestyle='None', markersize=6, label='Cluster 1')
    cl_2 = mlines.Line2D([], [], marker='o', color="black", linestyle='None', markersize=8, label='Cluster 2')
    cl_3 = mlines.Line2D([], [], marker='^', color="black", linestyle='None', markersize=7, label='Cluster 3')

    plt.legend(handles=[green, red, cl_1, cl_2, cl_3])
    # plt.legend("hey", title="Clusters", loc=(10.5, 10.5))
    # plt.legend("oh", title="CLass", loc=(10.05, 2))

    plt.savefig("graph5")


quest4()
quest5()
