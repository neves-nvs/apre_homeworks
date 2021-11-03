
import pandas as pd

import matplotlib.pyplot as plt

from scipy.io import arff
from scipy.sparse.construct import random

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest


GROUPN = 95

# Extract Data
data = pd.DataFrame( arff.loadarff( "breast.w.arff" )[0] )
# Elements array
X = data.drop(columns=data.columns[-1]).to_numpy().astype(int)
# Results array binarized
Y = data[data.columns[-1]].replace(b'benign', 0).replace(b'malignant', 1)

knn = StratifiedKFold(n_splits=10, random_state=GROUPN, shuffle=True)

def q5_1():
      
    train_accuracy_list, test_accuracy_list= list(), list()
        
    for k_val in [1, 3, 5, 9]:

        KBest = SelectKBest(mutual_info_classif, k=k_val).fit_transform(X, Y)

        clf = DecisionTreeClassifier(random_state=GROUPN)

        accuracy = cross_validate(clf, KBest, Y, scoring="accuracy", return_train_score=True)

        train_accuracy_list.append(sum(accuracy["train_score"])/len(accuracy["train_score"]))
        test_accuracy_list.append(sum(accuracy["test_score"])/len(accuracy["test_score"]))

    plt.plot(range(4), train_accuracy_list,label="Limited Features [Train]", linestyle="--", marker="o")
    plt.plot(range(4), test_accuracy_list,label="Limited Features [Test]", linestyle="--", marker="o")

    

def q5_2():

    train_accuracy_list, test_accuracy_list= list(), list()

    for k_val in [1, 3, 5, 9]:

        clf = DecisionTreeClassifier(max_depth=k_val, random_state=GROUPN)

        accuracy = cross_validate(clf, X, Y, scoring="accuracy", return_train_score=True)

        train_accuracy_list.append(sum(accuracy["train_score"])/len(accuracy["train_score"]))
        test_accuracy_list.append(sum(accuracy["test_score"])/len(accuracy["test_score"]))


    plt.plot(range(4), train_accuracy_list,label="Limited Depth [Train]", linestyle="--", marker="v")
    plt.plot(range(4), test_accuracy_list,label="Limited Depth [Test]", linestyle="--", marker="v")

q5_1()
q5_2()

plt.title('Mean Accuracy vs (Number of Features|Tree Depth)')
plt.xlabel('● - Number Selected Features | ▼ - Tree depth')
plt.xticks(range(4), [1, 3, 5, 9])
plt.ylabel('Mean Accuracy')
plt.savefig("graph.png")
plt.show()