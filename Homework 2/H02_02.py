from scipy.io import arff

from sklearn import tree

# Extract Data
data = pd.DataFrame( arff.loadarff( "breast.w.arff" )[0] )
# Elements array
X = data.drop(columns=data.columns[-1]).to_numpy().astype(int)
# Results array
Y = data[data.columns[-1]].replace(b'benign', 0).replace(b'malignant', 1)


clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)

