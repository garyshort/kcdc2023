from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score

# Load iris dataset as an example
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a SVC classifier
clf = svm.SVC(kernel='linear', C=1, random_state=42)

# Perform 5-fold cross validation
scores = cross_val_score(clf, X, y, cv=5)

print("Cross-validation scores: ", scores)
print("Mean cross-validation score: ", scores.mean())
