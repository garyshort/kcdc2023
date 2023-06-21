from sklearn import datasets, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the digits dataset
digits = datasets.load_digits()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

# Hyperparameters Grid
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# Variables to store the best parameters and the best accuracy
best_params = None
best_accuracy = 0

# Grid search
for params in param_grid:
    if params['kernel'] == ['linear']:
        params['gamma'] = [0]  # gamma doesn't matter for linear kernel
    for gamma in params['gamma']:
        for C in params['C']:
            clf = svm.SVC(kernel=params['kernel'][0], C=C, gamma=gamma)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'kernel': params['kernel'][0], 'C': C, 'gamma': gamma}

print(f"Best parameters: {best_params}")
print(f"Best accuracy: {best_accuracy}")
