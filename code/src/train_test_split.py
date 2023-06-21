from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Now X_train and y_train are used for training the model, and X_test and
#  y_test are used for testing it. # For example, let's train a simple logistic
# regression model

# Create a model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Test the model
score = model.score(X_test, y_test)
print(f"Model accuracy: {score*100:.2f}%")
