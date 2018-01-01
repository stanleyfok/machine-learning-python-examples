# This example use the digits dataset to illustrate how to use GridSearchCV
# Outcome: the best param setting and best score

# Import necessary modules
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

# Load the digits dataset
digits = datasets.load_digits()

# Create feature and target arrays
X = digits.data
y = digits.target

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Create a LogisticRegression
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit the classifier to the training data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}"
      .format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))
