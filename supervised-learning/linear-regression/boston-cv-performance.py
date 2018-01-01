# This example use the boston dataset to illustrate how to use the
# LinearRegression linear model
# Outcome: A plot of prediction score vs different value of folds

# Import necessary modules
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Load the digits dataset: digits
boston_ds = datasets.load_boston()

# set X and y
X = boston_ds.data
y = boston_ds.target

# reshape data
y = y.reshape(-1, 1)

# create linear regressor
reg = LinearRegression()

# test cv range 2 to 15
cv_range = range(2, 16)

scores = []

for cv in cv_range:
    cv_scores = cross_val_score(reg, X, y, cv=cv)
    cv_mean = np.mean(cv_scores)

    print(cv_scores)
    scores.append(cv_mean)

plt.plot(cv_range,
         scores,
         color='black',
         linewidth=3)

plt.xlabel('CV value')
plt.ylabel('Mean Score')
plt.show()
