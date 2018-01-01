# This example use the boston dataset to illustrate how to use the
# LinearRegression linear model
# Outcome: A plot of prediction value and actual value

# Import necessary modules
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import numpy as np
import matplotlib.pyplot as plt

# Load the boston dataset
boston_ds = datasets.load_boston()

# set X and y
X = boston_ds.data
y = boston_ds.target

# reshape data
y = y.reshape(-1, 1)

# create linear regressor
reg = LinearRegression()

# get the cross validation score
cv_scores = cross_val_score(reg, X, y, cv=6)
print(cv_scores)
print(np.mean(cv_scores))

# plot the actual and predicted value
y_pred = cross_val_predict(reg, X, y, cv=6)

plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)

plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.show()
