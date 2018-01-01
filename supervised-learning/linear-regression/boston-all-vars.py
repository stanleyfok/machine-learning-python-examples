# This example use the boston dataset to illustrate how to use the
# LinearRegression linear model
# Outcome: the regressor score and root mean square error

# Import necessary modules
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Load the digits dataset: digits
boston_ds = datasets.load_boston()

# set X and y
X = boston_ds.data
y = boston_ds.target

# reshape data
y = y.reshape(-1, 1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

# Create the regressor: reg
reg = LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
