# This example use the boston dataset to illustrate how to use the
# LinearRegression linear model
# Outcome: A scatter plot of training and testing data (only using number of
# rooms and price)

# Import necessary modules
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load the digits dataset: digits
boston_ds = datasets.load_boston()

# set X and y
X = boston_ds.data
y = boston_ds.target

# extract the number of rooms column
X_rooms = X[:, 5].reshape(-1, 1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_rooms, y,
                                                    test_size=0.3,
                                                    random_state=42)

# Create the regressor: reg
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

# Print R^2
print(reg.score(X_test, y_test))

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# Plot the training set scatter in blue
ax1.scatter(X_train, y_train, c='b')

# Plot the testing set scatter in red
ax1.scatter(X_test, y_pred, c='r')

# Create the prediction space
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)

# On the same graph, plot regression line
plt.plot(prediction_space,
         reg.predict(prediction_space),
         color='black',
         linewidth=3)

plt.xlabel('Number of Rooms')
plt.ylabel('Price')
plt.show()
