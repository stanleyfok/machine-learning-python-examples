# This example use the boston dataset to illustrate use of Lasso linear model
# for regularization
# Outcome: A plot of feature names and the lasso coefficient values

# Import necessary modules
from sklearn import datasets
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Load the boston dataset
boston_ds = datasets.load_boston()

# set X and y
X = boston_ds.data
y = boston_ds.target

# reshape data
y = y.reshape(-1, 1)

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.1, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(boston_ds.feature_names)), lasso_coef)
plt.xticks(range(len(boston_ds.feature_names)), boston_ds.feature_names,
           rotation=60)
plt.margins(0.02)
plt.show()
