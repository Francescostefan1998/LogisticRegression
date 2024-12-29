import numpy as np
from sklearn import datasets
from LogisticRegressionGD import LogisticRegressionGD
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, solver='lbfgs')

sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Generate example data
np.random.seed(42)  # For reproducibility
X_class_0 = np.random.normal(loc=[1, 1], scale=0.5, size=(5, 2))  # Class 0
X_class_1 = np.random.normal(loc=[3, 3], scale=0.5, size=(5, 2))  # Class 1
X_class_2 = np.random.normal(loc=[5, 1], scale=0.5, size=(5, 2))  # Class 2

# Combine into one dataset
X = np.vstack((X_class_0, X_class_1, X_class_2))

# Labels for the samples
y = np.array([0] * 5 + [1] * 5 + [2] * 5)

# Visualize the data
plt.scatter(X[:5, 0], X[:5, 1], color='red', label='Class 0')
plt.scatter(X[5:10, 0], X[5:10, 1], color='blue', label='Class 1')
plt.scatter(X[10:, 0], X[10:, 1], color='green', label='Class 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Create and fit the logistic regression model
model = LogisticRegression(multi_class='ovr')  # One-vs-rest for multi-class classification
model.fit(X, y)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
