import numpy as np
from sklearn import datasets
from LogisticRegressionGD import LogisticRegressionGD
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1, stratify= y
)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
# y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
# print(f'X subset : {X_train_01_subset}')
# print(f'y labels : {y_train_01_subset}')
# lrgd = LogisticRegressionGD(eta = 0.3, n_iter = 1000, random_state = 1)

def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^','v','<')
    colors  = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print(lab)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y== cl, 0],
                    y = X[y == cl, 1],
                    alpha = 0.8,
                    c=colors[idx],
                    marker = markers[idx],
                    label = f'Class {cl}',
                    edgecolor = 'black')
        
    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test set')
    

# lrgd.fit(X_train_01_subset, y_train_01_subset)
# plot_decision_regions(X = X_train_01_subset, y=y_train_01_subset, classifier = lrgd)


# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc = 'upper left')
# plt.tight_layout()
# plt.show()


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

for i in range(len(X_train_std)):
    print(f'Printing the {i} sample : {X_train_std[i]} label: {y_combined[i]}')

for e in range(len(X_test_std)):
    print(f'Printing the {e} sample : {X_test_std[e]} label: {y_combined[e + len(X_train_std)]}')

# print(f'X combined std : {X_combined_std}')
# print(f'y combined : {y_combined}')
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0, solver='lbfgs')
print("Model coefficients:")
#print(lr.coef_)
print("Model intercept:")
#print(lr.intercept_)
lr.fit(X_train_std, y_train)
lr.predict_proba(X_test_std[:3, :])
print(f'Extracted values to test the probability {X_test_std[:20, :]}')
print(f'Predict proba {lr.predict_proba(X_test_std[:20, :])}')
original_coef = lr.coef_.copy()
original_intercept = lr.intercept_.copy()
print("Original Coefficients:\n", original_coef)
print("Original Intercept:\n", original_intercept)

# Example: Scale the coefficients by 0.5
# modified_coef = original_coef * 8
# modified_intercept = original_intercept # + 50  # Offset intercept

# lr.coef_ = modified_coef
# lr.intercept_ = modified_intercept


plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# still not working 
