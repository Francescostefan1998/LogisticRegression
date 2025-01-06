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

# for i in range(len(X_train_std)):
#     print(f'Printing the {i} sample : {X_train_std[i]} label: {y_combined[i]}')

# for e in range(len(X_test_std)):
#     print(f'Printing the {e} sample : {X_test_std[e]} label: {y_combined[e + len(X_train_std)]}')

# print(f'X combined std : {X_combined_std}')
# print(f'y combined : {y_combined}')
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
# # print("Model coefficients:")
# # #print(lr.coef_)
# # print("Model intercept:")
# #print(lr.intercept_)
# lr.fit(X_train_std, y_train)
# lr.predict_proba(X_test_std[:3, :])
# print(f'Extracted values to test the probability {X_test_std[:3, :]}')
# print(f'Predict proba {lr.predict_proba(X_test_std[:3, :])}')
# # test if the sum for each line is equal to 1
# print(f'Sum: {lr.predict_proba(X_test_std[:3, :]).sum(axis= 1)}')

# print(f'ARG MAX: {lr.predict_proba(X_test_std[:3, :]).argmax(axis= 1)}') # check the max probability

# print(f'Obtaining the same by using the predict method {lr.predict(X_test_std[:3, :])}')

# # in order to take a single check the following example
# print(f'Single row: {lr.predict(X_test_std[0, :].reshape(1, -1))}')
# # original_coef = lr.coef_.copy()
# # original_intercept = lr.intercept_.copy()
# # print("Original Coefficients:\n", original_coef)
# # print("Original Intercept:\n", original_intercept)

# # Example: Scale the coefficients by 0.5
# # modified_coef = original_coef * 8
# # modified_intercept = original_intercept # + 50  # Offset intercept

# # lr.coef_ = modified_coef
# # lr.intercept_ = modified_intercept


# plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# weights, params = [], []
# # the following c parameter as it increse it should decrease the strenght of the regularization function
# for c in np.arange(-5, 5):
#     lr = LogisticRegression(C=10.**c, multi_class='ovr')
#     print(f'C value : {10.**c}')
#     lr.fit(X_train_std, y_train)
#     weights.append(lr.coef_[1])
#     params.append(10.**c)

# weights = np.array(weights)
# plt.plot(params, weights[:, 0], label='Petal length')
# plt.plot(params, weights[:, 1], linestyle='--', label='Petal width')
# plt.ylabel('Weight coefficient')
# plt.xlabel('C')
# plt.legend(loc='upper left')
# plt.xscale('log')
# plt.show()

from sklearn.svm import SVC
# svm = SVC(kernel = 'linear', C=1.0, random_state=1)
# svm.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='royalblue', marker='s',
            label='Class 1')
plt.scatter(X_xor[y_xor == 0, 0],
            X_xor[y_xor == 0, 1],
            c='tomato', marker='o',
            label='Class 0')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
# svm.fit(X_xor, y_xor)
# plot_decision_regions(X_xor, y_xor, classifier=svm)
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
# svm.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
# svm.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# def entroypy(p):
#     return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

# x = np.arange(0.0, 1.0, 0.01)
# ent = [entroypy(p) if p != 0 else None for p in x]
# plt.ylabel('Entropy')
# plt.xlabel('Class-membership probability p(i = 1)')
# plt.plot(x, ent)
# plt.show()



from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree_model, test_idx=range(105, 150))
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()