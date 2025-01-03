import numpy as np

class LogisticRegressionGD:


# We must consider that wehn we fit a logistic regression model, we have to keep in mind that it only works for binary classification tasks.
# So we consider just classes 0 and 1
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale= 0.01, size = X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X) # dotproduct
            output = self.activation(net_input) 
            errors = (y-output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) /X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output))- ((1-y).dot(np.log(1-output)))/X.shape[0])
            self.losses_.append(loss)
        
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    # the value 250 is just to limit the number otherwise if I don't limit it it could go towards infinity leading to errors
    # so the clip function will simply reduce all the numbers that exceed this range
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
