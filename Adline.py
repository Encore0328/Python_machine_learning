import numpy as np

class perceptron():
    
    def __init__(self, niter, deta=0.01, random_sate=1):
        self.niter = niter
        self.deta = deta
        self.random_sate = random_sate
    
    def fit(self, X, y):
        rng = np.random.RandomState(self.random_sate)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost = []

        for count in range(self.niter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.deta * X.T.dot(errors)
            self.w_[0] += self.deta * errors.sum()
            cost = (errors**2).sum() / 2           
            self.cost.append(cost)
        
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]    
    
    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >=0.0, 1, -1)
