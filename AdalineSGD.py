import numpy as np

class perceptron():
    
    def __init__(self, niter, shuffle_=False, deta=0.01, random_sate=None):
        self.niter = niter
        self.deta = deta
        self.random_sate = random_sate
        self.w_init = False
        self.shuffle_ = shuffle
    
    def fit(self, X, y):
        self.init_weight(X.shape[1])
        self.cost = []

        for count in range(self.niter):
            if self.shuffle_:
                X,y = self.shuffle(X, y)
            cost = []
        for Xi, target in zip(X, y):
            cost.append(self.update_weight(Xi, target))
        avg_cost = sum(cost) / len(y)
        self.cost.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_init:
            self.init_weight(X.shape[1])
        if y.ravel().shape[0] >1:
            for Xi, target in zip(X, y):
                self.update_weight(Xi, target)
            else:
                self.update_weight(X, y)
        return self
    
    def shuffle(self, X, y):
        r = self.rng.permutation(len(y))
        return X[r], y[r]


    def init_weight(self, m):
        self.rng = np.random.RandomState(self.random_sate)
        self.w_ = self.rng.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_init = True
    
    def update_weight(self, Xi, y):
        net_input = self.net_input(Xi)
        output = self.activation(net_input)
        errors = (y-output)
        self.w_[1:] = self.deta*Xi.dot(errors)
        self.w_[0] = self.deta*errors
        cost = 0.5 * errors **2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]    
    
    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >=0.0, 1, -1)
