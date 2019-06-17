import numpy as np

class perceptron():
    
    def __init__(self, niter, deta=0.01, random_sate=1):
        self.niter = niter
        self.deta = deta
        self.random_sate = random_sate
    
    def fit(self, X, y):
        rng = np.random.RandomState(self.random_sate)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.error = []

        for count in range(self.niter):
            errors = 0

            for a, b in zip(X, y):               
                update = self.deta * (b - self.predict(a))
                self.w_[1:] += update * a
                self.w_[0] += update
                errors += int(update != 0)  
            
            self.error.append(errors)
        
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]    
    
    def predict(self, X):
        return np.where(self.net_input(X) >=0.0, 1, -1)
