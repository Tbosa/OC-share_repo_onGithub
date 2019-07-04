import numpy as np


class adalineGD:

    def __init__(self, lrate=0.01, n_epochs=20, random_state=1):
        self.lrate = lrate
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):

        nb_gen = np.random.RandomState(self.random_state)
        self.w_ = nb_gen.normal(loc=0, scale=0.01, size=X.shape[1] + 1)

        self.SSE = list()

        for ep in range(self.n_epochs):
            nets = self.net_input(X)
            outputs = self.activation(nets)
            errors = (y - outputs)
            self.w_[1:] += self.lrate * (X.T.dot(errors))
            self.w_[0] += self.lrate * errors.sum()
            cost_function = 0.5*((errors**2).sum())

            self.SSE.append(cost_function)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, nets):
        return nets

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
