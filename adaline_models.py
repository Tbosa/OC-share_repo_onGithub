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

#############################################################################
#############################################################################

class adalineSGD:

    def __init__(self, lrate=0.01, n_epochs=20, shuffle=True, random_state=1):
        self.lrate = lrate
        self.n_epochs = n_epochs
        self._shuffle = shuffle
        self.random_state = random_state

    def shuffle(self, X, y):
        """ Ré-ordonne aléatoirement les points de données (xi, yi) pour un calcul non-biaisé du gradient (SGD) """
        idx_seq_gen = self.nb_gen.permutation(len(y))

        return X[idx_seq_gen, :], y[idx_seq_gen]

    def initialize_weights(self, p):
        """ Initialisation de w_(p+1) avec des valeurs proches de 0 issues d'une loi normale """
        self.nb_gen = np.random.RandomState(self.random_state)
        self.w_ = self.nb_gen.normal(loc=0, scale=0.01, size=p+1)

        self.w_initialized = True
        print(f'weights vector initialized ? {self.w_initialized}', '\n')

    def fit(self, X, y):

        self.SSE = list()
        self.initialize_weights(X.shape[1])

        print(self.w_, '\n')

        for ep in range(self.n_epochs):
            print(ep, '\n')
            if self._shuffle:
                X_shuffled, y_shuffled = self.shuffle(X, y)

            SSE_toavg = list()

            for xi, yi in zip(X_shuffled, y_shuffled):

                net_input = self.net_input(xi)
                output = self.activation(net_input)
                error = (yi - output)
                self.w_[1:] += self.lrate * (error*xi)
                self.w_[0] += self.lrate * error

                SSE = 0.5*(error**2)

                SSE_toavg.append(SSE)

            self.SSE.append(sum(SSE_toavg)/len(SSE_toavg))

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, nets):
        return nets

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
