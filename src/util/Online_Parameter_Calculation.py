import numpy as np

class Online_Parameter_Calculation():

    __slots__ = ['_n_1',
                 '_running_sum',
                 '_s_1',
                 '_mu_1',
                 ]

    def __init__(self):
        self._n_1 = 0
        self._running_sum = 0
        self._s_1 = 0
        self._mu_1 = 0

    def add_next_set(self, X, p = lambda x: True):
        indices = p(X)
        Y = X[indices]

        if self._n_1 > 0:
            self._mu_1 = self._running_sum / self._n_1
        else:
            self._mu_1 = 0

        self._running_sum += np.sum(Y)
        n_2 = np.sum(indices)

        if self._n_1 + n_2 > 0:
            mu_2 = np.mean(Y)
            mu = self._running_sum / (self._n_1 + n_2)
            s_2 = np.std(Y)

            self._s_1 = np.sqrt(( (self._n_1-1)*self._s_1**2 + \
                            (n_2-1)*s_2**2 + \
                             self._n_1*(self._mu_1-mu)**2 + \
                             n_2*(mu_2-mu)**2 ) / (self._n_1 + n_2 - 1))

            self._n_1 += n_2
            self._mu_1 = mu

    @property
    def mean(self):
        return self._mu_1

    @property
    def std(self):
        return self._s_1