import numpy as np

class Online_Parameter_Calculation():
    def __init__(self):
        self.n_1 = 0
        self.running_sum = 0
        self.s_1 = 0
        self.mu_1 = 0

    def add_next_set(self, X, p = lambda x: True):
        indices = p(X)
        Y = X[indices]

        if self.n_1 > 0:
            self.mu_1 = self.running_sum / self.n_1
        else:
            self.mu_1 = 0

        self.running_sum += np.sum(Y)
        n_2 = np.sum(indices)

        if self.n_1 + n_2 > 0:
            mu_2 = np.mean(Y)
            mu = self.running_sum / (self.n_1 + n_2)
            s_2 = np.std(Y)

            self.s_1 = np.sqrt(( (self.n_1-1)*self.s_1**2 + \
                            (n_2-1)*s_2**2 + \
                             self.n_1*(self.mu_1-mu)**2 + \
                             n_2*(mu_2-mu)**2 ) / (self.n_1 + n_2 - 1))

            self.n_1 += n_2
            self.mu_1 = mu

    def get_mean(self):
        return self.mu_1

    def get_std(self):
        return self.s_1