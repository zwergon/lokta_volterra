import numpy as np
from scipy.special import erf


class Law(object):

    def __init__(self, params):
        self.p = params

    def pdf(self, x):
        pass

    def cdf(self, x):
        pass

    def extrema(self):
        pass


class NormalLaw(Law):

    def __init__(self, mu, sigma):
        super(NormalLaw, self).__init__([mu, sigma])

    def pdf(self, x):
        norm = np.sqrt(2*np.pi)*self.p[1]
        return np.exp(-.5*((x-self.p[0])/self.p[1])**2) / norm

    def cdf(self, x):
        return .5 * (1 + erf((x - self.p[0]) / self.p[1] / np.sqrt(2)))

    def extrema(self):
        return self.p[0] - 100 * self.p[1], self.p[0] + 100 * self.p[1]


class BernouilliLaw(Law):

    def __init__(self, lbda):
        super(BernouilliLaw, self).__init__([lbda])

    def pdf(self, x):
        """x can only be 1 or O"""
        return self.p[0] if x == 1 else (1. - self.p[0])

    def cdf(self, k):
        if k < 0:
            return 0.
        elif 0 <= k < 1:
            return 1 - self.p[0]
        else:
            return 1

    def extrema(self):
        return -10, 10
