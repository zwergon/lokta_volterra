import random
from stats.law import Law


class Sampler1D:

    def __init__(self, law: Law):
        self.law = law

    def sample(self, n_samples):
        samples = []
        for i in range(n_samples):
            x = random.random()
            s = self._find_x(x)
            samples.append(s)

        return samples

    def _find_x(self, x):

        a, b = self.law.extrema()
        v1 = self.law.cdf(a)

        while b - a > .01:
            c = .5 * (a + b)
            v = self.law.cdf(c)
            if (x - v) * (x - v1) > 0:
                a = c
            else:
                b = c

        return c