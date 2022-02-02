from scipy.special import erf
import random
import numpy as np

def cdf(x, mu, sigma):
    return .5 * (1 + erf((x-mu)/sigma/np.sqrt(2)))

def find_x(mu, sigma, x):


    a = mu - 10*sigma
    b = mu + 10*sigma
    v1 = cdf(a, mu, sigma)
    v2 = cdf(b, mu, sigma)

    while b-a > .01:
        c = .5*(a+b)
        v = cdf(c, mu, sigma)
        if (x-v)*(x-v1) > 0:
            a= c
        else:
            b= c

    return c

def repartition(n_samples, mu, sigma):

    samples = []
    for i in range(n_samples):
        x = random.random()
        s = find_x(mu, sigma, x)
        samples.append(s)

    return samples