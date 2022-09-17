
import numpy as np


def metropolis_hasting(pdf, n_samples, x0, s=100):

    x = np.array(x0)
    p = pdf(x)
    samples = []

    for i in range(n_samples):
        xn = x + np.random.normal(size=(len(x),))
        pn = pdf(xn)

        if pn > p:
            p = pn
            x = xn
        else:
            u = np.random.rand()
            if u < pn / p:
                p = pn
                x = xn

        if i > s:
            samples.append(x)

    return samples


def gaussian(x, mu, sigma):
    norm = .5 / np.pi /np.sqrt(sigma[0]*sigma[1])
    return np.exp(-.5*(x[0]-mu[0])**2 / sigma[0]**2)*np.exp(- .5*(x[1]-mu[1])**2 / sigma[1]**2)


def pdf2d_polynomial(x):
    if x[0] < 0:
        return 0.
    if x[0] > 1:
        return 0.
    if x[1] < 0:
        return 0.
    if x[1] > 1:
        return 0.

    return x[0] + 1.5 * x[1] ** 2


def compute_pdf_2d():
    import matplotlib.pyplot as plt

    x = np.linspace(-10, 10., 100)
    y = np.linspace(-10., 10., 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(shape=X.shape)
    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            Z[i, j] = gaussian( [X[i, j], Y[i, j]] , mu=[1, -2], sigma=[5., 1.])

    from functools import partial
    samples = metropolis_hasting(partial(gaussian, mu=[1, -2], sigma=[5., 1.]), 1000, x0=[0.5, 0.5])

    xs = []
    ys = []
    for xy in samples:
        xs.append(xy[0])
        ys.append(xy[1])

    plt.pcolor(X, Y, Z)
    plt.scatter(xs, ys)
    plt.show()



if __name__ == '__main__':
    compute_pdf_2d()

    #print(np.random.normal(size=(2,)))
