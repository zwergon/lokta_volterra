import numpy as np
import random
import matplotlib.pyplot as plt


def main():
    n_realisations = 1000
    n_samples = 1000
    trajectories =  np.zeros(shape=(2, n_samples, n_realisations))
    sx = 1
    sy = 1

    # Variance pour la lui uniforme comprise entre -1 et 1
    # Var(X) = Summe(Var(X)) = Somme( (b-a)**2 / 12) = 1/3
    var_unif = 0.333333333

    for r in range(n_realisations):
        x = 0
        y = 0
        for t in range(n_samples):
            dx, dy = 2*(np.random.random(size=2) - .5)
            x += sx * dx
            y += sy * dy
            trajectories[0, t, r] = x
            trajectories[1, t, r] = y

    plt.plot(trajectories[0, :, 0], trajectories[1, :, 0])
    plt.plot(trajectories[0, :, 1], trajectories[1, :, 1])
    plt.plot(trajectories[0, :, 10], trajectories[1, :, 10])
    plt.plot(trajectories[0, :, 20], trajectories[1, :, 20])
    plt.show()

    srqt_t = np.zeros(shape=(n_samples,))
    for i in range(n_samples):
        srqt_t[i] = np.sqrt(i * var_unif)
    means = np.mean(trajectories, axis=2)
    std = np.std(trajectories, axis=2)
    plt.plot(means[0, :])
    plt.plot(means[1, :])
    plt.plot(std[0, :])
    plt.plot(std[1, :])
    plt.plot(srqt_t)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Histogramme des X')
    ax1.hist(trajectories[0, 10, :], bins=50)
    ax2.hist(trajectories[0, 100, :], bins=50)
    plt.show()





if __name__ == '__main__':
    main()