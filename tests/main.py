import numpy as np
from stats.sampler1d import Sampler1D
import matplotlib.pyplot as plt
from stats.law import BernouilliLaw, NormalLaw


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #samples  = metropolis_hasting(posterior, 100000)

    law = NormalLaw(sigma=7, mu=170)

    sampler = Sampler1D(law=law)

    samples = sampler.sample(100)

    print(np.mean(samples), np.var(samples))

    x = samples

    v = np.mean((x-np.mean(x))**2)
    print(v)

    print(np.var(x, ddof=1))


    # trajectories = np.zeros(shape=(10000, 100))
    #
    # for i in range(trajectories.shape[0]):
    #     samples = sampler.sample(100)
    #     trajectories[i, :] = np.cumsum(samples)
    #
    # # ----------------------------------------------------------------------------------------#
    # # plot posterior
    # for i in range(10):
    #     plt.plot(trajectories[i, :])
    # plt.show()
    #
    #
    # plt.hist(trajectories[:, 10], bins=11, density=True)
    # plt.show()
