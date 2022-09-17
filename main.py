import numpy as np
from stats.sampler1d import Sampler1D
import matplotlib.pyplot as plt
from stats.law import NormalLaw


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #samples  = metropolis_hasting(posterior, 100000)

    law = NormalLaw(mu=5, sigma=2)

    sampler = Sampler1D(law=law)
    samples = sampler.sample(10000)

    # ----------------------------------------------------------------------------------------#
    # plot posterior

    x_array = np.linspace(-5.0, 15.0, 100)
    y_array = np.asarray([law.pdf(x) for x in x_array])

    plt.plot(x_array, y_array)
    plt.hist(samples, bins=30, density=True)
    plt.show()
