import numpy as np
from metropolis_hasting import metropolis_hasting
from repartition import repartition
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------#
# define posterior distribution

def posterior(x):
    mu, sigma = 5, 2.0 # mean and standard deviation
    num = np.exp( - ( x - mu )**2 / ( 2.0 * sigma **2 ) )
    den = np.sqrt( 2 * np.pi * sigma **2)
    return  num / den


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #samples  = metropolis_hasting(posterior, 100000)

    samples = repartition(10000, 5, 2)

    # ----------------------------------------------------------------------------------------#
    # plot posterior

    x_array = np.linspace(-5.0, 15.0, 100)
    y_array = np.asarray([posterior(x) for x in x_array])

    plt.plot(x_array, y_array)
    plt.hist(samples, bins=30, density=True)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
