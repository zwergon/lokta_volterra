
import numpy as np
from pub_crawl.markov_sampling import sample


def resample(particles, weights):
    """
    Particles are sampled with replacement proportional to their weight and in arbitrary order. This leads
    to a maximum variance on the number of times a particle will be resampled, since any particle will be
    resampled between 0 and N times.

    Computational complexity: O(N log(M)

    :param particles: Samples that must be resampled.
    :param weights: weights for each sample in particles array
    :return: Resampled weighted particles.
    """

    N = particles.shape[0]

    # As long as the number of new samples is insufficient
    n = 0
    new_samples = np.zeros(shape=particles.shape)
    while n < N:

        m = sample(weights)
        # Add copy of the state sample (uniform weights)
        new_samples[n, :] = particles[m, :]

        # Added another sample
        n += 1

    return new_samples, np.full(shape=(N,), fill_value=1./N)