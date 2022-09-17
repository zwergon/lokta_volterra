
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from spark_volterra.lokta_volterra import LoktaVolterra2
from spark_volterra.resampling import resample

PREDATORS = 1
PREYS = 0



def likelihood(sample, observation):
    sigma = 100
    return multivariate_normal.pdf(observation, mean=sample, cov=sigma)


def particle_filter(lv: LoktaVolterra2, observations, sample_size):

    obs_size = observations.shape[0]

    particles = np.zeros(shape=(sample_size, observations.shape[1]))
    weights = np.zeros(shape=(sample_size,))
    predicted = np.zeros(shape=observations.shape)

    # Initialize t=0 with poisson processes
    particles[:, PREDATORS] = np.random.poisson(observations[0, PREDATORS], size=sample_size)
    particles[:, PREYS] = np.random.poisson(observations[0, PREYS], size=sample_size)

    t = 0
    for i in range(sample_size):
        observation = observations[t, :]
        weights[i] = likelihood(particles[i, :], observation)

    predicted[t, :] = np.average(particles, weights=weights, axis=0)
    log_likeli = np.log(np.mean(weights))
    weights /= np.sum(weights)
    particles, weights = resample(particles, weights)

    # Iteration with integration
    for t in range(1, obs_size):
        print(f"observation: {t}")
        observation = observations[t, :]

        for i in range(sample_size):

            lv.run_simple_euler_muruayama(particles[i, :], 1, time_scale=0.0001)
            particles[i, :] = lv.x[-1]

            # No need to multiply weights by likelihood
            # At the beginning of the loop all weights are evenly distributed
            weights[i] = likelihood(particles[i, :], observation)

        log_likeli += np.log(np.mean(weights))
        weights /= np.sum(weights)
        predicted[t, :] = np.average(particles, weights=weights, axis=0)

        particles, weights = resample(particles, weights)

    return predicted, log_likeli


