
import numpy as np
import random


def sample(probas):
    """
    Sample a discrete proba law given as 1D array
    :param probas: 1D discrete probabilities distribution np.sum(probas) = 1
    :return: One sample according probas
    """
    u = random.random()
    index = 0
    acc = probas[index]
    while acc < u:
        index += 1
        acc += probas[index]
    return index


def random_walk(i_from, transition):
    """
    Give next sample in markov chain using transition matrix
    :param i_from: index from where next sample need to be sampled
    :param transition: Transition Matrix of this markov chain
    :return: next index in proba array
    """
    return sample(transition[i_from, :])


def estimate_probas(transition):
    A = transition.transpose() - np.identity(transition.shape[0])
    b = np.zeros(transition.shape[0])
    A[-1, :] = 1
    b[-1] = 1
    return np.linalg.solve(A, b)


def compute_stat(row, row_length):
    max = np.max(row)
    stat = np.zeros(int(max)+1)
    for i in range(row_length):
        stat[int(row[i])] += 1
    return stat / row_length





