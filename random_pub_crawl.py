import numpy as np
from enum import Enum

from markov_sampling import estimate_probas, compute_stat, random_walk
import random
import matplotlib.pyplot as plt


class PUB(Enum):
    HOME = 0
    PERCH = 1
    ROYAL_OAK = 2,
    EAGLE = 3
    BEAR = 4
    CAPE = 5
    MAGDALEN = 6


def main():

    N_TRAJECTORIES = 1
    N_STEPS = 10000
    BURNIN = 100
    I_START = PUB.HOME.value

    transition = np.array(
        [
            [.5, 0, .5, 0, 0, 0, 0],  # Home
            [0, 0, .5, .5, 0, 0, 0],  # Perch
            [.25, 0.25, 0, .25, 0, .25, 0],  # ROYAL
            [0, 0.25, .25, 0, 0.25, .25, 0],  # EAGLE
            [0, 0, 0, 0.5, 0, 0.5, 0],  # BEAR
            [0, 0, 0.25, 0.25, 0.25, 0, .25],  # CAPE
            [0, 0, 0, 0, 0, 1, 0]  # MAGDALEN
        ]
    )

    random.seed(1)

    pubs = np.zeros(shape=(N_TRAJECTORIES, N_STEPS))

    for i_traj in range(N_TRAJECTORIES):
        i_from = I_START
        for i in range(-BURNIN, N_STEPS):
            i_to = random_walk(i_from, transition)
            if i >= 0:
                pubs[i_traj, i] = i_to
            i_from = i_to

        stat = compute_stat(pubs[i_traj, :], N_STEPS)
        print(stat)

    t_proba = estimate_probas(transition)
    print(t_proba)

    plt.bar(range(7), stat)
    plt.show()


if __name__ == '__main__':
    main()




