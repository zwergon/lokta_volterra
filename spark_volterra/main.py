import pandas as pd
import matplotlib.pyplot as plt
from spark_volterra.lokta_volterra import LoktaVolterra2
from spark_volterra.particle_filter import particle_filter, PREYS, PREDATORS


if __name__ == '__main__':

    # alpha, beta , gamma
    c = [0.55, 0.028, 0.84]

    lambda_i = [
        [1, -1, 0.],
        [0., 1, -1]
    ]

    def transition(c, x):
        return [
            c[0] * x[0],
            c[1] * x[0] * x[1],
            c[2] * x[1]
        ]


    lv = LoktaVolterra2(c, lambda_i, transition)

    obs_df = pd.read_csv("prey_predator.csv", sep=',')
    observations = obs_df.loc[:, ['Hare', 'Lynx']].to_numpy()

    predicted, log_likehood = particle_filter(lv, observations, 12)

    print("log_likehood :", log_likehood)
    plt.plot(predicted[:, PREYS])
    plt.plot(predicted[:, PREDATORS])
    plt.plot(observations[:, PREYS], '--')
    plt.plot(observations[:, PREDATORS], '--')
    plt.show()

