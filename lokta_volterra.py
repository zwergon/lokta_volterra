import numpy as np
import random
import warnings
warnings.simplefilter("error")


class LoktaVolterra:

    def __init__(self, c, lambda_i, transition, dt=1e-5):
        self.x = []
        self.times = []

        self.dt = 0.01
        self.c = c

        self._lambda_i = np.array(lambda_i)
        self._transition = transition

    @property
    def lambda_i(self):
        return 2.*self._lambda_i


    @staticmethod
    def _select_index(a0r2, a):
        index = 0
        acc = a[index]
        while acc < a0r2:
            index += 1
            acc += a[index]
        return index

    def _compute_drift(self, x):
        p_i = self._probas(x)
        mu_dt = self.dt * np.matmul(self.lambda_i, p_i)

        return x + mu_dt

    def _sample_next(self, x):

        p_i = self._probas(x)
        size = len(p_i)
        sqrt_lambda = np.identity(size) * np.sqrt(p_i)
        G = np.matmul(self.lambda_i, sqrt_lambda)
        dW = np.random.normal(size=size, scale=np.sqrt(self.dt))

        diffusion = np.matmul(G, dW)
        xn = self._compute_drift(x)

        return xn + diffusion

    def _probas(self, x):
        a = self._transition(self.c, x)
        return a / np.sum(a)

    def run_single_deterministic(self, x0, total_time):
        t = 0
        self.x = [x0]
        self.times = [t]

        x = x0
        while t < total_time:
            x = self._compute_drift(x)
            t += self.dt

            self.times.append(t)
            self.x.append(x)

    def run_simple_gillepsie(self, x0, total_time, time_scale=1):
        t = 0
        self.x = [x0]
        self.times = [t]

        x = x0

        while t < total_time:
            a = self._transition(self.c, x)
            a0 = np.sum(a)
            r1 = random.random()
            r2 = random.random()

            # Compute time to next reaction
            tau = -np.log(r1)/a0

            # Select index for the next reaction
            index = self._select_index(a0*r2, a)

            # Updates values of components using stoechiometrix matrix
            x = tuple([x[i] + self.lambda_i[i][index] for i in range(len(x))])
            t += tau

            self.times.append(t*time_scale)
            self.x.append(x)

    def run_simple_euler_muruayama(self, x0, total_time, time_scale=.01):

        t = 0
        self.x = [x0]
        self.times = [t]

        x = x0

        n_x = int(total_time / time_scale)
        self.dt = time_scale
        i = 1
        while i < n_x:
            x = self._sample_next(x)
            t += self.dt
            self.times.append(t)
            self.x.append(x)
            i += 1


class LoktaVolterra2(LoktaVolterra):

    def __init__(self, c, lambda_i, transition):
        LoktaVolterra.__init__(self, c, lambda_i, transition)

    @property
    def prey(self):
        return [x[0] for x in self.x]

    @property
    def predator(self):
        return [x[1] for x in self.x]


class LoktaVolterra3(LoktaVolterra):

    def __init__(self, c, lambda_i, transition):
        LoktaVolterra.__init__(self, c, lambda_i, transition)

    @property
    def prey(self):
        return [x[0] for x in self.x]

    @property
    def predator1(self):
        return [x[1] for x in self.x]

    @property
    def predator2(self):
        return [x[2] for x in self.x]


def compute3():

    c = [10., 0.005, 0.0025, 6, 3]

    lambda_i = [
        [1, -1, -1, 0, 0],
        [0, 1, 0, -1, 0],
        [0, 0, 1, 0, -1]
    ]

    def transition(c, x):
        return [
            c[0] * x[0],
            c[1] * x[0] * x[1],
            c[2] * x[0] * x[2],
            c[3] * x[1],
            c[4] * x[2]
        ]

    lv = LoktaVolterra3(c, lambda_i, transition)
    lv.run_simple_euler_muruayama((800, 500, 600), 50000., time_scale=100)

    plt.plot(lv.times, lv.prey)
    plt.plot(lv.times, lv.predator1)
    plt.plot(lv.times, lv.predator2)
    plt.show()


def compute2():

    ## alpha, beta , gamma
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

    ## x[0]: Hare, x[1]: Lynx
    total_time = 1000.
    n_run = 1
    timescale = .1
    n_x = int(total_time / timescale)

    y_results = np.zeros(shape=(2, n_run, n_x))
    i= 0
    while i < n_run:
        try:
            lv.run_simple_euler_muruayama((30, 4), total_time=total_time, time_scale=timescale)
            y_results[0, i, :] = np.array(lv.prey)
            y_results[1, i, :] = np.array(lv.predator)
            i += 1
        except RuntimeWarning:
            print("error in run_simple_euler_muruayama, retry...")

    time_stoc = lv.times.copy()

    lv.run_single_deterministic((30, 4), total_time)
    plt.plot(lv.times, lv.prey)
    plt.plot(lv.times, lv.predator)
    plt.plot(time_stoc, np.mean(y_results[0, :, :], axis=0))
    plt.plot(time_stoc, np.mean(y_results[1, :, :], axis=0))
    #plt.plot(time_stoc, y_results[0, 0, :])
    #plt.plot(time_stoc, y_results[1, 0, :])
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    compute2()


