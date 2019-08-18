import numpy as np
import matplotlib.pyplot as plt
import comparing_epsilons


class Bandit:
    def __init__(self, m):
        self.m = m
        self.mean = 10
        self.N = 0

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x


def run_experiment(m1, m2, m3, N, experiment_name):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in range(N):
        j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    for i, b in enumerate(bandits):
        print(f'{experiment_name}, bandit {i + 1} mean: {b.mean} win: {np.sum(data)}')

    return cumulative_average


if __name__ == '__main__':
    N = 100000

    M1 = 1.0
    M2 = 2.0
    M3 = 3.0

    oiv = run_experiment(M1, M2, M3, N, 'oiv')
    ce = comparing_epsilons.run_experiment(M1, M2, M3, 0.01, N, 'comparing_epsilons')
    # log scale plot
    plt.plot(oiv, label=f'optimistic_initial_values')
    plt.plot(ce, label=f'comparing_epsilons')
    plt.plot(np.ones(N) * M1)
    plt.plot(np.ones(N) * M2)
    plt.plot(np.ones(N) * M3)
    plt.legend()
    plt.xscale('log')
    plt.show()
