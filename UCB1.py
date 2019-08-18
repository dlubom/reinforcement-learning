import numpy as np
import matplotlib.pyplot as plt
import comparing_epsilons
from bandit import Bandit


def ucb(mean, n, nj):
    if nj == 0:
        return float('inf')
    return mean + np.sqrt(2 * np.log(n) / nj)


def run_experiment(m1, m2, m3, N, experiment_name):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in range(N):
        j = np.argmax([ucb(b.mean, i + 1, b.N) for b in bandits])
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

    oiv = run_experiment(M1, M2, M3, N, 'ucb1')
    ce = comparing_epsilons.run_experiment(M1, M2, M3, 0.01, N, 'comparing_epsilons')
    # log scale plot
    plt.plot(oiv, label=f'ucb1')
    plt.plot(ce, label=f'comparing_epsilons')
    plt.plot(np.ones(N) * M1)
    plt.plot(np.ones(N) * M2)
    plt.plot(np.ones(N) * M3)
    plt.legend()
    plt.xscale('log')
    plt.show()
