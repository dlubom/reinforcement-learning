
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, m):
        self.m = m
        self.mean = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x


def run_experiment(m1, m2, m3, eps, N, experiment_name):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in range(N):
        # epsilon greedy
        p = np.random.random()
        if p < eps:
            j = np.random.choice(3)
        else:
            j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    # # plot moving average ctr
    # plt.plot(cumulative_average)
    # plt.plot(np.ones(N) * m1)
    # plt.plot(np.ones(N) * m2)
    # plt.plot(np.ones(N) * m3)
    # plt.xscale('log')
    # plt.show()

    for i, b in enumerate(bandits):
        print(f'{experiment_name}, bandit {i + 1} mean: {b.mean} win: {np.sum(data)}')

    return cumulative_average


if __name__ == '__main__':
    N = 100000

    m1 = 1.0
    m2 = 2.0
    m3 = 3.0
    eps1 = 0.1
    eps2 = 0.05
    eps3 = 0.01

    exp_1 = run_experiment(m1, m2, m3, eps1, N, f'exp_{eps1}')
    exp_2 = run_experiment(m1, m2, m3, eps2, N, f'exp_{eps2}')
    exp_3 = run_experiment(m1, m2, m3, eps3, N, f'exp_{eps3}')

    # log scale plot
    plt.plot(exp_1, label=f'eps = {eps1}')
    plt.plot(exp_2, label=f'eps = {eps2}')
    plt.plot(exp_3, label=f'eps = {eps3}')
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    # plt.plot(exp_1, label=f'eps = {eps1}')
    # plt.plot(exp_2, label=f'eps = {eps2}')
    # plt.plot(exp_3, label=f'eps = {eps3}')
    # plt.legend()
    # plt.show()
