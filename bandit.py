import numpy as np


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