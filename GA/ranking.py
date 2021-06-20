from config.loader import Default

import numpy as np


class Elo(Default):
    def __init__(self, locked=False):
        super(Elo, self).__init__()

        self.n = 0
        self.elite = False
        self.locked = locked

        self.gamma = 1/6.


    def update(self, team_elo, other_elo, result):
        if not self.locked:
            self.start = self.start + self.k * (result - self.p(team_elo-other_elo))
            self.n += 1
            if not self.elite and self.n > self.old_age and self.start < self.elite_threshold:
                self.k = self.k_next
            elif self.start >= self.elite_threshold:
                self.elite = True
                self.k = self.k_elite


    def p(self, distance):
        return 1. / (1. + 10**(-np.clip(distance, -self.base, self.base)/self.base))

    def is_better_than(self, other):
        return self.current > other.current

    def __call__(self, *args, **kwargs):
        return self.start

    def win_prob(self, other_elo):
        return np.exp(-(self.p(self.start-other_elo)-0.5)**2/(2*self.gamma**2))/ (np.sqrt(2*np.pi)*self.gamma)


