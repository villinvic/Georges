import itertools

from config.loader import Default

import numpy as np
from math import comb
from time import sleep
from itertools import combinations


class Tournament(Default):
    def __init__(self, pop_size):
        super(Tournament, self).__init__()
        self.pop_size = pop_size

        self.n_teams = pop_size//2
        self.teams = np.random.choice(pop_size, (self.n_teams, 2), replace=False)
        self.pool_wins = np.zeros((self.n_teams,), dtype=np.int32)

        self.pools = None
        self.bracket = Bracket(size=self.n_pools*self.pool_qualifications)

        self.winners = None

    def do_pools(self):

        self.pools = []
        i = 0
        leftovers = self.n_teams % self.n_pools
        while i < self.n_teams:
            pool_size = int(min([self.n_teams-i, self.n_teams//self.n_pools]))

            if leftovers > 0:
                pool_size += 1
                leftovers -= 1

            self.pools.append(Pool(
                np.arange(i, i+pool_size),[team_players for team_players in self.teams[i:i+pool_size]]
            ))
            i += pool_size

        for i,p in enumerate(self.pools):
            for j, match in enumerate(p.matches()):
                t1, t2 = match
                yield 'pool', *t1, *t2


    def step(self, type, p1, p2, p3, p4, result):
        if result == 0:
            winner = np.argwhere(self.teams==p3)[0,0]
        elif result == 1 :
            winner = np.argwhere(self.teams==p1)[0,0]

        if type=='pool':
            self.pool_wins[winner] += 1
            for i,p in enumerate(self.pools):
                if not p.is_done :
                    qualified = p.qualified(self.pool_wins, n=self.pool_qualifications)
                    if qualified is not None:
                        for q in qualified:
                            self.bracket.register(q)

        elif type=='bracket':
            level, match_num = self.bracket.entrants[winner]
            level += 1
            match_num //= 2
            self.bracket.levels[level][match_num].append(winner)
            self.bracket.entrants[winner] = (level, match_num)


    def do_brackets(self):
        for match in self.bracket.matches():
            yield 'bracket', *self.teams[match[0]], *self.teams[match[1]]

    def __call__(self):
        for m in self.do_pools():
            yield m
        for m in self.do_brackets():
            yield m

        while self.bracket.winners() is None:
            yield None

        self.winners = self.bracket.winners()

    def result(self):
        if self.winners is None:
            return None
        else:
            return self.teams[self.winners][0]


class Pool:
    def __init__(self, team_indexes, teams):
        self.team_indexes = team_indexes
        self.teams = teams
        self.all_matches = itertools.combinations(self.teams, 2)
        self.n_matches = comb(len(team_indexes), 2)
        self.is_done = False

    def matches(self):
        return self.all_matches

    def qualified(self, pool_wins, n=2):
        scores = pool_wins[self.team_indexes]
        #print(np.sum(scores), self.n_matches)
        if np.sum(scores) < self.n_matches:
            return None
        else:
            rank = np.argsort(pool_wins[self.team_indexes])
            self.is_done = True
            return self.team_indexes[rank[-n:]]


class Bracket:
    def __init__(self, size):
        levels = []
        n = size
        self.size = size
        self.total_matches = 0
        while n > 1:
            n //= 2
            self.total_matches += n
            levels.append([[] for _ in range(n)])
        levels.append([[]])

        self.levels = np.array(levels)

        self.register_index = 0
        self.register_order = np.arange(size)
        np.random.shuffle(self.register_order)

        self.entrants = {}

    def register(self, team):
        # do not wait for full registration
        if self.register_index < self.size:
            match_num = self.register_order[self.register_index]//2
            self.entrants.update({team: (0, match_num)})
            self.levels[0][match_num].append(team)
            self.register_index += 1

        else :
            print('tournament full!')

    def matches(self):
        for level in self.levels[:-1]:
            for match in level:
                while None in match :
                    yield None
                yield match

    def winners(self):
        return self.levels[-1][0]


class Player:
    def __init__(self, ID, power=1.):
        self.id = ID
        self.power = power

    def __repr__(self):
        return str(self.power)


def play_match(m):
    t, p1, p2, p3, p4 = m
    probs = np.array([p1.power+p2.power, p3.power+p4.power])
    probs /= np.sum(probs)
    result = np.random.choice([1,0], None, p=probs)

    return result



"""
if __name__ == '__main__':

    x = 0.
    pop = np.array([Player(i, float(i+1)) for i in range(28)])
    pop[0].power = 10000
    winners = np.zeros(28)
    for _ in range(500):

        t = Tournament(28)

        for m in t():
            if m is None:
                t.step('bob', 0,0,0,0,0)
            else:
                mode, p1, p2, p3, p4 = m
                m = mode, pop[p1], pop[p2], pop[p3], pop[p4]
                result = play_match(m)
                t.step(mode, p1, p2, p3, p4, result)

        w1, w2 = t.result()
        winners[t.result()] += 1

        x += (pop[w1].power + pop[w2].power)*0.5
    print(x/500., print(winners))
"""


