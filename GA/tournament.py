import itertools

from config.loader import Default
from logger.logger import Logger

import numpy as np
from math import comb, ceil, log
import copy
import datetime
import sys
import os


class Tournament(Default, Logger):
    def __init__(self, pop_size):
        super(Tournament, self).__init__()

        self.n_teams = pop_size//2
        #self.teams = np.concatenate(
        #    [np.random.choice(pop_size, (self.n_teams, 2), replace=False) for _ in range(2)], axis=0)
        self.pool_wins = np.zeros((self.n_teams,), dtype=np.int32)
        self.teams = np.random.choice(pop_size, (self.n_teams, 2), replace=False)

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
                yield 'pool', (*t1, *t2)


    def step(self, type, p1, p2, p3, p4, result):
        if type in ['pool', 'bracket']:
            if result == 0:
                winner = np.argwhere(self.teams==p3)[0,0]
            elif result == 1 :
                winner = np.argwhere(self.teams==p1)[0,0]
            else:
                p = np.random.choice([p1,p3])
                winner = np.argwhere(self.teams==p)[0,0]

            if type=='pool':
                self.pool_wins[winner] += 1
                for i,p in enumerate(self.pools):
                    if not p.is_done :
                        qualified = p.qualified(self.pool_wins, n=self.pool_qualifications)
                        if qualified is not None:
                            self.logger.info('Teams %s qualified for brackets !' % str(qualified))
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
            if match is None:
                yield None
            else:
                yield 'bracket', (*self.teams[match[0]], *self.teams[match[1]])
    def __call__(self):
        for m in self.do_pools():
            yield m
        for m in self.do_brackets():
            yield m

        while len(self.bracket.winners()) == 0:
            yield None

        self.winners = self.bracket.winners()

    def result(self):
        if self.winners is None :
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

        self.visualizer = None

        self.entrants = {}

    def register(self, team):
        # do not wait for full registration
        if self.register_index < self.size:
            match_num = self.register_order[self.register_index]//2
            self.entrants.update({team: (0, match_num)})
            self.levels[0][match_num].append(team)
            self.register_index += 1

            print(self.register_index, self.levels[0])

        else :
            print('tournament full!')

    def matches(self):
        for level in self.levels[:-1]:
            for match in level:
                while len(match)<2 :
                    yield None
                yield match

    def winners(self):
        return self.levels[-1][0]


class BracketVisualizer:
    """
    Adapted from :
    https://github.com/cristiean/bracket/tree/e1e20397ef0405f09cf4854028f498ded21bfaa0
    """
    def __init__(self, teams):
        self.numTeams = len(teams)
        self.teams = list(teams)
        self.max = len(max(["Round "] + teams, key=len))
        self.numRounds = int(ceil(log(self.numTeams, 2)) + 1)
        self.totalNumTeams = int(2 ** ceil(log(self.numTeams, 2)))
        self.totalTeams = self.addTeams()
        self.lineup = ["bye" if "-" in str(x) else x for x in self.totalTeams]
        print(self.lineup)
        self.numToName()
        self.count = 0
        self.rounds = []
        for i in range(0, self.numRounds):
            self.rounds.append([])
            for _ in range(0, 2 ** (self.numRounds - i - 1)):
                self.rounds[i].append("-" * self.max)
        self.rounds[0] = list(self.totalTeams)

    def numToName(self):
        for i in range(0, self.numTeams):
            self.totalTeams[self.totalTeams.index(i + 1)] = self.teams[i]

    def update(self, rounds, teams):
        lowercase = [team.lower() for team in self.rounds[rounds - 2]]
        for team in teams:
            try:
                index = lowercase.index(team.lower())
                self.rounds[rounds - 1][int(index / 2)] = self.rounds[rounds - 2][index]
            except:
                return False
        if "-" * self.max in self.rounds[rounds - 1]:
            return False
        return True

    def show(self):
        print('='*15)
        print('')
        self.count = 0
        self.temp = copy.deepcopy(self.rounds)
        self.tempLineup = list(self.lineup)
        sys.stdout.write("Seed ")
        for i in range(1, self.numRounds + 1):
            sys.stdout.write(("Round " + str(i)).rjust(self.max + 3))
        print("")
        self.recurse(self.numRounds - 1, 0)

        print('')
        print('=' * 15)


    def recurse(self, num, tail):
        if num == 0:
            self.count += 1
            if tail == -1:
                print(str(self.tempLineup.pop(0)).rjust(4) + self.temp[0].pop(0).rjust(self.max + 3) + " \\")
            elif tail == 1:
                print(str(self.tempLineup.pop(0)).rjust(4) + self.temp[0].pop(0).rjust(self.max + 3) + " /")
        else:
            self.recurse(num - 1, -1)
            if tail == -1:
                print("".rjust(4) + "".rjust((self.max + 3) * num) + self.temp[num].pop(0).rjust(self.max + 3) + " \\")
            elif tail == 1:
                print("".rjust(4) + "".rjust((self.max + 3) * num) + self.temp[num].pop(0).rjust(self.max + 3) + " /")
            else:
                print("".rjust(4) + "".rjust((self.max + 3) * num) + self.temp[num].pop(0).rjust(self.max + 3))
            self.recurse(num - 1, 1)

    def addTeams(self):
        x = self.numTeams
        teams = [1]
        temp = []
        count = 0
        for i in range(2, x + 1):
            temp.append(i)
        for i in range(0, int(2 ** ceil(log(x, 2)) - x)):
            temp.append("-" * self.max)
        for _ in range(0, int(ceil(log(x, 2)))):
            high = max(teams)
            for i in range(0, len(teams)):
                index = teams.index(high) + 1
                teams.insert(index, temp[count])
                high -= 1
                count += 1
        return teams

"""
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


