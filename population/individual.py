
from GA.genotype import Genotype
from GA.ranking import Elo
from GA.naming import Name
from game.enums import PlayerType

import numpy as np


class Individual:
    def __init__(self,
                 ID,
                 char,
                 is_cpu = False,
                 state_space=None,
                 name=None
                 ):

        self.id = ID
        self.type = PlayerType.CPU if is_cpu else PlayerType.Human
        self.state_space = state_space
        self.genotype = Genotype(char, initial_brain=None, is_dummy=is_cpu)
        self.elo = Elo(locked=is_cpu)
        self.name = Name(name)
        self.data_used = 0
        self.mean_entropy = np.log(char.action_space.dim)
        # random policy
        # self.dist = np.ones(self.char.action_space.dim, dtype=np.float32) / self.char.action_space.dim
        # self.policy = lambda state: (np.random.choice(self.char.action_space.dim), self.dist)

    def win_probs(self, other_elo):
        return self.elo.win_prob(other_elo)

    def reward_shape(self):
        return self.genotype['experience']

    def get_genes(self):
        return self.genotype.get_params()

    def set_genes(self, new_genes):
        self.genotype.set_params( new_genes)

    def policy(self, observation):
        return self.genotype['brain'](observation)

    def char(self):
        return self.genotype['type']

    def inerit_from(self, *other_individuals):
        if len(other_individuals) == 1:
            self.genotype.set_params(other_individuals[0].genotype.get_params())
            self.name.inerit_from(other_individuals[0].name)
        elif len(other_individuals) == 2 :
            self.genotype.set_params(other_individuals[0].genotype.crossover(other_individuals[1].genotype))
            self.name.inerit_from(other_individuals[0].name, other_individuals[1].name)

        self.elo = Elo() # reset
        self.data_used = 0

    def perturb(self):
        self.genotype.perturb()

    def __repr__(self):
        return 'Individual_{ID} {name} :\n' \
                   'Elo = {elo}\n'\
                   'Age = {age}\n'\
                   'genes = {genotype}\n'.format(ID=self.id, name=self.name.get(), age=self.elo.n, elo=self.elo(),
                                              genotype=self.genotype if not self.type == PlayerType.CPU else 'ingame_CPU')






