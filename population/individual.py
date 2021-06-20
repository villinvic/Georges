from GA.genotype import Genotype
from GA.ranking import Elo
from GA.naming import Name
from game.enums import PlayerType
from game.state import GameState

import numpy as np
from datetime import datetime


class Individual:
    def __init__(self,
                 ID,
                 char,
                 is_cpu=False,
                 name=None,
                 test=False,
                 ):

        self.id = ID
        self.type = PlayerType.CPU if is_cpu else PlayerType.Human
        self.genotype = Genotype(GameState.size, char, is_dummy=is_cpu or test)
        self.elo = Elo(locked=is_cpu)
        self.name = Name(name)
        self.data_used = 0
        self.mean_entropy = np.log(char.action_space.dim)
        self.birthday = datetime.today()
        # random policy
        # self.dist = np.ones(self.char.action_space.dim, dtype=np.float32) / self.char.action_space.dim
        # self.policy = lambda state: (np.random.choice(self.char.action_space.dim), self.dist)

    def win_prob(self, other_elo):
        return self.elo.win_prob(other_elo)

    def get_reward_shape(self):
        return self.genotype['experience']

    def learning_params(self):
        return self.genotype['learning']

    def get_arena_genes(self):
        return {
            'brain': self.genotype['brain'].get_params(),
            'char' : self.genotype['type'],
            'type' : self.type,
        }

    def set_arena_genes(self, arena_genes):
        self.genotype['brain'].set_params(arena_genes['brain'])
        self.genotype['type'] = arena_genes['char']
        self.type = arena_genes['type']

    def get_genes(self):
        return self.genotype.get_params()

    def set_genes(self, new_genes):
        self.genotype.set_params(new_genes)

    def get_all(self):
        return dict(
            id=self.id,
            type=self.type,
            genotype=self.genotype.get_params(),
            elo=self.elo,
            name=self.name,
            data_used=self.data_used,
            mean_entropy=self.mean_entropy,
            birthday=self.birthday
        )

    def set_all(self, params, check_age=False):
        if not check_age or params['birthday'] >= self.birthday:
            self.type = params['type']
            self.genotype.set_params(params['genotype'])
            self.elo = params['elo']
            self.name = params['name']
            self.data_used = params['data_used']
            self.mean_entropy = params['mean_entropy']
            self.birthday = params['birthday']
        else:
            print('Tried to override an individual with older params')

    def policy(self, observation):
        if self.genotype['brain'] is None:
            return np.random.randint(0, self.char().get().action_space.dim), 0, None, None
        else:
            return self.genotype['brain'](observation)

    def char(self):
        return self.genotype['type']

    def inerit_from(self, *other_individuals):
        if len(other_individuals) == 1:
            self.genotype.set_params(other_individuals[0].genotype.get_params())
            self.name.inerit_from(other_individuals[0].name)
        elif len(other_individuals) == 2:
            self.genotype.set_params(other_individuals[0].genotype.crossover(other_individuals[1].genotype))
            self.name.inerit_from(other_individuals[0].name, other_individuals[1].name)

        self.elo = Elo()  # reset
        self.data_used = 0
        self.birthday = datetime.today()

    def perturb(self):
        self.genotype.perturb()

    def __repr__(self):
        return 'Individual_{ID} {name} :\n' \
               'Elo = {elo}\n' \
               'Age = {age}\n' \
               'genes = {genotype}\n'.format(ID=self.id, name=self.name.get(), age=self.elo.n, elo=self.elo(),
                                             genotype=self.genotype if not self.type == PlayerType.CPU else 'ingame_CPU')
