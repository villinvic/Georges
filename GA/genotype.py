import numpy as np

from config.loader import Default
from GA import misc
from characters.characters import Mario, available_chars, enum2index
from training.RL import Policy, AC
from collections import deque
from copy import deepcopy


class Genotype(Default):
    _base_keys = [
        'learning',
        'experience',
        'type'
    ]

    _special = [
        'brain'
    ]

    def __init__(self, state_dim, initial_char = Mario, trainable=False, is_dummy=False):
        super(Genotype, self).__init__()
        self.is_dummy = is_dummy
        self.layer_dims = []
        self.lstm_dim = 0

        if is_dummy:
            self._genes = {
                'brain' : None,
                'learning': None,
                'experience': None,
                'type': EvolvingCharacter(initial_char, frozen=True)
            }

        else:
            for type, dimension in self.brain_model:
                if type == 'Dense':
                    self.layer_dims.append(dimension)
                elif type == 'LSTM':
                    self.lstm_dim = dimension
                else:
                    print('Unsupported layer type... (%s)' %type)

            # get rid of the panda data
            del self.brain_model

            Brain_cls = AC if trainable else Policy
            self._genes = {
                'brain': Brain_cls(initial_char.action_space.dim, self.layer_dims, self.lstm_dim), # brain function (must have a __call__ and perturb function) Usually an NN
                'learning': LearningParams(),
                'experience': RewardShape(),
                'type': EvolvingCharacter(initial_char),
            }

            self._genes['brain'].init_body(np.zeros((1,1,state_dim)))

    def perturb(self):
        if not self.is_dummy:
            for gene_family in self._genes.values():
                if gene_family is not None:
                    gene_family.perturb()

    def get_params(self, full_brain=False, trainable=False):
        if full_brain:
            c = {'brain': self._genes['brain']}
        elif trainable:
            c= {'brain': self._genes['brain'].get_training_params()}
        else:
            c = {'brain': self._genes['brain'].get_params()}

        c.update({key: self._genes[key].copy() for key in self._base_keys})

        return c

    def set_params(self, new_genes, trainable=False):
        for key in self._base_keys:
            self._genes[key] = new_genes[key]
        if trainable:
            self._genes['brain'].set_training_params(new_genes['brain'])
        else:
            self._genes['brain'].set_params(new_genes['brain'])

    def __repr__(self):
        return self._genes.__repr__()

    def __getitem__(self, item):
        return self._genes[item]

    def __setitem__(self, key, value):
        self._genes[key] = value

    def crossover(self, other_genotype):
        genes = self.get_params(full_brain=True)
        for gene_family, other_gene_family in zip(genes.values(), other_genotype._genes.values()):
            if gene_family is not None:
                gene_family.crossover(other_gene_family)

        genes['brain'] = genes['brain'].get_training_params()

        return genes



class EvolvingFamily:
    def __init__(self):
        self._variables = {name: EvolvingVariable(name, (domain_lower, domain_higher), self.perturb_power, self.perturb_chance) for name, domain_lower, domain_higher
                          in self.variable_base}

        # get rid of the panda data
        del self.variable_base

    def __getitem__(self, item):
        return self._variables[item].get()

    def perturb(self):
        for variable in self._variables.values():
            variable.perturb()

    def crossover(self, other_family):
        for variable, other_variable in zip(self._variables.values(), other_family._variables.values()):
            variable.crossover(other_variable)

    def __repr__(self):
        return self._variables.__repr__()


class RewardShape(Default, EvolvingFamily):
    def __init__(self):
        super().__init__()

    def copy(self):
        new = RewardShape()
        new._variables = {}
        for k, v in self._variables.items():
            new._variables[k] = v.copy()
        return new


class LearningParams(Default, EvolvingFamily):
    def __init__(self):
        super().__init__()


    def copy(self):
        new = LearningParams()
        new._variables = {}
        for k, v in self._variables.items():
            new._variables[k] = v.copy()
        return new


class EvolvingVariable(Default):
    def __init__(self, name, domain, perturb_power=0.2, perturb_chance=0.05, frozen=False):
        super(EvolvingVariable, self).__init__()
        self.name = name
        self.domain = domain
        self._current_value = misc.log_uniform(*domain)
        self.perturb_power = perturb_power
        self.perturb_chance = perturb_chance
        self.history = deque([self._current_value], maxlen=int(self.history_max))

        self.frozen = frozen

    def perturb(self):
        if not self.frozen and np.random.random() < self.perturb_chance:
            perturbation = np.random.uniform(1.-self.perturb_power, 1.+self.perturb_power)
            self._current_value = np.clip(perturbation * self._current_value, *self.domain) # clip ??
            self.history.append(self._current_value)

    def crossover(self, other_variable):
        if np.random.random() < 0.5:
            self._current_value = other_variable._current_value

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def __repr__(self):
        return str(self._current_value)

    def get(self):
        return self._current_value

    def copy(self):
        new = EvolvingVariable(name=self.name, domain=(0,1), perturb_power=self.perturb_power, perturb_chance=self.perturb_chance, frozen=self.frozen)
        new.domain = self.domain
        new._current_value = self._current_value
        new.history = deepcopy(self.history)

        return new


class EvolvingCharacter(Default):
    def __init__(self, initial_char = Mario, frozen=False):
        super(EvolvingCharacter, self).__init__()

        self.frozen = frozen
        self._character = initial_char
        self.history = deque([self._character], maxlen=int(self.history_max))

    def perturb(self):
        if np.random.random() < self.perturb_chance:
            p = np.ones(len(available_chars), dtype=np.float32)
            for clone in self._character.clones:
                p[enum2index[clone]] += self.clone_weight / float(len(self._character.clones))

            self._character = np.random.choice(available_chars, None, p=p/np.sum(p))
            self.history.append(self._character)

    def crossover(self, other_char):
        if np.random.random() < 0.5:
            self._character = other_char._character
            self.history.append(self._character)

    def __repr__(self):
        return self._character.__name__

    def get(self):
        return self._character

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def copy(self):
        new = EvolvingCharacter(initial_char=self._character, frozen=self.frozen)
        new.history = deepcopy(self.history)

        return new

