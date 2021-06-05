import numpy as np

from config.loader import Default
from GA import misc
from characters.characters import Mario, available_chars, enum2index
from collections import deque


class Genotype(Default):
    def __init__(self, initial_char = Mario, initial_brain=None, is_dummy=False):
        super(Genotype, self).__init__()
        self.is_dummy = is_dummy

        if is_dummy:
            self._genes = {
                'brain' : None,
                'learning': None,
                'experience': None,
                'type': EvolvingCharacter(initial_char, frozen=True)
            }

        else:
            self._genes = {
                'brain': initial_brain, # brain function (must have a __call__ and perturb function) Usually an NN
                'learning': LearningParams(),
                'experience': RewardShape(),
                'type': EvolvingCharacter(initial_char),
            }

    def perturb(self):
        if not self.is_dummy:
            for gene_family in self._genes.values():
                if gene_family is not None:
                    gene_family.perturb()

    def get_params(self):
        return self._genes

    def set_params(self, new_genes):
        self._genes = new_genes

    def __repr__(self):
        return self._genes.__repr__()

    def __getitem__(self, item):
        return self._genes[item]

    def crossover(self, other_genotype):
        for gene_family, other_gene_family in zip(self._genes.values(), other_genotype._genes.value()):
            if gene_family is not None:
                gene_family.crossover(other_gene_family)



class EvolvingFamily:
    def __init__(self):
        self._variables = {name: EvolvingVariable(name, (domain_lower, domain_higher), self.perturb_power, self.perturb_chance) for name, domain_lower, domain_higher
                          in self.variable_base}

    def __getitem__(self, item):
        return self._variables[item].get()

    def perturb(self):
        for variable in self._variables.values():
            variable.perturb()

    def crossover(self, other_family):
        for variable, other_variable in zip(self._variables.values(), other_family.values()):
            variable.crossover(other_variable)

    def __repr__(self):
        return self._variables.__repr__()


class RewardShape(Default, EvolvingFamily):
    def __init__(self):
        super().__init__()


class LearningParams(Default, EvolvingFamily):
    def __init__(self):
        super().__init__()


class EvolvingVariable(Default):
    def __init__(self, name, domain, perturb_power=0.2, perturb_chance=0.05, frozen=False):
        super(EvolvingVariable, self).__init__()
        self.name = name
        self.domain = (10.**domain[0], 10.**domain[1])
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
                p[enum2index[clone]] += self.clone_weight

            self._character = np.random.choice(available_chars, None, p=p/np.sum(p))
            self.history.append(self._character)

    def crossover(self, other_char):
        if np.random.random() < 0.5:
            self._character = other_char._character

    def __repr__(self):
        return self._character.__name__

    def get(self):
        return self._character

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

