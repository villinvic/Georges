import numpy as np
import pickle
import json
import os

from population.individual import Individual
from characters import characters


class Population:
    def __init__(self, size, n_reference=1):

        self.individuals = np.empty((size+n_reference,), dtype=Individual)
        self.size = size
        self.n_reference = n_reference
        self.total_size = n_reference+self.size
        self.checkpoint_index = 0
        self.n = 0

        self.to_serializable_v = np.vectorize(lambda individual: individual.get_all())
        self.read_pickled_v = np.vectorize(lambda individual, x: individual.set_all(x))

    def ranking(self):
        return sorted(self.individuals[:self.size], key=lambda individual: individual.elo())

    def __getitem__(self, item):
        return self.individuals[item]

    def __setitem__(self, key, value):
        self.individuals[key] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.size:
            individual = self.individuals[self.n]
            self.n += 1
            return individual
        else:
            raise StopIteration

    def initialize(self, individual_ids=None, trainable=False, reference_name="20XX", reference_char=characters.Fox):

        if individual_ids is None:
            np.random.shuffle(characters.start_chars)
            n_chars = len(characters.start_chars)
            for ID in range(self.size):
                self.individuals[ID] = Individual(ID, characters.start_chars[ID % n_chars], trainable=trainable)
            for ID_reference in range(self.size, self.total_size):
                self.individuals[ID_reference] = Individual(ID_reference, reference_char, is_cpu=True, name=reference_name)
        else:
            for index, ID in enumerate(individual_ids):
                self.individuals[index] = Individual(ID, characters.Character, trainable=trainable)

    def __repr__(self):
        return self.individuals.__repr__()

    def to_dict(self):
        return {individual.id:individual.to_dict() for individual in self}

    def to_serializable(self):
        return self.to_serializable_v(self.individuals[:self.size])

    def read_pickled(self, params):
        self.read_pickled_v(self.individuals[:self.size], params)

    def save(self, path):
        for individual in self:
            with open(path + individual.name.get() + '.pkl',
                      'wb+') as f:
                pickle.dump(individual.get_all(), f)

        with open(path + 'population.params', 'w') as json_file:
            json.dump({
            "size": int(self.size),
            "n_reference": int(self.n_reference),
            "checkpoint_index": int(self.checkpoint_index),
        }, json_file)

    def load(self, path):
        if path[-1] != '/':
            path += '/'
        _, _, ckpts = next(os.walk(path))
        pop_index = 0
        for ckpt in ckpts:
            if '.pkl' in ckpt:
                try:
                    with open(path + ckpt, 'rb') as f:
                        self[pop_index].set_all(pickle.load(f))
                except Exception as e:
                    print(e)
                pop_index += 1
        try:
            with open(path + 'population.params',
                      'r') as json_file:
                params = json.load(json_file)
            for param_name, value in params.items():
                setattr(self, param_name, value)
        except Exception as e:
            print(e)



