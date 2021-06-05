import numpy as np

from population.individual import Individual
from characters import characters


class Population:
    def __init__(self, size, n_reference=1):

        self.individuals = np.empty((size+n_reference,), dtype=Individual)
        self.size = size
        self.n_reference = n_reference
        self.total_size = n_reference+self.size
        self.n = 0

    def ranking(self):
        return sorted(self.individuals, key=lambda individual: individual.elo())

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

    def initialize(self):

        np.random.shuffle(characters.start_chars)
        n_chars = len(characters.start_chars)
        for ID in range(self.size):
            self.individuals[ID] = Individual(ID, characters.start_chars[ID % n_chars])
        for ID_reference in range(self.size, self.total_size):
            self.individuals[ID_reference] = Individual(ID_reference, characters.Fox, is_cpu=True, name='20XX')

    def __repr__(self):
        return self.individuals.__repr__()



