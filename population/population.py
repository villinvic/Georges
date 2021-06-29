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

        self.to_serializable_v = np.vectorize(lambda individual: individual.get_all())
        self.read_pickled_v = np.vectorize(lambda individual, x: individual.set_all(x))

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

    def to_serializable(self):
        return self.to_serializable_v(self.individuals[:self.size])

    def read_pickled(self, params):
        self.read_pickled_v(self.individuals[:self.size], params)

    def save(self, path):
        pass



