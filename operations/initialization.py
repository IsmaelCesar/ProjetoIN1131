import numpy as np


class Initialization:

    def __init__(self, num_cidades: int, pop_size: int = 200, origin=None):
        self.num_cidades = num_cidades
        self.pop_size = pop_size
        self.origin = origin

    def random(self) -> np.ndarray:
        pop = np.empty((self.pop_size, self.num_cidades), dtype=int)
        for el_idx in range(len(pop)):
            pop[el_idx] = np.random.permutation(self.num_cidades)

        pop = self.remove_origin(pop)

        return pop

    def _remove_origin_from_individual(self, individual: np.ndarray) -> np.ndarray:
        # if there is an origin, removeit from 
        # recently added permutation
        if self.origin is not None: 
            origin_idx = np.where(individual == self.origin)[0][0]
            individual = np.delete(individual, origin_idx)

        return individual

    def remove_origin(self, population: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self._remove_origin_from_individual, 1, population)
