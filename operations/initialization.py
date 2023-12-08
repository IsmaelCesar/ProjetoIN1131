import numpy as np


class Initialization:

    def __init__(self, num_cidades: int, pop_size: int = 200):
        self.num_cidades = num_cidades
        self.pop_size = pop_size
    
    def random(self) -> np.ndarray:
        pop = np.empty((self.pop_size, self.num_cidades), dtype=int)
        for el_idx in range(len(pop)):
            pop[el_idx] = np.random.permutation(self.num_cidades)
        return pop
