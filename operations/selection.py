import numpy as np
from typing import Tuple, List

class SelectParents:

    def __init__(self, num_parents: int =  2):
        self.num_parents = num_parents
    
    def random(self, individuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly selects two individuals from the individuals array for reproduction.
        """
        pool_range = list(range(len(individuals)))
        (parent1_index, parent2_index) = np.random.choice(pool_range)

        parent1 = individuals[parent1_index]
        parent2 = individuals[parent2_index]

        return parent1, parent2

    def tournament(self, individuals: np.ndarray, fitness: np.ndarray) -> List[np.ndarray, np.ndarray]:
        """
        Select individuals from individuals array for reproduction based on tournament algorithm
        """
        parent_idx = 0
        parents = [None, None]
        pool_range = list(range(len(individuals)))

        while parent_idx < self.num_parents:
            individuals_indices = np.random.choice(pool_range, 3)
            best_one = individuals_indices[0]
            for indiv_idx in individuals_indices[1:]:
                if fitness[best_one] < fitness[indiv_idx]:
                    best_one = indiv_idx
                    pool_range.remove(indiv_idx)
            
            parents[parent_idx] = individuals[best_one]
            parent_idx += 1
        
        return parents