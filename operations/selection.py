import numpy as np
from typing import Tuple, List

class SelectIndividuals:

    def __init__(self, num_individuals: int =  2):
        self.num_individuals = num_individuals
    
    def random(self, individuals: np.ndarray) -> List[np.ndarray]:
        """
        Randomly selects two individuals from the individuals array for reproduction.
        """
        pool_range = list(range(len(individuals)))
        parents = [None] * self.num_individuals
        
        for  p_idx, selected_p in enumerate(np.random.choice(pool_range, self.num_individuals)):
            parents[p_idx] = selected_p
        
        return parents

    def tournament(self, individuals: np.ndarray, fitness: np.ndarray) -> List[np.ndarray]:
        """
        Select individuals from individuals array for reproduction based on tournament algorithm
        """
        parent_idx = 0
        parents = [None] * self.num_individuals
        pool_range = list(range(len(individuals)))

        while parent_idx < len(parents):
            individuals_indices = np.random.choice(pool_range, 3)
            best_one = individuals_indices[0]
            for indiv_idx in individuals_indices[1:]:
                if fitness[best_one] < fitness[indiv_idx]:
                    best_one = indiv_idx
                    pool_range.remove(indiv_idx)
            
            parents[parent_idx] = individuals[best_one]
            parent_idx += 1
        
        return parents

    def roulette_wheel(
            self, 
            individuals: np.ndarray, 
            fitness: np.ndarray,
            scale_fitness: bool = True) -> List[np.ndarray]:
        """
        Applies roulette wheel algorithm for selecting parents.
        """

        parents = [None] * self.num_individuals
        parent_idx = 0
        pool_range = list(range(len(individuals)))

        if scale_fitness:
            # scales fitness values to they all sum up to one
            temp_fit = fitness / fitness.sum()
        else: 
            temp_fit = scale_fitness

        while parent_idx < len(parents):

            random_prob = np.random.uniform(low= 0., high=1 / len(individuals))
            indiv_count = 0 
            while temp_fit[pool_range[indiv_count]] < random_prob: 
                indiv_count += 1

            parents[parent_idx]
            pool_range.remove(indiv_count)
            parent_idx += 1
        
        return parents
