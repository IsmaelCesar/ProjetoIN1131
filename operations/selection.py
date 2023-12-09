import numpy as np
from typing import Tuple, List

class SelectIndividuals:

    def __init__(
            self, 
            num_individuals: int =  2, 
            selection_type: str = "tournament",
            scale_fitness: bool = True):

        assert selection_type in ["random", "tournament", "roulette"]

        self.num_individuals = num_individuals
        self.selection_type = selection_type
        self.scale_fitness = scale_fitness

    def _scale_fitness(self, fitness: np.ndarray) -> np.ndarray:
        if self.scale_fitness:
            # scales fitness values to they all sum up to one
            temp_fit = np.array([(f_i - fitness.min())/ (fitness.max() - fitness.min()) for f_i in fitness])
        else: 
            temp_fit = fitness
        return temp_fit

    def random(self, individuals: np.ndarray) -> List[np.ndarray]:
        """
        Randomly selects two individuals from the individuals array for reproduction.
        """
        pool_range = list(range(len(individuals)))
        parents = [None] * self.num_individuals
        
        for  p_idx, selected_p_idx in enumerate(np.random.choice(pool_range, self.num_individuals)):
            parents[p_idx] = individuals[selected_p_idx]
        
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
            fitness: np.ndarray) -> List[np.ndarray]:
        """
        Applies roulette wheel algorithm for selecting parents.
        """

        parents = [None] * self.num_individuals
        parent_idx = 0
        pool_range = list(range(len(individuals)))

        temp_fit = self._scale_fitness(fitness)

        while parent_idx < len(parents):

            random_prob = np.random.uniform(low=0, high=1)
            indiv_count = 0 
            while temp_fit[pool_range[indiv_count]] < random_prob: 
                indiv_count += 1

            parents[parent_idx] = individuals[parent_idx]
            parent_idx += 1
        
        return parents

    def apply(self, individuals: np.ndarray, fitness: np.ndarray) -> List[np.ndarray]:

        if self.selection_type == "random": 
            return self.random(individuals)
        elif self.selection_type == "tournament":
            return self.tournament(individuals, fitness)
        elif self.selection_type == "roulette":
            return self.roulette_wheel(individuals, fitness)

class STSPKElitism:
    """
    Esta classe implementa o k-elitismo pars single
    traveling salesman problem,
    onde a mesma garante que os k-melhores individuos
    da população anterior passem para geração seguinte
    """
    def __init__(self, k: int = 1):
        self.k = k

    def apply(
            self, 
            old_population: np.ndarray, 
            old_fitness: np.ndarray, 
            new_population: np.ndarray, 
            new_fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Effectivelly applies K-elitism
        """

        pop_size = len(old_population)
        num_var = len(old_population[0])        

        # getting best k of old population
        best_k = old_fitness.argsort()[:self.k]
        
        # eliminating the worst k of new population
        rest_of_new = new_fitness.argsort()[::-1][self.k:]

        updated_pop = np.empty((pop_size, num_var), dtype=int)
        updated_fit = np.empty((pop_size,))
        
        updated_pop[:self.k] = old_population[best_k]
        updated_fit[:self.k] = old_fitness[best_k]

        updated_pop[self.k:] = new_population[rest_of_new]
        updated_fit[self.k:] = new_fitness[rest_of_new]

        return updated_pop, updated_fit
