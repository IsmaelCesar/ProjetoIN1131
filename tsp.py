import numpy as np
from operations.mutation import SingleTravelerMut
from operations.crossover import SingleTravelerX
from operations.initialization import Initialization
from operations.selection import SelectIndividuals
from operations.fitness import DistanceFitnessCalculator
from typing import Tuple

class SingleTSP:
    """
    This class implements single traveling salesmans problem
    """

    def __init__(self, n_gen: int) -> None:
        self.n_gen = n_gen
        self.statistics = {
            "mean_fitness": [],
            "best_overall": [],
            "best_individual": []
        }

    def sort_fitness(self, individuals: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sort individuals based on fitness
        """
        indiv_fit = np.concatenate((individuals, fitness[:, np.newaxis]), axis=1)
        indiv_fit = np.array(sorted(indiv_fit, key=lambda x: x[-1]))
        
        individuals = indiv_fit[:, :-1].astype(int)
        fitness = indiv_fit[:, -1:]

        return individuals, fitness
    
    def evolve(self,
               pop_initializer: Initialization,
               crossover_op: SingleTravelerX,
               mutation_op: SingleTravelerMut, 
               selection_op: SelectIndividuals,
               fitness_calculator: DistanceFitnessCalculator):

        pop_size = pop_initializer.pop_size
        #num_cidades = pop_initializer.num_cidades

        population = pop_initializer.random()
        fitness = np.apply_along_axis(fitness_calculator.distance_fitness, 1, population)
        population, fitness = self.sort_fitness(population, fitness)

        for gen_idx in range(self.n_gen):
            new_population = []

            # new population creation
            for _ in range(0, pop_size // 2):
                parent_1, parent_2 = selection_op.apply(population, fitness)
                child_1, child_2 = crossover_op.apply(parent_1, parent_2)
                mchild_1 = mutation_op.apply(child_1)
                mchild_2 = mutation_op.apply(child_2)
                new_population += [mchild_1, mchild_2]

            new_population = np.array(new_population, dtype=int)
            new_fitness = np.apply_along_axis(fitness_calculator.distance_fitness, 1, new_population)




            
