import numpy as np
from operations.mutation import SingleTravelerMut
from operations.crossover import SingleTravelerX
from operations.initialization import Initialization
from operations.selection import SelectIndividuals
from operations.fitness import DistanceFitnessCalculator

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
    
    def evolve(self,
               pop_initializer: Initialization,
               crossover_op: SingleTravelerX,
               mutation_op: SingleTravelerMut, 
               selection_op: SelectIndividuals,
               fitness_calculator: DistanceFitnessCalculator):

        population = pop_initializer.random()
        fitness = np.apply_along_axis(fitness_calculator.distance_fitness, 1, population)

        for gen_idx in range(self.n_gen):
            new_population = np.empty((pop_initializer.pop_size, pop_initializer.num_cidades))
            
