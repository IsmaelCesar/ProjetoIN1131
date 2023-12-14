import logging
import numpy as np
from operations.mutation import SingleTravelerMut
from operations.crossover import SingleTravelerX
from operations.initialization import Initialization
from operations.selection import SelectIndividuals, STSPKElitism
from operations.fitness import DistanceFitnessCalculator, MTSPFitnessCalculator
from typing import Tuple, List

logger = logging.getLogger("main.tsp_ga")

# >> setting up logging >>>>>
stream_handler = logging.StreamHandler()

logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class SingleTSP:
    """
    This class implements single traveling salesmans problem
    """

    def __init__(self, n_gen: int) -> None:
        self.n_gen = n_gen
        self.statistics = {
            "mean_fitness": [],
            "std_fitness": [],
            "best_overall": [],
            "best_individual": [],
            "best_fitness": [],
        }

    def sort_fitness(self, individuals: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sort individuals based on fitness
        """
        indiv_fit = np.concatenate((individuals, fitness[:, np.newaxis]), axis=1)
        indiv_fit = np.array(sorted(indiv_fit, key=lambda x: x[-1]))
        
        individuals = indiv_fit[:, :-1].astype(int)

        # remove o eixo adicional introduzido por np.newaxis
        fitness = indiv_fit[:, -1:].reshape(-1)

        return individuals, fitness

    def save_statistics(self, individuals: np.ndarray, fitness: np.ndarray) -> None:

        min_fitness_idx = fitness.argmin()
        
        self.statistics["best_individual"].append(individuals[min_fitness_idx].copy())
        self.statistics["best_fitness"].append(fitness[min_fitness_idx])
        self.statistics["mean_fitness"].append(fitness.mean())
        self.statistics["std_fitness"].append(fitness.std())
    
    def evolve(self,
               pop_initializer: Initialization,
               crossover_op: SingleTravelerX,
               mutation_op: SingleTravelerMut, 
               selection_op: SelectIndividuals,
               fitness_calculator: DistanceFitnessCalculator,
               k_elitism: STSPKElitism):

        pop_size = pop_initializer.pop_size

        population = pop_initializer.random()
        fitness = np.apply_along_axis(fitness_calculator.distance_fitness, 1, population)
        #population, fitness = self.sort_fitness(population, fitness)

        self.save_statistics(population, fitness)

        for gen_idx in range(self.n_gen):
            
            best_overall = self.statistics["best_fitness"][-1]
            logger.info(f"Generation {gen_idx}, Best overall solution: {best_overall}")

            new_population = []

            # new population creation
            for _ in range(0, pop_size // 2):
                parent_1, parent_2 = selection_op.apply(population, fitness)
                child_1, child_2 = crossover_op.apply(parent_1, parent_2)
                mchild_1 = mutation_op.apply(child_1)
                mchild_2 = mutation_op.apply(child_2)
                new_population += [mchild_1, mchild_2]

            # computing new fitness
            new_population = np.array(new_population, dtype=int)
            new_fitness = np.apply_along_axis(fitness_calculator.distance_fitness, 1, new_population)
            
            # sorting by fitness
            #new_population, new_fitness = self.sort_fitness(new_population, new_fitness)

            population, fitness = k_elitism.apply(population, fitness, new_population, new_fitness)

            self.save_statistics(population, fitness)

class MTSP:

    def __init__(self, n_gen: int, traveler_breaks: List[int], combine_multiple_x: bool=False, combine_multiple_mut: bool = False):
        """
        Genetic algorithm for the Multi Traveling Salesmen problem.

        Parameters:
        -----------
        n_gen: int the total number of gneerations
        traveler_breaks: List[int] tell wich segment belongs to which traveler
        combine_multiple_x: bool Tell the algorithm whether or not to probabilistcally choose among the multiple
                                 crossover operations
        combine_multiple_mut: bool Tell the algorithm whether or not to probabilistcally choose among the multiple
                                   mutations operations

        """

        self.n_gen = n_gen
        self.traveler_breaks = traveler_breaks
        self.combine_multiple_x = combine_multiple_x
        self.combine_multiple_mut = combine_multiple_mut

        self.statistics = {
            "mean_fitness": [],
            "std_fitness": [],
            "best_overall": [],
            "best_individual": [],
            "best_fitness": [],
        }

    def save_statistics(self, individuals: np.ndarray, fitness: np.ndarray) -> None:

        min_fitness_idx = fitness.argmin()
        
        self.statistics["best_individual"].append(individuals[min_fitness_idx].copy())
        self.statistics["best_fitness"].append(fitness[min_fitness_idx])
        self.statistics["mean_fitness"].append(fitness.mean())
        self.statistics["std_fitness"].append(fitness.std())

    def _monitor_crossover_operations(self, crossover_op: SingleTravelerX, parent1:np.ndarray, parent2: np.ndarray ) -> Tuple[np.ndarray, np.ndarray]:
        if self.combine_multiple_x:
            print("probabilistically choosing one of the multiple mutations")
        
        return crossover_op.apply(parent1, parent2)


    def _monitor_mutation_operations(self, mutation_op: SingleTravelerMut, child: np.ndarray) -> np.ndarray:
        
        if self.combine_multiple_mut:
            print("Probabilistically chosing one of the multiple mutations")
        
        return mutation_op.apply(child)

    def evolve(self,
               pop_initializer: Initialization,
               crossover_op: SingleTravelerX,
               mutation_op: SingleTravelerMut, 
               selection_op: SelectIndividuals,
               fitness_calculator: MTSPFitnessCalculator,
               survivor_selection: STSPKElitism):

        pop_size = pop_initializer.pop_size

        population = pop_initializer.random()
        fitness = np.apply_along_axis(fitness_calculator.distance_fitness, 1, population, self.traveler_breaks)

        self.save_statistics(population, fitness)
        #population, fitness = self.sort_fitness(population, fitness)

        for gen_idx in range(self.n_gen):
            
            best_overall = self.statistics["best_fitness"][-1]
            logger.info(f"Generation {gen_idx}, Best overall solution: {best_overall}")

            new_population = []

            # new population creation
            for _ in range(0, pop_size // 2):
                parent_1, parent_2 = selection_op.apply(population, fitness)
                child_1, child_2 = crossover_op.apply(parent_1, parent_2)
                mchild_1 = mutation_op.apply(child_1)
                mchild_2 = mutation_op.apply(child_2)
                new_population += [mchild_1, mchild_2]

            # computing new fitness
            new_population = np.array(new_population, dtype=int)
            new_fitness = np.apply_along_axis(fitness_calculator.distance_fitness, 1, new_population, self.traveler_breaks)
            
            # sorting by fitness
            #new_population, new_fitness = self.sort_fitness(new_population, new_fitness)

            population, fitness = survivor_selection.apply(population, fitness, new_population, new_fitness)

            self.save_statistics(population, fitness)

