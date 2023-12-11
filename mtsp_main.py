import numpy as np
import logging
from tsp import MTSP
from operations.mutation import SingleTravelerMut
from operations.crossover import SingleTravelerX
from operations.initialization import Initialization
from operations.selection import SelectIndividuals, STSPKElitism
from operations.fitness import MinSumFitnessCalculator, MinMaxFitnessCalculator, extract_routes
from population import get_predefined_data
from plotting import plot_cities, plot_cycle, plot_objective_function
from utils import compute_cycle
from typing import List
from scipy.spatial.distance import cdist


logger = logging.getLogger("tsp_ga")


def min_sum(traveler_breaks: List[int]):

    # >> setting up logging >>>>>
    stream_handler = logging.StreamHandler()
    
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    cidades_codigo, cidades, coordenadas_cidades = get_predefined_data()
    distance_matrix = cdist(coordenadas_cidades, coordenadas_cidades)

    #plot_cities(coordenadas_cidades, cidades_codigo, len(cidades))

    random_origin = np.random.randint(0, len(cidades))

    mtsp = MTSP(n_gen=100, traveler_breaks=traveler_breaks)
    mtsp.evolve(
        pop_initializer=Initialization(num_cidades=10, pop_size=10, origin=random_origin),
        crossover_op=SingleTravelerX(crossover_type="order"),
        mutation_op=SingleTravelerMut(mutation_type="inverse"),
        selection_op=SelectIndividuals(),
        fitness_calculator=MinSumFitnessCalculator(distance_matrix),
        k_elitism=STSPKElitism()
    )

    routes = extract_routes(
                individual=mtsp.statistics["best_individual"][-1], 
                traveler_breaks=traveler_breaks,
                origin=random_origin)
    

def min_max(traveler_breaks: List[int]):
    print("--"*50)
    print("Min-Max")

if __name__ == "__main__":
    traveler_breaks = [4, 8, 12]
    min_sum(traveler_breaks)
    min_max(traveler_breaks)
