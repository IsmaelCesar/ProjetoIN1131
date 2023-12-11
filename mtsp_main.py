import numpy as np
import logging
from tsp import MTSP
from operations.mutation import SingleTravelerMut
from operations.crossover import SingleTravelerX
from operations.initialization import Initialization
from operations.selection import SelectIndividuals, STSPKElitism
from operations.fitness import MinSumFitnessCalculator, MinMaxFitnessCalculator, extract_routes
from population import get_predefined_data
from plotting import plot_cities, plot_mtsp_cycles, plot_objective_function
from utils import compute_cycle
from typing import List
from scipy.spatial.distance import cdist


logger = logging.getLogger("tsp_ga")

# >> setting up logging >>>>>
stream_handler = logging.StreamHandler()

logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def min_sum(traveler_breaks: List[int]):

    cidades_codigo, cidades, coordenadas_cidades = get_predefined_data()
    distance_matrix = cdist(coordenadas_cidades, coordenadas_cidades)

    plot_cities(coordenadas_cidades, cidades_codigo, len(cidades))

    random_origin = 2 #np.random.randint(0, len(cidades))

    mtsp = MTSP(n_gen=1000, traveler_breaks=traveler_breaks)
    mtsp.evolve(
        pop_initializer=Initialization(num_cidades=len(cidades), pop_size=200, origin=random_origin),
        crossover_op=SingleTravelerX(crossover_type="order", probability=.8),
        mutation_op=SingleTravelerMut(mutation_type="scramble", probability=.8),
        selection_op=SelectIndividuals(),
        fitness_calculator=MinSumFitnessCalculator(distance_matrix),
        k_elitism=STSPKElitism()
    )

    routes = extract_routes(
                individual=mtsp.statistics["best_individual"][-1], 
                traveler_breaks=traveler_breaks,
                origin=random_origin)

    best_individual = mtsp.statistics["best_individual"][-1]
    best_fitness = mtsp.statistics["best_fitness"][-1]
    logger.info(f"Origin City: {cidades_codigo[random_origin]}, Origin City Index: {random_origin}")
    logger.info(f"Best individual: {best_individual}")
    logger.info(f"Best individual fitness: {best_fitness}")
    logger.info(f"Traveler routes: \n{routes}")
    
    plot_mtsp_cycles(
        coordenadas_cidades=coordenadas_cidades,
        rotas=routes, 
        cidades_codigo=cidades_codigo,
        origin=random_origin)
    
    plot_objective_function(mtsp.statistics["mean_fitness"], mtsp.statistics["best_fitness"])
    
def min_max(traveler_breaks: List[int]):

    cidades_codigo, cidades, coordenadas_cidades = get_predefined_data()
    distance_matrix = cdist(coordenadas_cidades, coordenadas_cidades)

    plot_cities(coordenadas_cidades, cidades_codigo, len(cidades))

    random_origin = 2 #np.random.randint(0, len(cidades))

    mtsp = MTSP(n_gen=1000, traveler_breaks=traveler_breaks)
    mtsp.evolve(
        pop_initializer=Initialization(num_cidades=len(cidades), pop_size=200, origin=random_origin),
        crossover_op=SingleTravelerX(crossover_type="order", probability=.8),
        mutation_op=SingleTravelerMut(mutation_type="scramble", probability=.8),
        selection_op=SelectIndividuals(),
        fitness_calculator=MinMaxFitnessCalculator(distance_matrix),
        k_elitism=STSPKElitism()
    )

    routes = extract_routes(
                individual=mtsp.statistics["best_individual"][-1], 
                traveler_breaks=traveler_breaks,
                origin=random_origin)

    best_individual = mtsp.statistics["best_individual"][-1]
    best_fitness = mtsp.statistics["best_fitness"][-1]
    logger.info(f"Origin City: {cidades_codigo[random_origin]}, Origin City Index: {random_origin}")
    logger.info(f"Best individual: {best_individual}")
    logger.info(f"Best individual fitness: {best_fitness}")
    logger.info(f"Traveler routes: \n{routes}")
    
    plot_mtsp_cycles(
        coordenadas_cidades=coordenadas_cidades,
        rotas=routes, 
        cidades_codigo=cidades_codigo,
        origin=random_origin)
    
    plot_objective_function(mtsp.statistics["mean_fitness"], mtsp.statistics["best_fitness"])

if __name__ == "__main__":
    traveler_breaks = [4, 8, 13]
    min_sum(traveler_breaks)
    min_max(traveler_breaks)
