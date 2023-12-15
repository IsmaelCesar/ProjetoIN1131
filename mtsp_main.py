import numpy as np
import logging
from tsp import MTSP
from operations.mutation import SingleTravelerMut
from operations.crossover import SingleTravelerX
from operations.initialization import Initialization
from operations.selection import SelectIndividuals, STSPKElitism, FitnessProportional
from operations.fitness import MinSumFitnessCalculator, MinMaxFitnessCalculator, extract_routes
from population import get_predefined_data, get_data_escolas
from plotting import plot_cities, plot_mtsp_cycles, plot_objective_function
from utils import compute_cycle
from typing import List
from scipy.spatial.distance import cdist


logger = logging.getLogger("tsp."+__name__)

# >> setting up logging >>>>>
#stream_handler = logging.StreamHandler()
#logger.setLevel(logging.INFO)
#logger.addHandler(stream_handler)
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def min_sum(traveler_breaks: List[int]):

    escolas_codigo,escolas_id, coordenadas_escolas = get_data_escolas()
    #escolas_codigo, escolas_id, coordenadas_escolas = get_predefined_data()
    distance_matrix = cdist(coordenadas_escolas, coordenadas_escolas, metric="euclidean")

    plot_cities(coordenadas_escolas, escolas_id, len(escolas_id))

    random_origin = np.random.randint(0, len(escolas_id))

    mtsp = MTSP(n_gen=500, traveler_breaks=traveler_breaks, combine_multiple_x=True, combine_multiple_mut=False)
    pop_size = 14
    mtsp.evolve(
        pop_initializer=Initialization(num_cidades=len(escolas_id), pop_size=pop_size, origin=random_origin),
        crossover_op=SingleTravelerX(crossover_type="order", probability=.8),
        mutation_op=SingleTravelerMut(mutation_type="scramble", probability=.8),
        selection_op=SelectIndividuals(),
        fitness_calculator=MinSumFitnessCalculator(distance_matrix),
        #survivor_selection= STSPKElitism()#FitnessProportional(pop_size=pop_size, num_cidades=len(escolas_id) - 1)
        survivor_selection= FitnessProportional(pop_size=pop_size, num_cidades=len(escolas_id) - 1)
    )

    routes = extract_routes(
                individual=mtsp.statistics["best_individual"][-1], 
                traveler_breaks=traveler_breaks,
                origin=random_origin)

    best_individual = mtsp.statistics["best_individual"][-1]
    best_fitness = mtsp.statistics["best_fitness"][-1]
    logger.info(f"Origin City: {escolas_id[random_origin]}, Origin City Index: {random_origin}")
    logger.info(f"Best individual: {best_individual}")
    logger.info(f"Best individual fitness: {best_fitness}")
    logger.info(f"Traveler routes: \n{routes}")
    logger.info(f"Crossover Operation Counts: \n {mtsp._x_op_counts}")
    logger.info(f"Mutation Operation Counts: \n {mtsp._mut_op_counts}")

    plot_mtsp_cycles(
        coordenadas_cidades=coordenadas_escolas,
        rotas=routes, 
        cidades_codigo=escolas_id,
        origin=random_origin)
    
    plot_objective_function(mtsp.statistics["mean_fitness"], mtsp.statistics["best_fitness"])
    
def min_max(traveler_breaks: List[int]):

    escolas_codigo, escolas_id, coordenadas_escolas = get_data_escolas()
    #escolas_codigo, escolas_id, coordenadas_escolas = get_predefined_data()
    distance_matrix = cdist(coordenadas_escolas, coordenadas_escolas, metric="euclidean")

    #plot_cities(coordenadas_escolas, escolas_id, len(escolas_id))

    random_origin = np.random.randint(0, len(escolas_id))

    mtsp = MTSP(n_gen=1000, traveler_breaks=traveler_breaks, combine_multiple_x=True, combine_multiple_mut=True)
    pop_size = 24
    mtsp.evolve(
        pop_initializer=Initialization(num_cidades=len(escolas_id), pop_size=pop_size, origin=random_origin),
        crossover_op=SingleTravelerX(crossover_type="edge", probability=.5),
        mutation_op=SingleTravelerMut(mutation_type="insert", probability=.3),
        selection_op=SelectIndividuals(),
        fitness_calculator=MinMaxFitnessCalculator(distance_matrix),
        #survivor_selection= STSPKElitism()#FitnessProportional(pop_size=pop_size, num_cidades=len(escolas_id) - 1)
        survivor_selection= FitnessProportional(pop_size=pop_size, num_cidades=len(escolas_id) - 1)
    )

    routes = extract_routes(
                individual=mtsp.statistics["best_individual"][-1], 
                traveler_breaks=traveler_breaks,
                origin=random_origin)

    best_individual = mtsp.statistics["best_individual"][-1]
    best_fitness = mtsp.statistics["best_fitness"][-1]
    logger.info(f"Origin City: {escolas_id[random_origin]}, Origin City Index: {random_origin}")
    logger.info(f"Best individual: {best_individual}")
    logger.info(f"Best individual fitness: {best_fitness}")
    logger.info(f"Traveler routes: \n{routes}")
    logger.info(f"Crossover Operation Counts: \n {mtsp._x_op_counts}")
    logger.info(f"Mutation Operation Counts: \n {mtsp._mut_op_counts}")

    plot_mtsp_cycles(
        coordenadas_cidades=coordenadas_escolas,
        rotas=routes, 
        cidades_codigo=escolas_id,
        origin=random_origin)
    
    plot_objective_function(mtsp.statistics["mean_fitness"], mtsp.statistics["best_fitness"])

if __name__ == "__main__":
    #[1,1,1,1,1,2,2,2,2]
    init = 0
    traveler_breaks = []
    n_travelers = 2
    n_total = 384
    for _ in range(n_travelers):
        init += n_total//n_travelers
        traveler_breaks += [init]

    traveler_breaks[-1] -= 1
    #print(traveler_breaks)

    #traveler_breaks = [4, 8, 13]
    min_sum(traveler_breaks)
    #min_max(traveler_breaks)
