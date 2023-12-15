import os
import numpy as np
import logging
from tsp import MTSP
from operations.mutation import SingleTravelerMut
from operations.crossover import SingleTravelerX
from operations.initialization import Initialization
from operations.selection import SelectIndividuals, STSPKElitism, FitnessProportional
from operations.fitness import MinSumFitnessCalculator, MinMaxFitnessCalculator, extract_routes
from population import get_data_cidades
from plotting import plot_cities, plot_mtsp_cycles, plot_objective_function
from utils import save_statistics_as_json, compute_traveler_breaks, check_create_dir
from typing import List, Tuple
from scipy.spatial.distance import cdist


logger = logging.getLogger("tsp")

# >> setting up logging >>>>>
#stream_handler = logging.StreamHandler()
#logger.setLevel(logging.INFO)
#logger.addHandler(stream_handler)
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def min_sum(
        traveler_breaks: List[int],
        cidades_id: List[int],
        coordenadas_cidades: np.ndarray,
        n_gen: int,
        pop_size: int,
        execution_index: int,
        op_probabilities: Tuple[float, float] = (.8, .2),
        combine_operations: bool = True,
        method_name: str = "min_sum",
        results_dir: str = "results"):
    
    # handling directories <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    method_dir = os.path.join(results_dir, method_name)
    check_create_dir(method_dir)

    min_sum_results_dir = os.path.join(method_dir, f"execution_{execution_index}")
    check_create_dir(min_sum_results_dir)

    method_file_handler = logging.FileHandler(f"{min_sum_results_dir}/{method_name}.log", mode="w+")
    logger.addHandler(method_file_handler)
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    distance_matrix = cdist(coordenadas_cidades, coordenadas_cidades, metric="euclidean")

    random_origin = np.random.randint(0, len(cidades_id))

    mtsp = MTSP(n_gen=n_gen, traveler_breaks=traveler_breaks, combine_multiple_x=combine_operations, combine_multiple_mut=combine_operations)
    pop_size = pop_size
    mtsp.evolve(
        pop_initializer=Initialization(num_cidades=len(cidades_id), pop_size=pop_size, origin=random_origin),
        crossover_op=SingleTravelerX(crossover_type="order", probability=op_probabilities[0]),
        mutation_op=SingleTravelerMut(mutation_type="scramble", probability=op_probabilities[1]),
        selection_op=SelectIndividuals(),
        fitness_calculator=MinSumFitnessCalculator(distance_matrix),
        #survivor_selection= STSPKElitism()#FitnessProportional(pop_size=pop_size, num_cidades=len(escolas_id) - 1)
        survivor_selection= FitnessProportional(pop_size=pop_size, num_cidades=len(cidades_id) - 1)
    )

    routes = extract_routes(
                individual=np.array(mtsp.statistics["best_individual"][-1]),
                traveler_breaks=traveler_breaks,
                origin=random_origin)

    best_individual = mtsp.statistics["best_individual"][-1]
    best_fitness = mtsp.statistics["best_fitness"][-1]
    logger.info(f"Origin City: {cidades_id[random_origin]}, Origin City Index: {random_origin}")
    logger.info(f"Best individual: {best_individual}")
    logger.info(f"Best individual fitness: {best_fitness}")
    logger.info(f"Traveler routes: \n{routes}")
    logger.info(f"Crossover Operation Counts: \n {mtsp._x_op_counts}")
    logger.info(f"Mutation Operation Counts: \n {mtsp._mut_op_counts}")

    plot_mtsp_cycles(
        coordenadas_cidades=coordenadas_cidades,
        rotas=routes, 
        cidades_codigo=cidades_id,
        origin=random_origin,
        filename=f"{min_sum_results_dir}/cycle.png",
        show_plot=False)
    
    plot_objective_function(
        mtsp.statistics["mean_fitness"], 
        mtsp.statistics["best_fitness"],
        filename=f"{min_sum_results_dir}/objective_fn.png",
        show_plot=False)

    save_statistics_as_json(mtsp.statistics, f"{min_sum_results_dir}/minsum_statistics.json")
    save_statistics_as_json(mtsp._x_op_counts, f"{min_sum_results_dir}/minsum_op_counts_crossover.json")
    save_statistics_as_json(mtsp._mut_op_counts, f"{min_sum_results_dir}/minsum_op_counts_mutation.json")
    
def min_max(
        traveler_breaks: List[int],
        cidades_id: List[int],
        coordenadas_cidades: np.ndarray,
        n_gen: int,
        pop_size: int,
        execution_index: int,
        op_probabilities: Tuple[float, float] = (.8, .2),
        combine_operations: bool = True,
        method_name: str = "min_max",
        results_dir: str = "results"):
    
    # handling directories <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    method_dir = os.path.join(results_dir, method_name)
    check_create_dir(method_dir)

    min_max_results_dir = os.path.join(method_dir, f"execution_{execution_index}")
    check_create_dir(min_max_results_dir)

    method_file_handler = logging.FileHandler(f"{min_max_results_dir}/{method_name}.log", mode="w+")
    logger.addHandler(method_file_handler)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    distance_matrix = cdist(coordenadas_cidades, coordenadas_cidades, metric="euclidean")

    random_origin = np.random.randint(0, len(cidades_id))

    mtsp = MTSP(n_gen=n_gen, traveler_breaks=traveler_breaks, combine_multiple_x=combine_operations, combine_multiple_mut=combine_operations)
    pop_size = pop_size
    mtsp.evolve(
        pop_initializer=Initialization(num_cidades=len(cidades_id), pop_size=pop_size, origin=random_origin),
        crossover_op=SingleTravelerX(crossover_type="edge", probability=op_probabilities[0]),
        mutation_op=SingleTravelerMut(mutation_type="insert", probability=op_probabilities[1]),
        selection_op=SelectIndividuals(),
        fitness_calculator=MinMaxFitnessCalculator(distance_matrix),
        #survivor_selection= STSPKElitism()#FitnessProportional(pop_size=pop_size, num_cidades=len(escolas_id) - 1)
        survivor_selection= FitnessProportional(pop_size=pop_size, num_cidades=len(cidades_id) - 1)
    )

    routes = extract_routes(
                individual=np.array(mtsp.statistics["best_individual"][-1]),
                traveler_breaks=traveler_breaks,
                origin=random_origin)

    best_individual = mtsp.statistics["best_individual"][-1]
    best_fitness = mtsp.statistics["best_fitness"][-1]
    logger.info(f"Origin City: {cidades_id[random_origin]}, Origin City Index: {random_origin}")
    logger.info(f"Best individual: {best_individual}")
    logger.info(f"Best individual fitness: {best_fitness}")
    logger.info(f"Traveler routes: \n{routes}")
    logger.info(f"Crossover Operation Counts: \n {mtsp._x_op_counts}")
    logger.info(f"Mutation Operation Counts: \n {mtsp._mut_op_counts}")

    plot_mtsp_cycles(
        coordenadas_cidades=coordenadas_cidades,
        rotas=routes, 
        cidades_codigo=cidades_id,
        origin=random_origin,
        filename=f"{min_max_results_dir}/cycle.png",
        show_plot=False)
    
    plot_objective_function(
        mtsp.statistics["mean_fitness"], 
        mtsp.statistics["best_fitness"],
        filename=f"{min_max_results_dir}/objective_fn.png",
        show_plot=False)

    save_statistics_as_json(mtsp.statistics, f"{min_max_results_dir}/minmax_statistics.json")
    save_statistics_as_json(mtsp.statistics, f"{min_max_results_dir}/minmax_statistics.json")
    save_statistics_as_json(mtsp._x_op_counts, f"{min_max_results_dir}/minmax_op_counts_crossover.json")
    save_statistics_as_json(mtsp._mut_op_counts, f"{min_max_results_dir}/minmax_op_counts_mutation.json")

if __name__ == "__main__":

    RESULTS_DIR  = "results/"

    # configuring logging <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    main_logger = logging.getLogger("mtsp_main")
    main_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"{RESULTS_DIR}/experiment.log", mode="w+")
    stream_handler = logging.StreamHandler()
    
    main_logger.addHandler(file_handler)
    main_logger.addHandler(stream_handler)
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    cidades_codigo, cidades_id, coordenadas_cidades = get_data_cidades()

    traveler_breaks = compute_traveler_breaks(3, len(cidades_id))

    main_logger.info(f"Number of cities: {len(cidades_id)}")
    main_logger.info(f"Traveler breaks: {traveler_breaks}")

    n_gen = 1000
    pop_size = 10
    combine_operations = True
    op_probabilities = (.7, .3) # probabilitys of crossover and mutation to occur

    main_logger.info(f"Total Generations: {n_gen}")
    main_logger.info(f"Population size: {pop_size}")

    n_executions = 30

    for exec_idx in range(n_executions):
        main_logger.info(f"Running Min-Sum Execution {exec_idx}")
        min_sum(
            traveler_breaks=traveler_breaks,
            cidades_id=cidades_id,
            coordenadas_cidades=coordenadas_cidades,
            n_gen=n_gen,
            pop_size=pop_size,
            execution_index=exec_idx,
            op_probabilities=op_probabilities,
            combine_operations=combine_operations)

        main_logger.info(f"Running Min-Max Execution {exec_idx}")

        min_max(
            traveler_breaks=traveler_breaks,
            cidades_id=cidades_id,
            coordenadas_cidades=coordenadas_cidades,
            n_gen=n_gen,
            pop_size=pop_size,
            execution_index=exec_idx,
            op_probabilities=op_probabilities,
            combine_operations=combine_operations)
