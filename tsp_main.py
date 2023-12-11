import numpy as np
import logging
from tsp import SingleTSP
from operations.mutation import SingleTravelerMut
from operations.crossover import SingleTravelerX
from operations.initialization import Initialization
from operations.selection import SelectIndividuals, STSPKElitism
from operations.fitness import DistanceFitnessCalculator
from population import get_predefined_data
from plotting import plot_cities, plot_cycle, plot_objective_function
from utils import compute_cycle
from scipy.spatial.distance import cdist

logger = logging.getLogger("tsp_ga")

def main():

    # >> setting up logging >>>>>
    stream_handler = logging.StreamHandler()
    
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    cidades_codigo, cidades, coordenadas_cidades = get_predefined_data()
    distance_matrix = cdist(coordenadas_cidades, coordenadas_cidades)

    stsp = SingleTSP(n_gen=1000)

    plot_cities(coordenadas_cidades, cidades_codigo, len(cidades))

    stsp.evolve(
        Initialization(num_cidades=len(cidades), pop_size=200),
        SingleTravelerX(crossover_type="cycle", probability=.8),
        SingleTravelerMut(mutation_type="scramble", probability=.8),
        SelectIndividuals(),
        DistanceFitnessCalculator(distance_matrix),
        STSPKElitism()
    )

    best_individual = stsp.statistics["best_individual"][-1]

    logger.info(f"Best individual: {best_individual}")

    #computed_cycle = compute_cycle(best_individual, coordenadas_cidades)
    cycle = np.zeros((len(best_individual) + 1, ), dtype=int) - 1
    cycle[:-1] = best_individual
    cycle[-1] = best_individual[0]

    plot_cycle(coordenadas_cidades, cycle, cidades_codigo)
    plot_objective_function(stsp.statistics["mean_fitness"], stsp.statistics["best_fitness"])

if __name__ == "__main__":
    main()
