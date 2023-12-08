import sys
sys.path.append("../.")

from tsp import SingleTSP
from operations.mutation import SingleTravelerMut
from operations.crossover import SingleTravelerX
from operations.initialization import Initialization
from operations.selection import SelectIndividuals
from operations.fitness import DistanceFitnessCalculator
from population import get_predefined_data
from scipy.spatial.distance import cdist

def test_evolution():
    cidades_codigo, cidades, coordenadas_cidades = get_predefined_data()
    distance_matrix = cdist(coordenadas_cidades, coordenadas_cidades)

    stsp = SingleTSP(n_gen=2)

    stsp.evolve(
        Initialization(num_cidades=len(cidades), pop_size=10),
        SingleTravelerX(crossover_type="order"),
        SingleTravelerMut(mutation_type="swap"),
        SelectIndividuals(),
        DistanceFitnessCalculator(distance_matrix)
    )

if __name__ == "__main__":
    test_evolution()
