from tsp import SingleTSP
from operations.mutation import SingleTravelerMut
from operations.crossover import SingleTravelerX
from operations.initialization import Initialization
from operations.selection import SelectIndividuals, STSPKElitism
from operations.fitness import DistanceFitnessCalculator
from population import get_predefined_data
from scipy.spatial.distance import cdist

def main():
    cidades_codigo, cidades, coordenadas_cidades = get_predefined_data()
    distance_matrix = cdist(coordenadas_cidades, coordenadas_cidades)

    stsp = SingleTSP(n_gen=200)

    stsp.evolve(
        Initialization(num_cidades=len(cidades), pop_size=100),
        SingleTravelerX(crossover_type="order"),
        SingleTravelerMut(mutation_type="inverse"),
        SelectIndividuals(),
        DistanceFitnessCalculator(distance_matrix),
        STSPKElitism()
    )

if __name__ == "__main__":
    main()
