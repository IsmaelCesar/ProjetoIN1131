from operations.fitness import distance_fitness
from population import get_predefined_data
from scipy.spatial.distance import cdist


def test_distance_fitness(): 

    cidades_codigo, cidades, coordenadas_cidades = get_predefined_data()
    distance_matrix = cdist(coordenadas_cidades, coordenadas_cidades, metric="euclidean")

    print("Individual: ", cidades )

    fit = distance_fitness(cidades, distance_matrix)
    print("Raw individual fitness: ", fit)

if __name__ == "__main__":
    test_distance_fitness()
