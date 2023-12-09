import numpy as np
from operations.fitness import MinMaxFitnessCalculator, _compute_distance_fitness
from scipy.spatial.distance import cdist

def test_minmax_fitness():

    individual = np.array([5, 6, 4, 0, 3, 1, 2, 7, 10, 8, 9], dtype=int)
    positions = np.array([[4., 1.],
                          [4., 5.],
                          [9., 8.],
                          [1., 1.],
                          [1., 1.],
                          [1., 1.],
                          [1., 1.],
                          [1., 1.],
                          [7., 5.],
                          [6., 7.],
                          [3., 4.]])
    
    distance_matrix = cdist(positions, positions)
    breaks = [3, 8, 11]
    cmp_dist = _compute_distance_fitness(individual[3:8], distance_matrix)

    fitness_calculator = MinMaxFitnessCalculator(distance_matrix)

    print("--"*50)
    print("Individual: ", individual)
    print("Positions: \n", positions)
    fitness = fitness_calculator.distance_fitness(individual, breaks)
    print("Fitness: ", fitness)
    print("CMP Fitness: ", fitness == cmp_dist)
    print("--"*50)

def test_application_of_fitness_over_an_axis():

    individuals = np.array([[5, 6, 4, 0, 3, 1, 2, 7, 10, 8, 9],
                            [4, 0, 2, 7, 3, 1, 9, 5, 6, 10, 8],
                            [4, 0, 2, 7, 3, 1, 5, 6, 10, 8, 9]], dtype=int)
    positions = np.random.rand(len(individuals[0]), 2)
    
    distance_matrix = cdist(positions, positions)
    breaks = [3, 8, 11]

    fitness_calculator = MinMaxFitnessCalculator(distance_matrix)

    print("--"*50)
    print("Individuals: \n", individuals)
    print("Positions: \n", positions)
    fitness = np.apply_along_axis(fitness_calculator.distance_fitness, 1, individuals, breaks)
    print("Fitness: ", fitness)
    print("--"*50)

if __name__ == "__main__":
    test_minmax_fitness()
    test_application_of_fitness_over_an_axis()
