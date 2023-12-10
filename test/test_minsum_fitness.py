import numpy as np
from operations.fitness import MinSumFitnessCalculator, _compute_distance_fitness
from scipy.spatial.distance import cdist

def test_minsum_fitness():
    """
    Compute distance fitness without including the origin
    """
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
    breaks = [3, 8, 11]

    distance_matrix = cdist(positions, positions)
    cmp_dist = 0.
    temp_idx = 0
    for b in breaks:
        cmp_dist += _compute_distance_fitness(individual[temp_idx:b], distance_matrix)
        temp_idx = b
    
    fitness_calculator = MinSumFitnessCalculator(distance_matrix)

    print("--"*50)
    print("Individual: ", individual)
    print("Positions: \n", positions)
    fitness = fitness_calculator.distance_fitness(individual, breaks)
    print("Fitness: ", fitness)
    cmp_result = "OK" if np.isclose(fitness, cmp_dist) else "NOT OK"
    print("CMP Fitness: ", cmp_result)
    print("--"*50)

if __name__ == "__main__":
    test_minsum_fitness()
