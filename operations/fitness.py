import numpy as np

class DistanceFitnessCalculator:

    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix

    def distance_fitness(self, individual: np.ndarray) -> float:
        """
        Computes the fitness of a single individual according to the distance matrix
        """

        point_a_range = range(0, len(individual) - 1)
        point_b_range = range(1, len(individual))
        individual_fitness = 0.
        for point_a, point_b in zip(point_a_range, point_b_range):
            # extract an edge from a route
            individual_pos1 = individual[point_a]
            individual_pos2 = individual[point_b]
            
            # compute fitness
            individual_fitness += self.distance_matrix[individual_pos1][individual_pos2]

        return individual_fitness
