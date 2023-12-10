import numpy as np
from typing import List
from abc import ABC, abstractmethod

def _compute_distance_fitness(individual: np.ndarray, distance_matrix: np.ndarray) -> float:
    point_a_range = range(0, len(individual) - 1)
    point_b_range = range(1, len(individual))
    individual_fitness = 0.
    for point_a, point_b in zip(point_a_range, point_b_range):
        # extract an edge from a route
        individual_pos1 = individual[point_a]
        individual_pos2 = individual[point_b]
        
        # compute fitness
        individual_fitness += distance_matrix[individual_pos1][individual_pos2]

    return individual_fitness

def _extract_routes(individual: np.ndarray, traveler_breaks: List[int], origin: int = None) -> List[np.ndarray]:
    """
    Extract traveler routes based on traveler_breaks.
    If origin is not None then it will include the origin in the route extraction
    """
    routes = []
    temp_idx = 0
    include_origin = 2 if not origin else 0

    for t_break in traveler_breaks:
        #defining route
        rt = np.zeros(((t_break - temp_idx) + include_origin), dtype=int) -1
        rt[1:-1] = individual[temp_idx:t_break]

        if not origin: 
            rt[0] = origin
            rt[-1] = origin

        #including in route list
        routes += [rt]
        temp_idx = t_break
    return routes

class DistanceFitnessCalculator:

    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix

    def distance_fitness(self, individual: np.ndarray) -> float:
        """
        Computes the fitness of a single individual according to the distance matrix
        """
        return _compute_distance_fitness(individual, self.distance_matrix)

class MTSPFitnessCalculator(ABC):

    def __init__(self, distance_matrix: np.ndarray, origin: int):
        self.distance_matrix = distance_matrix
        self.origin = origin
    
    @abstractmethod
    def distance_fitness(self, individual: np.ndarray, traveler_breaks: List[int]): 
        raise Exception("Not implemented error")

class MinMaxFitnessCalculator(MTSPFitnessCalculator):

    def __init__(self, distance_matrix: np.ndarray):
        #self.distance_matrix = distance_matrix
        super().__init__(distance_matrix)

    def _compute_route_legnth(self, routes) -> List[int]:
        routes_length = []
        for r in routes:
            routes_length += [len(r)]

        return routes_length

    def distance_fitness(self, individual: np.ndarray, traveler_breaks: List[int]) -> float:
        
        routes = _extract_routes(individual, traveler_breaks)

        routes_length = self._compute_route_legnth(routes)
        max_route_idx = np.argmax(routes_length)
        return _compute_distance_fitness(routes[max_route_idx], self.distance_matrix)


class MinSumFitnessCalculator(MTSPFitnessCalculator): 

    def __init__(self, distance_matrix: np.ndarray):
        #self.distance_matrix = distance_matrix
        super().__init__(distance_matrix)
    
    def distance_fitness(self, individual: np.ndarray, traveler_breaks: List[int]) -> float:
        routes = _extract_routes(individual, traveler_breaks)

        min_sum = 0
        for rt in routes:
            min_sum += _compute_distance_fitness(rt, self.distance_matrix)

        return min_sum         
