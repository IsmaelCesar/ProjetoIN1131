
import numpy as np
import copy 
from typing import Tuple

def _combine_second_parent_segment(child: np.ndarray, parent: np.ndarray, end: int):
    """
    auxiliary method of the _apply_order_1_x for combining the second parent segment
    """
    
    parent_idx = child_idx = end+1

    if parent_idx > len(parent) - 1:
        parent_idx = 0

    if child_idx > len(child) - 1 : 
        child_idx = 0

    for _ in range(len(child)):

        if parent[parent_idx] not in child:
            child[child_idx] = parent[parent_idx]
            child_idx += 1

            if child_idx > len(child) - 1:
                child_idx = 0

        parent_idx += 1

        if parent_idx > len(parent) - 1:
            parent_idx = 0

    return child

def _apply_order_1_x(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Effectively applies order 1 crossover on the cromossomes of the parents
    """
    cromossome_size = len(parent1)
    start = np.random.randint(0, cromossome_size-2)
    end = np.random.randint(start+1, cromossome_size-1)

    child_1 = np.zeros(cromossome_size, dtype=int) -1
    child_2 = np.zeros(cromossome_size, dtype=int) -1
    
    # first child
    child_1[start: end+1] = copy.deepcopy(parent1[start:end+1])
    child_1 = _combine_second_parent_segment(child_1, parent2, end)

    #second child:
    child_2[start: end+1] = copy.deepcopy(parent2[start:end+1])
    child_2 = _combine_second_parent_segment(child_2, parent1, end)

    return child_1, child_2

def _combine_child_parent_pmx(child: np.ndarray, parent1: np.ndarray, parent2: np.ndarray, start: int , end: int) -> np.ndarray:
    """
    Auxiliary procedure for the _apply_pmx method
    """

    # dealing with the crossover segment
    for seg_idx in range(start, end + 1): 
        if parent2[seg_idx] not in child:
            #taking position of elements from parent1 in parent2
            p1_p2_idx = np.where(parent2 == parent1[seg_idx])[0][0]
            if child[p1_p2_idx] == -1: #vazio
                child[p1_p2_idx] = parent2[p1_p2_idx]
            else:
                #taking other position of elements from parent1 in parent2
                p1_p2_idx = np.where(parent2 == parent1[p1_p2_idx])[0][0]
                child[p1_p2_idx] = parent2[seg_idx]

    #copying the rest of parent 2
    for c_idx, (c_element, p2_element) in enumerate(zip(child, parent2)):
        if c_element == -1: 
            child[c_idx] = p2_element

    return child

def _apply_pmx(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Effectively apply pmx crossover on the cromossome of the parents
    """
    cromossome_size = len(parent1)
    start = np.random.randint(0, cromossome_size - 1)
    end = np.random.randint(start, cromossome_size - 1)

    child_1 = np.zeros(cromossome_size, dtype=int) -1
    child_2 = np.zeros(cromossome_size, dtype=int) -1

    child_1[start: end + 1] = copy.deepcopy(parent1[start: end + 1])
    child_1 = _combine_child_parent_pmx(child_1, parent1, parent2, start, end)

    child_2[start: end + 1] = copy.deepcopy(parent2[start: end + 1])
    child_2 = _combine_child_parent_pmx(child_2, parent2, parent1, start, end)

    return child_1, child_2

class SingleTravelerX:

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def order_1(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rand_prob = np.random.rand()
        if rand_prob <= self.probability:
          return _apply_order_1_x(parent1, parent2)

        return copy.deepcopy(parent1), np.deepcopy(parent2)

    def pmx(self, parent1: np.ndarray, parent2: np.array) -> Tuple[np.ndarray, np.ndarray]: 
        rand_prob = np.random.rand()
        if rand_prob <= self.probability:
            return _apply_pmx(parent1, parent2)
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
