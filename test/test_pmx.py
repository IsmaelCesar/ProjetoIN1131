import numpy as np
import copy
from operations.crossover import _combine_child_parent_pmx

def test_pmx():
    print("here")

    #op = SingleTravelerX(probability=1.)
    parent1 = np.array([10,  5,  2,  4,  9, 7, 11,  3, 12, 1,  0, 13,  6, 12])
    parent2 = np.array([13, 10,  4, 2, 3, 11, 1, 7, 6, 8, 0, 9, 12, 5])
    cromossome_size = len(parent1)
    start = 2
    end  = 9

    print("Parent 1: ", parent1)
    print("Parent 2: ", parent2)

    child_1 = np.zeros(cromossome_size, dtype=int) -1
    child_2 = np.zeros(cromossome_size, dtype=int) -1

    child_1[start: end + 1] = copy.deepcopy(parent1[start: end + 1])
    child_1 = _combine_child_parent_pmx(child_1, parent1, parent2, start, end)
    
    child_2[start: end + 1] = copy.deepcopy(parent2[start: end + 1])
    child_2 = _combine_child_parent_pmx(child_2, parent2, parent1, start, end)

    print("Child 1: ", child_1)
    print("Child 2: ", child_2)

if __name__ == "__main__":
    test_pmx()
