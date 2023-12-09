import numpy as np
import copy
from operations.crossover import _combine_child_parent_pmx
from utils import check_repetition

def test_breaking_pmx_case1():
    print("--"*50)
    print("Breaking PMX case 1")

    #op = SingleTravelerX(probability=1.)
    parent1 = np.array([8, 1, 9, 11, 7, 12, 13, 0, 6, 10, 2, 5, 4, 3])
    parent2 = np.array([0, 5, 1, 3, 12, 4,  6, 13, 7, 2,  9, 11, 8, 10])
    cromossome_size = len(parent1)
    start = 4
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
    print("Repetition Child 1: ", check_repetition(list(range(cromossome_size)), child_1))
    print("Repetition Child 2: ", check_repetition(list(range(cromossome_size)), child_2))
    print("--"*50)

def test_breaking_pmx_case2():
    print("--"*50)
    print("Breaking PMX case 2")

    #op = SingleTravelerX(probability=1.)
    parent1 = np.array([6, 12, 2, 5, 8, 3, 7, 11, 13, 4, 10, 1, 9, 0])
    parent2 = np.array([7, 6, 3, 2, 4, 13, 0, 10, 9, 12, 11, 5, 8, 1])
    cromossome_size = len(parent1)
    start = 1
    end  = 8

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
    print("Repetition Child 1: ", check_repetition(list(range(cromossome_size)), child_1))
    print("Repetition Child 2: ", check_repetition(list(range(cromossome_size)), child_2))
    print("--"*50)

if __name__ == "__main__":
    test_breaking_pmx_case1()
    test_breaking_pmx_case2()
