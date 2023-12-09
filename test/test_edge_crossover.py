import numpy as np
from utils import check_repetition
from operations.crossover import _construct_child_edge_x

def test_edge_crossover():
    print("--"*50)
    print("Breaking PMX case 1")

    #op = SingleTravelerX(probability=1.)
    parent1 = np.array([8, 1, 9, 11, 7, 12, 13, 0, 6, 10, 2, 5, 4, 3])
    parent2 = np.array([0, 5, 1, 3, 12, 4,  6, 13, 7, 2,  9, 11, 8, 10])
    cromossome_size = len(parent1)

    print("Parent 1: ", parent1)
    print("Parent 2: ", parent2)

    child_1 = _construct_child_edge_x(parent1, parent2)
    child_2 = _construct_child_edge_x(parent2, parent1)
    print("Child 1: ", child_1)
    print("Child 2: ", child_2)
    print("Repetition Child 1: ", check_repetition(list(range(cromossome_size)), child_1))
    print("Repetition Child 2: ", check_repetition(list(range(cromossome_size)), child_2))
    print("--"*50)

if __name__ == "__main__":
    test_edge_crossover()
