import numpy as np
from population import get_predefined_data
from plotting import plot_cities
from operations.crossover import SingleTravelerX

def test_order1(): 
    parent1 = np.array([5, 6, 4, 0, 3, 1, 2])
    parent2 = np.array([1, 4, 6, 2, 5, 3, 0])

    print("Parent 1: ", parent1)
    print("Parent 2: ", parent2)
    crossover = SingleTravelerX(probability=1.)
    child_1, child_2 = crossover.order_1(parent1, parent2)
    print("First child: ", child_1)
    print("Second child: ", child_2)


def test_pmx():
    parent1 = np.array([5, 6, 4, 0, 3, 1, 2])
    parent2 = np.array([1, 4, 6, 2, 5, 3, 0])

    print("Parent 1: ", parent1)
    print("Parent 2: ", parent2)
    crossover = SingleTravelerX(probability=1.)
    child_1, child_2 = crossover.pmx(parent1, parent2)
    print("First child: ", child_1)
    print("Second child: ", child_2)

if __name__ == "__main__": 
    test_pmx()