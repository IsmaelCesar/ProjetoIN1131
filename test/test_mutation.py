import numpy as np
from operations.mutation import SingleTravelerMut

def test_swap(): 
    individual = np.array([5, 6, 4, 0, 3, 1, 2, 7, 10, 8, 9])
    mutation = SingleTravelerMut(probability=1.)

    print("Original Individual: ", individual)
    m_individual = mutation.swap(individual)
    print("Mutated Individual: ", m_individual)

def test_scramble():
    individual = np.array([5, 6, 4, 0, 3, 1, 2, 7, 10, 8, 9])
    mutation = SingleTravelerMut(probability=1.)

    print("Original Individual: ", individual)
    m_individual = mutation.scramble(individual)
    print("Mutated Individual: ", m_individual)

def test_inverse():
    individual = np.array([5, 6, 4, 0, 3, 1, 2, 7, 10, 8, 9])
    mutation = SingleTravelerMut(probability=1.)

    print("Original Individual: ", individual)
    m_individual = mutation.scramble(individual)
    print("Mutated Individual: ", m_individual)

def test_insert():
    individual = np.array([5, 6, 4, 0, 3, 1, 2, 7, 10, 8, 9])
    mutation = SingleTravelerMut(probability=1.)

    print("Original Individual: ", individual)
    m_individual = mutation.insert(individual)
    print("Mutated Individual: ", m_individual)

if __name__ == "__main__":
    #test_swap()
    #test_scramble()
    #test_inverse()
    test_insert()