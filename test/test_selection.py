import numpy as np
from operations.selection import SelectIndividuals

def test_random_selection():
    individuals = np.array([
        [0, 1, 2, 3, 6, 9, 7, 8, 4, 5],
        [2, 0, 1, 9, 3, 6, 5, 7, 8, 4],
        [2, 3, 1, 8, 6, 5, 7, 0, 9, 4],
        [8, 2, 3, 1, 5, 7, 6, 4, 0, 9],
        [5, 7, 2, 3, 4, 8, 0, 9, 1, 6],
        [5, 2, 7, 3, 9, 4, 8, 1, 6, 0],
        [5, 7, 3, 1, 6, 2, 8, 0, 9, 4]
    ])

    fitness = np.array([7, 8, 4, 3, 1, 2, 1])

    selector = SelectIndividuals()

    print("Original individuals: \n", individuals)
    selected_individuals = selector.random(individuals)
    print("Selected individuals: \n", selected_individuals)

def test_tournament_selection():
    individuals = np.array([
        [0, 1, 2, 3, 6, 9, 7, 8, 4, 5],
        [2, 0, 1, 9, 3, 6, 5, 7, 8, 4],
        [2, 3, 1, 8, 6, 5, 7, 0, 9, 4],
        [8, 2, 3, 1, 5, 7, 6, 4, 0, 9],
        [5, 7, 2, 3, 4, 8, 0, 9, 1, 6],
        [5, 2, 7, 3, 9, 4, 8, 1, 6, 0],
        [5, 7, 3, 1, 6, 2, 8, 0, 9, 4]
    ])

    fitness = np.array([3, 1, 2, 1, 7, 8, 4])

    selector = SelectIndividuals()

    print("Original individuals: \n", individuals)
    print("Individual fitness: ", fitness)
    selected_individuals = selector.tournament(individuals, fitness)
    print("Selected individuals: \n", selected_individuals)

def test_roulette_selection():
    individuals = np.array([
        [0, 1, 2, 3, 6, 9, 7, 8, 4, 5],
        [2, 0, 1, 9, 3, 6, 5, 7, 8, 4],
        [2, 3, 1, 8, 6, 5, 7, 0, 9, 4],
        [8, 2, 3, 1, 5, 7, 6, 4, 0, 9],
        [5, 7, 2, 3, 4, 8, 0, 9, 1, 6],
        [5, 2, 7, 3, 9, 4, 8, 1, 6, 0],
        [5, 7, 3, 1, 6, 2, 8, 0, 9, 4]
    ])

    fitness = np.array([3, 1, 2, 1, 7, 8, 4])
    scaled_fit = np.array([(fi - fitness.min())/(fitness.max() - fitness.min()) for fi in fitness])
    selector = SelectIndividuals()

    print("Original individuals: \n", individuals)
    print("Individual fitness: ", fitness)
    print("Scaled fitness: ", scaled_fit)
    selected_individuals = selector.roulette_wheel(individuals, fitness)
    print("Selected individuals: \n", selected_individuals)

if __name__ == "__main__":
    #test_random_selection()
    #test_tournament_selection()
    test_roulette_selection()
