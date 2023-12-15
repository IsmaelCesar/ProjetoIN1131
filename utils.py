import os
import json
import numpy as np



def check_create_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def compute_cycle(best_overall, cidades_coordenadas):
    """
    Gets the coorinates and the city codes from the bets overall solution
    found by the genetic algorithm
    """
    # computing cycle
    computed_cycle = np.zeros((len(best_overall) + 1, ), dtype=int) -1 
    computed_cycle[:-1] = best_overall
    computed_cycle[-1] = best_overall[0]
    
    #getting coordinates
    coordenadas_ciclo = cidades_coordenadas[computed_cycle]

    return computed_cycle, coordenadas_ciclo


def compute_traveler_breaks(n_travelers: int, n_var: int):
    traveler_breaks = []

    offset = 0
    for t_idx in range(n_travelers - 1):
        offset += n_var // n_travelers
        traveler_breaks += [offset]
    
    traveler_breaks += [n_var - 1]
    return traveler_breaks


def check_repetition(cidades_range: np.ndarray, individual: np.ndarray ) -> bool:
    city_dict = {}

    # initialize zero
    for city_key in cidades_range:
        city_dict[city_key] = 0
    
    # count 
    for city_indiv in individual: 
        city_dict[city_indiv] += 1
    
    reps_arr = np.array(list(city_dict.values()))
    return len(reps_arr[reps_arr >= 2]) > 0

def save_statistics_as_json(statistics: dict, filename: str):

    with open(filename, "w+") as file: 
        json.dump(statistics, file)

def load_statistics_from_json(filename: str) -> dict:

    with open(filename, "r") as file:
        statistics = json.load(file)

    return statistics
