import numpy as np

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
