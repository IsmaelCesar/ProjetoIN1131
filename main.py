import os
import re
from datetime import datetime
import numpy as np
from variation_operators import Mutation, Cross_over
from selection_operators import Parent_Selection, Elitism
from population import get_predefined_data, Initialization, fitness_calculation
import statistics
import logging
from plotting import plot_cities, Plot_Objectve_Function, Plot_Cycle
from utils import Compute_Cycle
from scipy.spatial.distance import cdist

logger = logging.getLogger("tsp_ga")

def check_create_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def main():
    # cria um diretório com o datetime da execucao atual
    datetime_digits = re.findall(r"\d+", str(datetime.now()))
    datetime_string = "".join(datetime_digits)
    execution_dir = "results/{}".format(datetime_string)
    check_create_dir(execution_dir)

    # configurando logging >>>
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("{}/{}".format(execution_dir, "execution.log"), mode="w+") # salva logs em arquivo
    stream_handler = logging.StreamHandler() # printa logs no prompt
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<

    cidades_codigo, cidades, coordenadas_cidades = get_predefined_data()

    # creating distance matrix
    Dist = cdist(coordenadas_cidades, coordenadas_cidades, metric="euclidean")
    #for i in range(len(Cidades)):
    #    aux = []
    #    for k in range(len(Cidades)):
    #        aux.append(((X_Coord[i] - X_Coord[k])**2 +(Y_Coord[i] - Y_Coord[k])**2)**1/2)
    #    Dist.append(aux)

    cities_fname = os.path.join(execution_dir, "cities.png")
    plot_cities(coordenadas_cidades, cidades_codigo, len(cidades), Filename=cities_fname)

    """**RESOLUÇÃO DO PROBLEMA EM UMA ÚNICA RODADA**"""

    #############################################################################################################################################################################
    ####################################################### Nesta célula eu de fato executo meu algoritmo evolucionário ########################################################
    #############################################################################################################################################################################
    population = Initialization(cidades, pop=200)
    fitness = fitness_calculation(population, cidades, Dist)
    Sorted_Population = np.array(sorted(fitness, reverse=True, key=lambda x:x[-1]))

    Best_of_Generation, Average_of_Generation = [], []
    Best_Overall = Fitness[np.random.randint(0,len(Fitness)-1)]

    for major_it in range(int(len(Population))):
        New_Population = np.empty((0,len(Cidades)))
        for minor_it in range(int(len(Population)/2)):
            Parent_1, Parent_2 = Parent_Selection(Fitness, Population)
            Child_1, Child_2 = Cross_over(Parent_1, Parent_2)
            Mutated_Child_1, Mutated_Child_2 = Mutation(Child_1, Child_2)
            Mutated_Child_1 = np.array(Mutated_Child_1)
            Mutated_Child_2 = np.array(Mutated_Child_2)
            New_Population = np.vstack((New_Population,Mutated_Child_1))
            New_Population = np.vstack((New_Population,Mutated_Child_2))
        New_Fitness = Fitness_Calculation(New_Population, Cidades, Dist)
        Sorted_New_Population = np.array(sorted(New_Fitness, reverse=True, key=lambda x:x[-1]))
        Sorted_New_Population_Elitism = Elitism(Sorted_Population, Sorted_New_Population)

        Average_of_Generation.append(statistics.mean([Sorted_New_Population[i][-1] for i in range(len(Sorted_New_Population))]))
        Best_of_Generation.append(Sorted_New_Population_Elitism[0])
        if Best_of_Generation[major_it][-1] > Best_Overall[-1]:
            Best_Overall = Best_of_Generation[major_it].copy()

        Fitness = Sorted_New_Population_Elitism.copy()
        Sorted_Population = np.array(sorted(Fitness, reverse=True, key=lambda x:x[-1]))
        logger.info('Generation {}, Best_Overall_Solution {}'.format(major_it+1,Best_Overall[-1]))

        #Plot_Average = [-Average_of_Generation[i] for i in range(len(Average_of_Generation))]
        #Plot_Best = [-Best_of_Generation[i][-1] for i in range(len(Best_of_Generation))]
    
    Plot_Average = [-Average_of_Generation[i] for i in range(len(Average_of_Generation))]
    Plot_Best = [-Best_of_Generation[i][-1] for i in range(len(Best_of_Generation))]

    objective_fn_fname = os.path.join(execution_dir, "objective_fn.png")
    Plot_Objectve_Function(Plot_Average, Plot_Best, Filename=objective_fn_fname)

    Computed_Cycle, Cycle_X, Cycle_Y, Cycle_Codigos = Compute_Cycle(Best_Overall, 
                                                                    X_Coord,
                                                                    Y_Coord,
                                                                    Cidades_Codigo)

    cycle_plot_fname = os.path.join(execution_dir, "cycle.png")
    Plot_Cycle(Cycle_X, Cycle_Y, Computed_Cycle, Cycle_Codigos, Filename=cycle_plot_fname)

if __name__ == "__main__":
    main()
