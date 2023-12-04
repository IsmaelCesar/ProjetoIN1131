import os
import re
from datetime import datetime
import numpy as np
from variation_operators import Mutation, Cross_over
from selection_operators import Parent_Selection, Elitism
from population import Get_Predefined_Data, Initialization, Fitness_Calculation
import statistics
import logging
import matplotlib.pyplot as plt

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

    Cidades_Codigo, Cidades, X_Coord, Y_Coord = Get_Predefined_Data()

    # computing distances
    Dist = []
    for i in range(len(Cidades)):
        aux = []
        for k in range(len(Cidades)):
            aux.append(((X_Coord[i] - X_Coord[k])**2 +(Y_Coord[i] - Y_Coord[k])**2)**1/2)
        Dist.append(aux)

    
    X = X_Coord
    Y = Y_Coord

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    for i in range(len(Cidades)):
        ax.text(X_Coord[i], Y_Coord[i], Cidades_Codigo[i], fontsize=6, fontweight='bold', color='black', ha='center', va='center')

    plt.scatter(X, Y, s=20, c=None)
    plt.title("Travelling Salesman Problem", fontsize=14, fontweight='bold')
    plt.xlabel('X_Coord')
    plt.ylabel('Y_Coord')
    plt.show()

    """**RESOLUÇÃO DO PROBLEMA EM UMA ÚNICA RODADA**"""

    #############################################################################################################################################################################
    ####################################################### Nesta célula eu de fato executo meu algoritmo evolucionário ########################################################
    #############################################################################################################################################################################
    Population = Initialization(Cidades, pop=200)
    Fitness = Fitness_Calculation(Population, Cidades, Dist)
    Sorted_Population = np.array(sorted(Fitness, reverse=True, key=lambda x:x[-1]))

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

    plt.plot(Plot_Average, color='gray')
    plt.plot(Plot_Best, color='blue')
    plt.xlabel('Generations')
    plt.ylabel('Objective_Function')
    plt.show()

    Plot_Cycle = [Best_Overall[i] for i in range(len(Best_Overall) -1)]
    Plot_X, Plot_Y, Plot_Codigos = [], [], []
    for i in Plot_Cycle:
        Plot_X.append(X_Coord[int(i)])
        Plot_Y.append(Y_Coord[int(i)])
        Plot_Codigos.append(Cidades_Codigo[int(i)])

    Plot_X.append(Plot_X[0])
    Plot_Y.append(Plot_Y[0])
    Plot_Codigos.append(Plot_Codigos[0])
    Plot_Cycle.append(Plot_Cycle[0])

    fig = plt.figure(figsize=(8,20))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    for i in range(len(Plot_Cycle)):
        ax.text(Plot_X[i], Plot_Y[i], Plot_Codigos[i], fontsize=6, fontweight='bold', color='black', ha='center', va='center')

    plt.scatter(Plot_X, Plot_Y, s=20, c=None)
    plt.plot(Plot_X, Plot_Y)
    plt.title("Travelling Salesman Problem Optimized", fontsize=14, fontweight='bold')
    plt.xlabel('X_Coord')
    plt.ylabel('Y_Coord')
    plt.show()


if __name__ == "__main__":
    main()
