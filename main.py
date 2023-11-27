import numpy as np
from variation_operators import Mutation, Cross_over
from selection_operators import Parent_Selection, Elitism
from population import Get_Predefined_Data, Initialization, Fitness_Calculation
import statistics

def main():
    Cidades_Codigo, Cidades, X_Coord, Y_Coord = Get_Predefined_Data()

    # computing distances
    Dist = []
    for i in range(len(Cidades)):
        aux = []
        for k in range(len(Cidades)):
            aux.append(((X_Coord[i] - X_Coord[k])**2 +(Y_Coord[i] - Y_Coord[k])**2)**1/2)
        Dist.append(aux)


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
        print('Generation {}, Best_Overall_Solution {}'.format(major_it+1,Best_Overall[-1]))

        Plot_Average = [-Average_of_Generation[i] for i in range(len(Average_of_Generation))]
        Plot_Best = [-Best_of_Generation[i][-1] for i in range(len(Best_of_Generation))]
    

if __name__ == "__main__":
    main()
