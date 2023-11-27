
#seleção de pais
#Faço um torneio para determinar a escolha dos pais, em que três candidatos são escolhidos aleatoriamente e somente o melhor será vencedor#
import numpy as np
from numpy import random

# TODO: Improve code, and remove repeated lines
def Parent_Selection(Fitness, Population):
  #Faço todo o procedimento para o pai 1#
  candidate_1 = random.randint(0,len(Population)-1)
  candidate_2 = random.randint(0,len(Population)-1)
  candidate_3 = random.randint(0,len(Population)-1)
  while candidate_1 == candidate_2:
    candidate_1 = random.randint(0,len(Population)-1)
  while candidate_2 == candidate_3:
    candidate_3 = random.randint(0,len(Population)-1)
  while candidate_1 == candidate_3:
    candidate_3 = random.randint(0,len(Population)-1)
  fitness_1 = Fitness[candidate_1][-1]
  fitness_2 = Fitness[candidate_2][-1]
  fitness_3 = Fitness[candidate_3][-1]
  parent_1 = Population[candidate_1]
  if fitness_2 >= fitness_1 and fitness_2 >= fitness_3:
    parent_1 = Population[candidate_2]
  if fitness_3 >= fitness_1 and fitness_3 >= fitness_2:
    parent_1 = Population[candidate_3]
  #Em seguida, faço o mesmo para o pai 2#
  candidate_1 = random.randint(0,len(Population)-1)
  candidate_2 = random.randint(0,len(Population)-1)
  candidate_3 = random.randint(0,len(Population)-1)
  while candidate_1 == candidate_2:
    candidate_1 = random.randint(0,len(Population)-1)
  while candidate_2 == candidate_3:
    candidate_3 = random.randint(0,len(Population)-1)
  while candidate_1 == candidate_3:
    candidate_3 = random.randint(0,len(Population)-1)
  fitness_1 = Fitness[candidate_1][-1]
  fitness_2 = Fitness[candidate_2][-1]
  fitness_3 = Fitness[candidate_3][-1]
  parent_2 = Population[candidate_1]
  if fitness_2 >= fitness_1 and fitness_2 >= fitness_3:
    parent_2 = Population[candidate_2]
  if fitness_3 >= fitness_1 and fitness_3 >= fitness_2:
    parent_2 = Population[candidate_3]

  return parent_1, parent_2

# Seleção de sobreviventes
#Crio ainda uma função elitista, para garantir que sempre a melhor solução da geração anterior permaneça na nova geração#
#Esta solução de elite substituirá a pior solução da nova geração, mas fora isso, toda a nova geração será composta por novos indivíduos#
def Elitism(old_sorted_population, new_sorted_population):
  Elite = old_sorted_population[0]
  new_population_elitism = new_sorted_population.copy()
  new_population_elitism[-1] = Elite
  new_population_elitism = np.array(sorted(new_population_elitism, reverse=True, key=lambda x:x[-1]))

  return new_population_elitism
