
import numpy as np
from typing import Tuple, List

def get_predefined_data() -> Tuple[List[str], List[int], np.ndarray]:
    cidades_codigo = ['Abreu e Lima','Araçoiaba','Cabo de Santo Agostinho','Camaragibe',
                  'Igarassu','Ilha de Itamaracá','Ipojuca','Itapissuma','Jaboatão dos Guararapes',
                  'Moreno','Olinda','Paulista','Recife','São Lourenço da Mata']
    cidades = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

    x_coord = [951488.02,931841.18,937345.52,943003.09,952093.07,
            960467.82,933679.41,953048.35,938660.18,929576.04,
            957514.70,955859.31,954449,938503.56]

    y_coord = [9124112.67,9137218.04,9081735.21,9111087.24,9131426.67,
               9140872.45,9069453.99,9138239.85,9100920.74,9100080.03,
               9111913.35,9119995.54,9108698.82,9113340.75]
    
    coordenadas_cidades = np.vstack([x_coord, y_coord]).T
    
    return cidades_codigo, cidades, coordenadas_cidades


#Gero aleatoriamente uma população inicial#
def Initialization(cidades, pop = 100):
  population = np.empty((0,len(cidades)))
  for t in range(pop):
    solution = cidades.copy()
    np.random.shuffle(solution)
    population = np.vstack((population,solution))

  return population

#Verifico a função de aptidão de cada uma das soluções nesta população#
def fitness_calculation(population, cidades, Dist):
  fitness = np.empty((0,len(cidades)+1))
  for j in range(len(population)):
    solution = []
    for i in range(len(population[j])):
      if i+1 in range(len(population[j])):
        solution.append(Dist[int(population[j][i])][int(population[j][i+1])])
      else:
        solution.append(Dist[int(population[j][i])][int(population[j][0])])
    solution = -sum(solution) ################# Por ser um problema de minimização, deixo os valores de aptidão negativos para que o menor seja o melhor #################
    aux = [int(i) for i in population[j]]
    aux.append(solution)
    aux = np.array(aux)
    fitness = np.vstack((fitness,aux))

  return fitness
