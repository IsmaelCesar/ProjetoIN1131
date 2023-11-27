
import numpy as np

def Get_Predefined_Data():
    Cidades_Codigo = ['Abreu e Lima','Araçoiaba','Cabo de Santo Agostinho','Camaragibe',
                  'Igarassu','Ilha de Itamaracá','Ipojuca','Itapissuma','Jaboatão dos Guararapes',
                  'Moreno','Olinda','Paulista','Recife','São Lourenço da Mata']
    Cidades = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

    X_Coord = [951488.02,931841.18,937345.52,943003.09,952093.07,
            960467.82,933679.41,953048.35,938660.18,929576.04,
            957514.70,955859.31,954449,938503.56]

    Y_Coord = [9124112.67,9137218.04,9081735.21,9111087.24,9131426.67,
            9140872.45,9069453.99,9138239.85,9100920.74,9100080.03,
            9111913.35,9119995.54,9108698.82,9113340.75]
    
    return Cidades_Codigo, Cidades, X_Coord, Y_Coord


#Gero aleatoriamente uma população inicial#
def Initialization(Cidades, pop = 100):
  population = np.empty((0,len(Cidades)))
  for t in range(pop):
    solution = Cidades.copy()
    np.random.shuffle(solution)
    population = np.vstack((population,solution))

  return population

#Verifico a função de aptidão de cada uma das soluções nesta população#
def Fitness_Calculation(Population, Cidades, Dist):
  fitness = np.empty((0,len(Cidades)+1))
  for j in range(len(Population)):
    solution = []
    for i in range(len(Population[j])):
      if i+1 in range(len(Population[j])):
        solution.append(Dist[int(Population[j][i])][int(Population[j][i+1])])
      else:
        solution.append(Dist[int(Population[j][i])][int(Population[j][0])])
    solution = -sum(solution) ################# Por ser um problema de minimização, deixo os valores de aptidão negativos para que o menor seja o melhor #################
    aux = [int(i) for i in Population[j]]
    aux.append(solution)
    aux = np.array(aux)
    fitness = np.vstack((fitness,aux))

  return fitness
