from numpy import random

#Como operador de recombinação, faço um cruzamento de ordem entre os dois pais; incluo uma probabilidade padrão de 85%#
def Cross_over(Parent_1, Parent_2, prob = 0.85):
  probability = random.random()
  #Faço primeiro a determinação dos genes de referência para o cruzamento#
  if probability < prob:
    Child_1 = [None] * len(Parent_1)
    gene_1 = random.randint(0,len(Parent_1))
    gene_2 = random.randint(0,len(Parent_1))
    while gene_2 == gene_1:
      gene_2 = random.randint(0,len(Parent_1))
    if gene_2 < gene_1:
      aux = gene_2
      gene_2 = gene_1
      gene_1 = aux
  #Faço todo o procedimento para a criança 1#
    for i in range(gene_1, gene_2):
      Child_1[i] = Parent_1[i]
    if gene_2 < len(Child_1):
      temp_gene_2 = gene_2 #Esse gene temporário existe somente para evitar que haja uma sobreposição de alelos já selecionados#
      for i in range(gene_2,len(Child_1)):
        for k in range(gene_2,len(Parent_2)):
          aux = Parent_2[k]
          if aux not in Child_1 and Child_1[i] == None:
            Child_1[i] = aux
            temp_gene_2 += 1
            break
      for i in range(temp_gene_2,len(Child_1)):
        for k in range(gene_2):
          aux = Parent_2[k]
          if aux not in Child_1:
            Child_1[i] = aux
            break
    if None in Child_1:
      for i in range(gene_1):
        for k in range(len(Parent_2)):
          aux = Parent_2[k]
          if aux not in Child_1:
            Child_1[i] = aux
            break

  #Em seguida, faço o mesmo para a criança 2, só não preciso mexer nos genes novamente#
    Child_2 = [None] * len(Parent_2)
    for i in range(gene_1, gene_2):
      Child_2[i] = Parent_2[i]
    if gene_2 < len(Child_2):
      temp_gene_2 = gene_2 #Novamente, esse gene temporário existe somente para evitar que haja uma sobreposição de alelos já selecionados#
      for i in range(gene_2,len(Child_2)):
        for k in range(gene_2,len(Parent_1)):
          aux = Parent_1[k]
          if aux not in Child_2 and Child_2[i] == None:
            Child_2[i] = aux
            temp_gene_2 += 1
            break
      for i in range(temp_gene_2,len(Child_2)):
        for k in range(gene_2):
          aux = Parent_1[k]
          if aux not in Child_2:
            Child_2[i] = aux
            break
    if None in Child_2:
      for i in range(gene_1):
        for k in range(len(Parent_1)):
          aux = Parent_1[k]
          if aux not in Child_2:
            Child_2[i] = aux
            break
  else:
    Child_1 = Parent_1.copy()
    Child_2 = Parent_2.copy()

  return Child_1, Child_2

#Como operador de mutação, faço um simples procedimento de troca (swap) entre dois genes; incluo uma probabilidade padrão de 10%#
def Mutation(Child_1, Child_2, prob = 0.1):
  #Faço todo o procedimento para a criança mutada 1#
  probability = random.random()
  if probability < prob:
    mutated_child_1 = Child_1.copy()
    mutated_child_2 = Child_2.copy()
    gene_1 = random.randint(0,len(Child_1)-1)
    gene_2 = random.randint(0,len(Child_1)-1)
    while gene_1 == gene_2:
      gene_2 = random.randint(0,len(Child_1)-1)
    aux1 = mutated_child_1[gene_2]
    aux2 = mutated_child_2[gene_2]
    mutated_child_1[gene_2] = mutated_child_1[gene_1]
    mutated_child_1[gene_1] = aux1
    mutated_child_2[gene_2] = mutated_child_2[gene_1]
    mutated_child_2[gene_1] = aux2
  else:
    mutated_child_1 = Child_1.copy()
    mutated_child_2 = Child_2.copy()

  return mutated_child_1, mutated_child_2

