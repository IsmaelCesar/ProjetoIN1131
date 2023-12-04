

def Compute_Cycle(Best_Overall, X_Coord, Y_Coord, Cidades_Codigo):
    """
    Gets the coorinates and the city codes from the bets overall solution
    found by the genetic algorithm
    """
    Computed_Cycle = [Best_Overall[i] for i in range(len(Best_Overall) -1)]
    Cycle_X, Cycle_Y, Cycle_Codigos = [], [], []
    for i in Computed_Cycle:
        Cycle_X.append(X_Coord[int(i)])
        Cycle_Y.append(Y_Coord[int(i)])
        Cycle_Codigos.append(Cidades_Codigo[int(i)])

    return Computed_Cycle, Cycle_X, Cycle_Y, Cycle_Codigos
