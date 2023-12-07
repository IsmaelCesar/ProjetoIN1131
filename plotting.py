
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_cities(coordenadas_cidades: np.ndarray, cidades_codigo: List[str] , num_cidades: str, filename: str = None):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    for i in range(num_cidades):
        ax.text(coordenadas_cidades[i, 0], coordenadas_cidades[i, 1], cidades_codigo[i], fontsize=6, fontweight='bold', color='black', ha='center', va='center')

    plt.scatter(coordenadas_cidades[:, 0], coordenadas_cidades[:, 1], s=20, c=None)
    plt.title("Travelling Salesman Problem", fontsize=14, fontweight='bold')
    plt.xlabel('X_Coord')
    plt.ylabel('Y_Coord')
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()
    plt.clf()


def Plot_Objectve_Function(Average, Best, Filename: str = None):
    plt.plot(Average, color='gray', label="Average")
    plt.plot(Best, color='blue', label="Best")
    plt.xlabel('Generations')
    plt.ylabel('Objective_Function')
    plt.legend(loc="best")
    if Filename is not None:
        plt.savefig(Filename)
    plt.show()
    plt.close()
    plt.clf()


def Plot_Cycle(Plot_X, Plot_Y, Cycle, Cycle_Codigos, Filename: str = None):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    for i in range(len(Cycle)):
        ax.text(Plot_X[i], Plot_Y[i], Cycle_Codigos[i], fontsize=6, fontweight='bold', color='black', ha='center', va='center')
    plt.scatter(Plot_X, Plot_Y, s=20, c=None)
    plt.plot(Plot_X, Plot_Y)
    plt.title("Travelling Salesman Problem Optimized", fontsize=14, fontweight='bold')
    plt.xlabel('X_Coord')
    plt.ylabel('Y_Coord')
    if Filename is not None:
        plt.savefig(Filename)
    plt.show()
    plt.close()
    plt.clf()
