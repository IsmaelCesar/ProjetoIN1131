
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


def plot_objective_function(average, best, filename: str = None):
    plt.plot(average, color='gray', label="Average")
    plt.plot(best, color='blue', label="Best")
    plt.xlabel('Generations')
    plt.ylabel('Objective_Function')
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()
    plt.clf()


def plot_cycle(coordenadas_cidades, ciclo, codigos_ciclo, filename: str = None):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    for ciclo_value in ciclo:
        ax.text(
            coordenadas_cidades[ciclo_value, 0], 
            coordenadas_cidades[ciclo_value, 1], 
            codigos_ciclo[ciclo_value], 
            fontsize=6, 
            fontweight='bold', 
            color='black', 
            ha='center', 
            va='center')
    ax.scatter(coordenadas_cidades[:, 0], coordenadas_cidades[:, 1], s=20, c=None)
    ax.plot(coordenadas_cidades[ciclo][:, 0], coordenadas_cidades[ciclo][:, 1])
    ax.set_title("Travelling Salesman Problem Optimized", fontsize=14, fontweight='bold')
    ax.set_xlabel('X_Coord')
    ax.set_ylabel('Y_Coord')
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()
    plt.clf()


def _get_cycle_codigos(ciclo):

def plot_mtsp_cycles(coordenadas_cidades, rotas, codigos_ciclo, filename: str = None):

