import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time

#from Quantum_walk_libs_cuda import *
from Quantum_walk_libs import up, down, hadamard_coin
from Quantum_walk_libs import position, print_dists
from Quantum_walk_libs import generate_coin_array_random, multi_coin_flip, multi_walk, multi_circular_walk

def benchmark(r_values, N, steps, method):
    results = []
    for r in r_values:
        start_time = time.time()
        if method == 'loop':
            # Rodar a versão com loop explícito
            main(r=r, N=N, steps=steps, threshold=r+1)  # Forçar uso de multi_coin_flip_small
        else:
            # Rodar a versão vetorizada
            main(r=r, N=N, steps=steps, threshold=0)  # Forçar uso de multi_coin_flip
        elapsed_time = time.time() - start_time
        results.append((r, elapsed_time))
        with open(f"benchmark_{method}.txt", "a") as f:
            f.write(f"{method}: {r=}: {elapsed_time} seconds\n")
        print(f"Method: {method}, r={r}, Time: {elapsed_time}")
    return results


#####################################################################################################
#                                                                                                   #
#                                              MAIN                                                 #
#                                                                                                   #
#####################################################################################################

def main(r=1000, N=2001, steps=1000, threshold=2000, circular=False):
    """
    Execute a fully vectorized quantum walk simulation.

    Parameters
    ----------
    r : int
        The number of repetitions (walks) to simulate.
    N : int
        The number of positions in the quantum walk.
    steps : int
        The number of steps in each quantum walk.

    Returns
    -------
    dists : numpy array
        A r x 2 x N array representing the final probability distributions of all quantum walks.
    """
    start_time = time.time()
    # Inicializando o estado inicial (psi)
    psi = up
    print("--------------------------------")

    # Inicializando a matriz de distribuições para r caminhadas
    dists = cp.zeros((r, 2, N))
    
    # Posicionando o estado inicial psi para todas as distribuições em 'dists'
    dists[:, :, position(0, N)] = psi

    # Gerando uma matriz de r moedas quânticas com ângulos aleatórios para todas as repetições
    coin_array = generate_coin_array_random(r)
    
    # Executando a simulação de maneira completamente vetorizada
    for step in range(steps):
        # Aplicando a operação de flip da moeda para todas as distribuições simultaneamente
        if r<threshold:
            multi_coin_flip(coin_array, dists)

        # Executando o passo de caminhada vetorizado
        if circular:
            multi_circular_walk(dists)
        else:
            multi_walk(dists, N)
    print(f"--- %s seconds in Total ---" % (time.time() - start_time))
    print_dists(dists, N, r)
    return dists


if __name__ == "__main__":
    main(r=1000, N=2001, steps=1000, threshold=500, circular=False)