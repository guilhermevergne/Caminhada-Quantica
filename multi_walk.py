import numpy as np
import matplotlib.pyplot as plt
import time



up = np.array((1,0))
down = np.array((0,1))
hadamar_coin = np.array([[1,1],[1,-1]])*(0.5**0.5)

import numpy as np

def generate_single_coin(theta):
    """
    Generate a 2x2 rotation matrix representing a single quantum coin operation.

    Parameters
    ----------
    theta : float
        The angle of rotation in radians.

    Returns
    -------
    numpy array
        A 2x2 matrix representing the quantum coin operation.
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def position(i,n):
    return int((n+1)/2+i)-1


def print_dist(dist, N):
    """
    Plot the probability distribution of a quantum walk.

    Parameters
    ----------
    dist : numpy array
        A 2xN array representing the probability distribution of the quantum walk.
    N : int
        The number of positions in the quantum walk.
    """
    limit = N//2
    prob = np.abs(dist[0])**2+np.abs(dist[1])**2
    prob *= 100
    plt.plot(np.arange(-limit,limit+1), prob)
    plt.ylim(0,prob.max()*1.1)
    plt.xlim(-limit-5,limit+5)
    plt.savefig("walk.png")
    plt.show()


def coin_flip(coin,dist):
    """
    Apply a quantum coin vectorized operation to a quantum walk.

    Parameters
    ----------
    coin : numpy array
        A 2x2 matrix representing the quantum coin operation.
    dist : numpy array
        A 2xN array representing the probability distribution of the quantum walk.
    """
    
    dist[:,:] = np.matmul(coin,dist)

def old_coin_flip(dist, coin, N):
    for i in range(N):
        dist[:,i] = np.matmul(coin,dist[:,i])

def circular_walk(dist, N):
    """
    Apply a vectorized quantum walk step to a quantum circular walk.

    Parameters
    ----------
    dist : numpy array
        A 2xN array representing the probability distribution of the quantum walk.
    N : int
        The number of positions in the quantum walk.

    """
    dist[0,1:] = dist[0,:N-1]
    dist[0,0] = dist[0,N-1]
    dist[1,:N-1] = dist[1,1:]
    dist[1,N-1] = dist[1,0]

def walk(dist, N):
    """
    Apply a vectorized quantum walk step to a quantum walk.

    Parameters
    ----------
    dist : numpy array
        A 2xN array representing the probability distribution of the quantum walk.
    N : int
        The number of positions in the quantum walk.

    """
    dist[0,1:] = dist[0,:N-1]
    dist[1,:N-1] = dist[1,1:]

def old_walk(dist, N):
    for i in range(1,N):
        dist[0,N-i],dist[0,N-i-1] = dist[0,N-i+1],dist[0,N-i]
    dist[0,N-1] = dist[0,0]
    for i in range(N-1):
        dist[1,i+1],dist[1,i+2] = dist[1,i],dist[1,i+1]
    dist[1,0] = dist[1,N-1]


#####################################################################################################
#                                                                                                   #
#                                             Multi-walk                                            #
#                                                                                                   #
#####################################################################################################

def generate_coin_array_random(n):
    """
    Generate an array containing n quantum coins, each represented by a 2x2 rotation matrix.
    The angles for the coins are randomly generated between 30 and 45 degrees.

    Parameters
    ----------
    n : int
        The number of coins to generate.

    Returns
    -------
    numpy array
        A numpy array containing n 2x2 matrices representing quantum coin operations.
    """
    # Gerando n valores de theta aleatórios entre 30 e 45 graus (convertendo para radianos)
    thetas = np.random.uniform(np.deg2rad(30), np.deg2rad(45), n)
    
    # Gerando as matrizes das moedas quânticas
    coin_array = np.array([generate_single_coin(theta) for theta in thetas])
    
    return coin_array


def print_dists(dists, N, r):
    """
    Plot the probability distribution of a quantum multi-walk.

    Parameters
    ----------
    dist : numpy array 
        A r dist instances, which are each a 2xN array representing the probability distribution of the quantum walk.
    N : int
        The number of positions in the quantum walk.
    r : int
        The number of repetitions (walks) to simulate.
    """
    limit = N//2
    start_graph_time = time.time()
    for dist in dists:
        prob = np.abs(dist[0])**2+np.abs(dist[1])**2
        prob *= 100/r
    plt.plot(np.arange(-limit,limit+1), prob)
    plt.ylim(0,prob.max()*1.1)
    plt.xlim(-limit-5,limit+5)
    plt.savefig("multi_walk.png")
    print("--- %s seconds to plot ---" % (time.time() - start_graph_time))
    plt.show()


def multi_coin_flip(coin_array, dists):
    """
    Apply a vectorized quantum coin flip operation to all quantum walks.

    Parameters
    ----------
    coin_array : numpy array
        A r x 2 x 2 array containing the coin operation matrices for each walk.
    dists : numpy array
        A r x 2 x N array representing the quantum walk distributions.
    """
    # Aplicar a moeda a todas as distribuições em uma única operação
    dists[:, :, :] = np.einsum('rij,rjk->rik', coin_array, dists)

def multi_coin_flip_small(coin_array, dists):
    """
    Apply a quantum coin flip operation to all quantum walks.

    Parameters
    ----------
    coin_array : numpy array
        A r x 2 x 2 array containing the coin operation matrices for each walk.
    dists : numpy array
        A r x 2 x N array representing the quantum walk distributions.
    """
    # Aplicar a moeda quântica individualmente em cada repetição de forma mais clara
    for repetition in range(coin_array.shape[0]):
        dists[repetition] = np.dot(coin_array[repetition], dists[repetition])



def multi_walk(dists):
    """
    Apply a vectorized quantum walk step to all distributions.

    Parameters
    ----------
    dists : numpy array
        A r x 2 x N array representing the quantum walk distributions.
    N : int
        The number of positions in the quantum walk.
    """
    # Caminhar para a esquerda (qubit superior) e direita (qubit inferior)
    dists[:, 0, 1:] = dists[:, 0, :-1]  # Caminhante qubit 'up' move para a esquerda
    dists[:, 1, :-1] = dists[:, 1, 1:]  # Caminhante qubit 'down' move para a direita


def multi_circular_walk(dists):
    """
    Apply a vectorized quantum walk step with circular boundary conditions.

    Parameters
    ----------
    dists : numpy array
        A r x 2 x N array representing the quantum walk distributions.
    N : int
        The number of positions in the quantum walk.
    """
    # Caminhar circularmente para a esquerda e para a direita
    dists[:, 0, 1:] = dists[:, 0, :-1]  # Caminhante qubit 'up' move para a esquerda
    dists[:, 0, 0] = dists[:, 0, -1]    # O último vai para a primeira posição (circular)

    dists[:, 1, :-1] = dists[:, 1, 1:]  # Caminhante qubit 'down' move para a direita
    dists[:, 1, -1] = dists[:, 1, 0]    # O primeiro vai para a última posição (circular)


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

def main(r=1000, N=2001, steps=1000, threshold=500):
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

    # Inicializando a matriz de distribuições para r caminhadas
    dists = np.zeros((r, 2, N))
    
    # Posicionando o estado inicial psi para todas as distribuições em 'dists'
    dists[:, :, position(0, N)] = psi

    # Gerando uma matriz de r moedas quânticas com ângulos aleatórios para todas as repetições
    coin_array = generate_coin_array_random(r)
    
    # Executando a simulação de maneira completamente vetorizada
    for step in range(steps):
        # Aplicando a operação de flip da moeda para todas as distribuições simultaneamente
        if r<threshold:
            multi_coin_flip(coin_array, dists)
        else:
            multi_coin_flip_small(coin_array, dists)

        # Executando o passo de caminhada vetorizado
        multi_circular_walk(dists)
    
    #print(f"--- %s seconds in Total ---" % (time.time() - start_time))
    print_dists(dists, N, r)
    return dists


def old_main(r=1000, N=2001, steps=1000):
    start_time = time.time()

    psi = up

    dist = np.zeros(2*N).reshape((2,N))
    dist[:,position(0,N)] = psi

    for repetition in range(r):
        theta = np.random.uniform(np.deg2rad(30), np.deg2rad(45))
        coin = generate_single_coin(theta)
        dists = np.array([dist for _ in range(r)])
        for step in range(steps):
            coin_flip(coin,dists[repetition])
            circular_walk(dists[repetition], N)
    
    print("--- %s seconds in total ---" % (time.time() - start_time))
    print_dists(dists, N, r)
    return dists


if __name__ == "__main__":
    main()



