import numpy as np
import matplotlib.pyplot as plt
import time



up = np.array((1,0))
down = np.array((0,1))
hadamar_coin = np.array([[1,1],[1,-1]])*(0.5**0.5)

def position(i,n):
    return int((n+1)/2+i)-1

def print_dist(dist, N):
    limit = N//2
    prob = np.abs(dist[0])**2+np.abs(dist[1])**2
    prob *= 100
    plt.plot(np.arange(-limit,limit+1), prob)
    plt.ylim(0,prob.max()*1.1)
    plt.xlim(-limit-5,limit+5)
    plt.show()

def coin_flip(coin,dist):
    dist[:,:] = np.matmul(coin,dist)

def old_coin_flip(coin,N):
    for i in range(N):
        dist[:,i] = np.matmul(coin,dist[:,i])

def walk(dist, N):
    dist[0,1:] = dist[0,:N-1]
    dist[0,0] = dist[0,N-1]
    dist[1,:N-1] = dist[1,1:]
    dist[1,N-1] = dist[1,0]

def old_walk(dist, N):
    for i in range(1,N):
        dist[0,N-i],dist[0,N-i-1] = dist[0,N-i+1],dist[0,N-i]
    dist[0,N-1] = dist[0,0]
    for i in range(N-1):
        dist[1,i+1],dist[1,i+2] = dist[1,i],dist[1,i+1]
    dist[1,0] = dist[1,N-1]


#####################################################################################################
#                                                                                                   #
#                                              MAIN                                                 #
#                                                                                                   #
#####################################################################################################

if __name__ == "__main__":

    psi = up

    N = 1001
    steps = N//2
    coin = hadamar_coin
    dist = np.zeros(2*N).reshape((2,N))
    dist[:,position(0,N)] = psi

    start_time = time.time()
    for i in range(steps):
        coin_flip(coin,dist)
        walk(dist, N)
    print("--- %s seconds ---" % (time.time() - start_time))
    print_dist(dist, N)

