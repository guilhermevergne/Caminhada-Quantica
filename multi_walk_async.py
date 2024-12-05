import numpy as np
import matplotlib.pyplot as plt
import time
import asyncio

from Quantum_walk_libs import up, down, hadamard_coin
from Quantum_walk_libs import position, print_dists
from Quantum_walk_libs import generate_single_coin, coin_flip, walk, circular_walk

#####################################################################################################
#                                                                                                   #
#                                              MAIN                                                 #
#                                                                                                   #
#####################################################################################################

async def cycle(dist, N=2001, steps=1000, circular = False):
    theta = np.random.uniform(np.deg2rad(30), np.deg2rad(45))
    coin = generate_single_coin(theta)
    for step in range(steps):
        coin_flip(coin,dist)
        if circular:
            circular_walk(dist, N)
        else:
            walk(dist, N)


async def main(r=1000, N=2001, steps=1000, circular = False):
    start_time = time.time()

    psi = up

    dist = np.zeros(2*N).reshape((2,N))
    dist[:,position(0,N)] = psi
    dists = np.array([dist for _ in range(r)])
    tasks = []

    for repetition in range(r):
        theta = np.random.uniform(np.deg2rad(30), np.deg2rad(45))
        coin = generate_single_coin(theta)
        tasks.append(asyncio.create_task(cycle(dists[repetition], N, steps, circular)))
    [await task for task in tasks]
    print("--- %s seconds in total ---" % (time.time() - start_time))
    print_dists(dists, N, r)
    return dists


if __name__ == "__main__":
    asyncio.run(main())