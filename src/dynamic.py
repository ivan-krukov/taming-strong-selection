import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

N = 10

@lru_cache(maxsize=500)
def P(j, i, n, depth=0, verbose=False):
    """Neutral sample building"""

    # v = 0

    # if (n == 0):
    #     v = 0
    # elif (i < 0) or (j < 0):
    #     v = 0
    # elif n == 1:
    #     if (j == i == 1) or (j == i == 0):
    #         v = 1
    if (n == 0) or (i < 0) or (j < 0) or (j > n) or (i > n):
        v = 0
    elif n == 1:
        if (j, i) == (1, 1):
            v = 1
        elif (j, i) == (0, 0):
            v = 1
        elif (j, i) == (1, 0):
            v = 0
        elif (j, i) == (0, 1):
            v = 0
        else:
            raise Exception("Impossible")

    else:
        j1i1 = P(j-1, i-1, n-1, depth+1, verbose)
        j1i0 = P(j-1, i  , n-1, depth+1, verbose)
        j0i1 = P(j  , i-1, n-1, depth+1, verbose)
        j0i0 = P(j  , i  , n-1, depth+1, verbose)

        # derived picked
        d  = j1i1 * (1 - ((n-1) / N))
        d += j1i1 * ((j-1) / N)
        d += j1i0 * ((n-j) / N)

        # ancestal picked
        a  = j0i0 * (1 - ((n-1) / N))
        a += j0i0 * ((n-j-1) / N)
        a += j0i1 * (j / N)

        v = ((j/n) * d) + ((n-j)/n * a)

    if verbose:
        print((" " * depth) + f"{j}/{n} -> {i}/{n} = {v}")

    return v

matrices = [np.full((i + 1, i + 1), np.nan) for i in range(50 + 1)]

def Pd(j, i, n):
    
    if (n == 0) or (i < 0) or (j < 0) or (j > n) or (i > n):
        v = 0
    elif n == 1:
        if (j, i) == (1, 1):
            v = 1
        elif (j, i) == (0, 0):
            v = 1
        elif (j, i) == (1, 0):
            v = 0
        elif (j, i) == (0, 1):
            v = 0
        else:
            raise Exception("Impossible")

    else:
        v = matrices[n][j, i]
        if np.isnan(v):
            j1i1 = Pd(j-1, i-1, n-1)
            j1i0 = Pd(j-1, i  , n-1)
            j0i1 = Pd(j  , i-1, n-1)
            j0i0 = Pd(j  , i  , n-1)

            # derived picked
            d  = j1i1 * (1 - ((n-1) / N))
            d += j1i1 * ((j-1) / N)
            d += j1i0 * ((n-j) / N)

            # ancestal picked
            a  = j0i0 * (1 - ((n-1) / N))
            a += j0i0 * ((n-j-1) / N)
            a += j0i1 * (j / N)

            v = ((j/n) * d) + ((n-j)/n * a)
            matrices[n][j, i] = v

    return v
