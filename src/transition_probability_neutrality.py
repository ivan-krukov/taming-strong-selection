import numpy as np
import matplotlib.pyplot as plt


def Pd(j, i, n, N, matrices):
    
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
            j1i1 = Pd(j-1, i-1, n-1, N, matrices)
            j1i0 = Pd(j-1, i  , n-1, N, matrices)
            j0i1 = Pd(j  , i-1, n-1, N, matrices)
            j0i0 = Pd(j  , i  , n-1, N, matrices)

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


def matrix_neutrality(n, N):
    matrices = [np.full((i + 1, i + 1), np.nan) for i in range(n + 1)]

    Q = np.zeros((n + 1, n + 1))
    for i in range(0, n + 1):
        for j in range(0, n + 1):
            Q[j, i] = Pd(j, i, n, N, matrices)
    return Q
