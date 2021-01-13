import numpy as np
from scipy.stats import hypergeom
from transition_probability import build_cache_matrices, calcJK13

def hypergeom_projection(N, n):
    rN = np.arange(0, N+1)
    rn = np.arange(0, n+1)
    return np.array([hypergeom(N, i, n).pmf(rn) for i in rN])


def matrix_jackknife(n=10, k=1, N=10_000, s=1/10_000, max_t=3):
    cache = build_cache_matrices(n, k, N, s, max_t)

    M = np.zeros((n+1, n+1))
    for i in range(0, n+1):
        x = hypergeom_projection(n, i) @ cache[i]
        M += x

    J = np.eye(n+1)
    for i in range(n+1, n+k+1):
        J = calcJK13(i+1) @ J
        x = J.T @ cache[i]
        M += x

    return M
