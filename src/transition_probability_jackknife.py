import numpy as np
from scipy.stats import hypergeom
from transition_probability import build_cache_matrices, calcJK13


def hypergeom_projection(N, n):
    rN = np.arange(0, N + 1)
    rn = np.arange(0, n + 1)
    return np.array([hypergeom(N, i, n).pmf(rn) for i in rN])


def matrix_jackknife(n=10, j=1, N=10_000, s=1 / 10_000, max_t=3):
    cache = build_cache_matrices(n, j, N, s, max_t)

    M = np.zeros((n + 1, n + 1))
    for i in range(0, n + 1):
        x = hypergeom_projection(n, i) @ cache[i]
        M += x

    J = np.eye(n + 1)
    for i in range(n + 1, n + j + 1):
        J = calcJK13(i + 1) @ J
        x = J.T @ cache[i]
        M += x

    return M


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sample-size", "-n", default=10, type=int)
    parser.add_argument("--jackknife", "-j", default=1, type=int)
    parser.add_argument("--population-size", "-N", default=1000, type=int)
    parser.add_argument("--selection", "-Ns", default=1, type=int)
    parser.add_argument("--max-resample", "-r", default=3, type=int)
    args = parser.parse_args()
    globals().update(vars(args))

    M = matrix_jackknife(
        n=sample_size,
        j=jackknife,
        N=population_size,
        s=selection / population_size,
        max_t=max_resample,
    )
    print(1 - (M[-1, :].sum()))
