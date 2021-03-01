import numpy as np
import scipy.stats as st


def psi_diploid(i, N, s=0, h=0.5, u=1e-9, v=1e-9):
    j = (N) - i
    w_00, w_01, w_11 = 1 + s, 1 + (s * h), 1
    a, b, c = w_00 * i * i, w_01 * i * j, w_11 * j * j
    w_bar = a + (2 * b) + c
    return (((a + b) * (1 - u)) + ((b + c) * v)) / w_bar


def psi_haploid(i, N, s=0, u=0, v=0):
    j = (N) - i
    f = 1 + s
    iu = 1 - u
    return ((i * f * iu) + (j * v)) / ((i * f) + j)


def wright_fisher(N, s=0, h=0.5, u=0, v=0):
    P = np.zeros((N + 1, N + 1))
    r = np.arange(0, N + 1)

    for i in np.arange(0, N + 1):
        P[i, :] = st.binom.pmf(r, N, psi_diploid(i, N, s, h, u, v))

    return P


def wright_fisher_haploid(N, s=0, u=0, v=0):
    P = np.zeros((N + 1, N + 1))
    r = np.arange(0, N + 1)

    for i in np.arange(0, N + 1):
        P[i, :] = st.binom.pmf(r, N, psi_haploid(i, N, s, u, v))

    return P
