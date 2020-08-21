import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.stats import hypergeom, binom
import scipy.sparse as sr
from numba import jit
from occupancy import reduced_occupancy, dist_num_anc

@jit(nopython=True)
def Qs(io, no, ic, nc, N, s, cache):
    """Transition probability matrix entries with selection"""
    if (
        (no < 0)
        or (nc < 0)
        or (io < 0)
        or (ic < 0)
        or (io > no) # There cannot be more derived offspring than offspring
        or (ic > nc) # There cannot be more derived contributors than contributors
        or (no > 0 and nc == 0) # There cannot be offspring without contributors
        or (io>0 and ic==0) # There cannot be derived offspring without derived parent
        or (io<no and ic == nc) # There cannot be ancestral offspring without derived ancestors
    ):
        return 0
    v = cache[nc,no,ic,io]

    if np.isnan(v):
        if (io, no, ic, nc) == (1, 1, 1, 1):
            v = (1 - s) + (s/N)
        elif (io, no, ic, nc) == (0, 1, 0, 1):
            v = 1
        elif (io, no, ic, nc) == (1, 1, 2, 2):
            v = s * (1-(1/N))
        elif (io, no, ic, nc) == (0, 1, 1, 2):
            v = (s / 2) * (1-(1/N))
        else:
            # Out-sample
            oos = 1 - ((nc-1) / N)
            oos2 = 1 - ((nc-2) / N)
            # In all cases, we force the last attempt to sucseed
            # If there are two attempts, the first one fails

            # Ancestral succeeds
            # out-of-sample ancestral
            Q = Qs(io, no-1, ic, nc-1, N, s, cache)
            Q1a = 0 if Q == 0 else oos * ((nc-ic)/nc) * Q

            # in-sample ancestral
            Q = Qs(io, no-1, ic, nc, N, s, cache)
            Q2a = 0 if Q == 0 else ((nc-ic)/N) * Q

            # out-of-sample derived (fail), then out-of-sample ancesral (success)
            Q = Qs(io, no-1, ic-1, nc-2, N, s, cache)
            Q3a = 0 if Q == 0 else oos2 * (ic/nc) * s * oos * ((nc-ic)/(nc-1)) * Q

            # in-sample derived (fail), then in-sample ancestral (success)
            Q = Qs(io, no-1, ic, nc, N, s, cache)
            Q4a = 0 if Q == 0 else (ic/N) * s * ((nc-ic)/(N)) * Q

            # in-sample derived (fail), then out-of-sample ancestral (success)
            Q = Qs(io, no-1, ic, nc-1, N, s, cache)
            Q5a = 0 if Q == 0 else (ic/N) * s * oos * ((nc-ic)/(nc)) * Q # should the last one be nc-1?

            # out-of-sample derived (fail), then in-sample ancestral (success)
            Q = Qs(io, no-1, ic-1, nc-1, N, s, cache)
            Q6a = 0 if Q == 0 else oos * (ic/nc) * s * ((nc - ic) / N) * Q

            # Derived succeeds
            # out-of-sample derived
            Q = Qs(io-1, no-1, ic-1, nc-1, N, s, cache)
            Q1d = 0 if Q == 0 else oos * (ic/nc) * (1-s) * Q

            # in-sample derived
            Q = Qs(io-1, no-1, ic, nc, N, s, cache)
            Q2d = 0 if Q == 0 else (ic/N) * (1-s) * Q

            # out-of-sample derived (fail), then out-of-sample derived (success)
            Q = Qs(io-1, no-1, ic-2, nc-2, N, s, cache)
            Q3d = 0 if Q == 0 else oos2 * (ic/nc) * s * oos * ((ic-1)/(nc-1)) * Q

            # in-sample derived (fail), then in-sample derived (success)
            Q = Qs(io-1, no-1, ic, nc, N, s, cache)
            Q4d = 0 if Q == 0 else (ic/N) * s * ((ic-1)/(N)) * Q

            # in-sample derived (fail), then out-of-sample derived (success)
            Q = Qs(io-1, no-1, ic-1, nc-1, N, s, cache)
            Q5d = 0 if Q == 0 else ((ic-1)/N) * s * oos * (ic/nc) * Q

            # out-of-sample derived (fail), then in-sample derived (success)
            Q = Qs(io-1, no-1, ic-1, nc-1, N, s, cache)
            Q6d = 0 if Q == 0 else oos * (ic/nc) * s * ((ic-1) / N) * Q

            v = Q1a + Q1d + Q2a + Q2d + Q3a + Q3d + Q4a + Q4d + Q5a + Q5d + Q6a + Q6d

            # extra cases - pick and reject the same allele - both derived

            # out-of-sample derived fail, then same success
            Q = Qs(io-1, no-1, ic-1, nc-1, N, s, cache)
            Q7 = 0 if Q == 0 else oos * (ic / nc) * s * (1/(N)) * Q

            # in-sample derived fail, then same success
            Q = Qs(io-1, no-1, ic, nc, N, s, cache)
            Q8 = 0 if Q == 0 else (ic / N) * s * (1/(N)) * Q

            v += Q7 + Q8

            cache[nc,no,ic,io] = v

    return v


def matrix_selection(n, N, s=0):
    mtx = np.zeros((n + 1, n + 1))
    cache = np.full((n+1, n+1, n+1, n+1), np.nan) # this is messier (we allocate way more than we need, but it's easier for numba)
    # cache = [[np.full((i+1, j+1), np.nan) for j in range(1, n+1)] for i in range(1, n+1)] # this is the minimal size allocation possible
    for nc in range(0, n + 1):
        for ic in range(0, nc + 1):
            # vectorized over ip
            p = hypergeom.pmf(ic, n, np.arange(0, n+1), nc)
            for io in range(0, n + 1):
                mtx[:, io] += Qs(io, n, ic, nc, N, s, cache) * p
    return mtx, cache

def matrix_selection_nop(n, N, s=0):
    mtx = np.zeros((n + 1, n + 1))
    cache = np.full((n+1, n+1, n+1, n+1), np.nan) # this is messier (we allocate way more than we need, but it's easier for numba)
    # cache = [[np.full((i+1, j+1), np.nan) for j in range(1, n+1)] for i in range(1, n+1)] # this is the minimal size allocation possible
    for nc in range(0, n + 1):
        for ic in range(0, nc + 1):
            # vectorized over ip
            # p = hypergeom.pmf(ic, n, np.arange(0, n+1), nc)
            for io in range(0, n + 1):
                mtx[:, io] += Qs(io, n, ic, nc, N, s, cache)
    return mtx, cache


def matrix_selection_more_contributors(n, N, s=0):
    rocc = reduced_occupancy(N)
    mtx = np.zeros((n + 1, n + 1))
    # cache is (io, no, ip, nc)
    cache = np.full((2*n+1, n+1, 2*n+1, n+1), np.nan)
    for ip in range(0, n + 1):
        for io in range(0, n + 1):
            for nc in range(0, n+1):
                for ic in range(0, nc+1):
                    p = binom.pmf(ic, nc, ip/n)

                    mtx[ip, io] += Qs(io, n, ic, nc, N, s, cache) * p

    # for nc in range(0, n+1):
    #     for ic in range(0, nc+1):
    #         # p = binom.pmf(ic, nc, np.arange(0, n+1)/n)
    #         p = hypergeom.pmf(ic, n, np.arange(0, n+1), nc)
    #         for io in range(0, n + 1):
    #             mtx[:, io] += Qs(io, n, ic, nc, N, s, cache) * p

    return mtx, cache


def plot_cache(cache):
    """Plot sparse cache"""
    n0 = len(cache)
    n1 = len(cache[0])
    fig, ax = plt.subplots(nrows=n0, ncols=n1)
    for i in range(n0):
        for j in range(n1):
            ax[i,j].set(xlim=(-0.5,n0+0.5), ylim=(-0.5,n1+0.5))
            ax[i,j].set_axis_off()
            ax[i,j].matshow(cache[i][j])
    return fig

def plot_dense_cache(cache):
    n0 = len(cache)
    n1 = len(cache[0])
    fig, ax = plt.subplots(nrows=n0, ncols=n1)
    for i in range(n0):
        for j in range(n1):
            ax[i,j].set(xlim=(-0.5,n0+0.5), ylim=(-0.5,n1+0.5))
            ax[i,j].set_axis_off()
            ax[i,j].matshow(cache[i,j])
    return fig


# WIP
def matrix_double_sample_selection(n, N, s=0):
    mtx = np.zeros((2*n + 1, n + 1))
    # cache = [[np.full((i+1, j+1), np.nan) for j in range(1, n+1)] for i in range(1, 2*n+1)]
    cache =  np.full((2*n+1, 2*n+1, 2*n+1, 2*n+1), np.nan)
    for ip in range(0, 2*n + 1):
        for io in range(0, n + 1):
            for nc in range(0, 2*n+1):
                for ic in range(0, nc+1):
                    mtx[ip, io] += Qs(io, n, ic, nc, N, s, cache) * hypergeom.pmf(ic, 2*n, ip, nc)
    return mtx, cache
