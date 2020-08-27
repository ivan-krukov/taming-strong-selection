import numpy as np
from scipy.stats import hypergeom, binom
from numba import jit

@jit(nopython=True)
def basecases_t(io, no, ic, nc, s, t):
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
        return 0  # TODO: check if nc> N is also a concern
    elif (io, no, ic, nc) == (0, 0, 0, 0): # Not sure what that means.
        if (t==0):
            return 1
        else:
            return 0
    else:
        return np.nan

@jit(nopython=True)
def Pf(io, no, ic, nc, s, N, t, max_t, cache_P0, cache_Pf):
    """The probability of not having had a transition after t failures"""
    v = cache_Pf[io, no, ic, nc, t]
    
    #print("v in pf", v, "for params ",io, no, ic, nc, s, t)
    if np.isnan(v):
        v = basecases_t(io, no, ic, nc, s, t)
        if np.isnan(v):
            
            # t is the number of failures
            if t <= 0:
                # We will build a recursion over the number of successfully drawn offspring n_0.
                # We need to start our recursion on
                v = P0(io, no, ic, nc, s, N, max_t, cache_P0, cache_Pf)

            else:  # the curent number of failures was obtained from a previous number of failures
                oos = 1 - ((nc - 1) / N)  # out-of-sample
                # B and C correspond to the cases in P0
                b = oos * (ic / nc) * s * Pf(io, no, ic - 1, nc - 1, s, N, t - 1, max_t, cache_P0, cache_Pf)

                c = (ic / N) * s * Pf(io, no, ic, nc, s, N, t - 1, max_t, cache_P0, cache_Pf)
                v = b + c

    cache_Pf[io, no, ic, nc, t] = v
    return v

@jit(nopython=True)
def P0(io, no, ic, nc, s, N, max_t, cache_P0, cache_Pf):
    """max_t is the maximum number of failures. """

    
    v = cache_P0[io, no, ic, nc]
    if np.isnan(v):  # Recursion on what happened in the last lineage.
        v = basecases_t(io, no, ic, nc, s, t=max_t)
        if np.isnan(v):
            oos = 1 - ((nc - 1) / N)  # out-of-sample
            sel = 1 - s

            # out of sample, ancestral
            af = 0
            for t in range(max_t, -1, -1):
                af +=  Pf(io, no - 1, ic, nc - 1, s, N, t, max_t, cache_P0, cache_Pf)
            a = oos * ((nc - ic) / nc) * af

            # out of sample, derived
            bf = 0
            for t in range(max_t, -1, -1):
                bf += Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, t, max_t, cache_P0, cache_Pf)
            b = oos * (ic / nc) * sel * bf

            # in sample, derived
            cf = 0
            for t in range(max_t, -1, -1):
                cf += Pf(io - 1, no - 1, ic, nc, s, N, t, max_t, cache_P0, cache_Pf)
            c = (ic / N) * sel * cf

            # in sample, ancestral
            df = 0
            for t in range(max_t, -1, -1):
                df += Pf(io, no - 1, ic, nc, s, N, t, max_t, cache_P0, cache_Pf)
            d = ((nc - ic) / N) * df

            v = a + b + c + d
            v += (oos * (ic / nc) * s * Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, max_t, max_t, cache_P0, cache_Pf)) + (
                (ic / N) * s * Pf(io - 1, no - 1, ic, nc, s, N, max_t, max_t, cache_P0, cache_Pf)
            )  # Forcing success of cases b and c, respectively

    cache_P0[io, no, ic, nc] = v
    return v


def matrix(no=5, s=1 / 1000, N=1000, max_t=1):
    mtx = np.zeros((no + 1, no + 1))
    cache_P0 = np.full((no+1, no+1, no+1, no+1), np.nan)
    cache_Pf = np.full((no+1, no+1, no+1, no+1, max_t+1), np.nan)
    for nc in range(0, no + 1):
        for ic in range(0, nc + 1):
            # vectorized over ip
            p = hypergeom.pmf(ic, no, np.arange(0, no + 1), nc)
            for io in range(0, no + 1):
                mtx[:, io] += P0(io, no, ic, nc, s, N, max_t, cache_P0, cache_Pf) * p
    return mtx, cache_P0, cache_Pf
