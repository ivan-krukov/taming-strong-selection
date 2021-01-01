from time import perf_counter
import numpy as np
from scipy.stats import hypergeom

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
        return 0  # TODO: check if nc>N is also a concern
    elif (io, no, ic, nc) == (0, 0, 0, 0): # Not sure what that means.
        if (t==0):
            return 1
        else:
            return 0
    else:
        return np.nan

def Pf(io, no, ic, nc, s, N, t, max_t, cache):
    # v = cache[io, no, ic, nc, t]
    z = cache[t][nc][no]
    print(t, nc, no, ic, io, z.shape)


    v = basecases_t(io, no, ic, nc, s, t)
    #print("v in pf", v, "for params ",io, no, ic, nc, s, t)
    if np.isnan(v):
        v = cache[t][nc][no][ic, io]
        if np.isnan(v):
            if t <= 0:
                assert t == 0
                v = P0(io, no, ic, nc, s, N, max_t, cache)
                # assert v == cache[io, no, ic, nc, t]

            else:  # the curent number of failures was obtained from a previous number of failures
                oos = 1 - ((nc - 1) / N)  # out-of-sample
                b = oos * (ic / nc) * s * Pf(io, no, ic - 1, nc - 1, s, N, t - 1, max_t, cache)
                c = (ic / N) * s * Pf(io, no, ic, nc, s, N, t - 1, max_t, cache)
                v = b + c

    if (nc >= 0) and (no >= 0) and (ic >= 0) and (io >= 0) and (ic <= nc) and (io <= no):
        cache[t][nc][no][ic, io] = v
            # cache[io, no, ic, nc, t] = v
    print(v)
    return v

def P0(io, no, ic, nc, s, N, max_t, cache):

    # v = cache[io, no, ic, nc, 0]
    v = basecases_t(io, no, ic, nc, s, t=max_t)
    if np.isnan(v):  # Recursion on what happened in the last lineage.
        # v = np.nan
        v = cache[0][nc][no][ic, io]
        if np.isnan(v):
            oos = 1 - ((nc - 1) / N)  # out-of-sample
            sel = 1 - s

            af = 0
            for t in range(max_t, -1, -1):
                af +=  Pf(io, no - 1, ic, nc - 1, s, N, t, max_t, cache)
            a = oos * ((nc - ic) / nc) * af

            bf = 0
            for t in range(max_t, -1, -1):
                bf += Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, t, max_t, cache)
            b = oos * (ic / nc) * sel * bf

            cf = 0
            for t in range(max_t, -1, -1):
                cf += Pf(io - 1, no - 1, ic, nc, s, N, t, max_t, cache)
            c = (ic / N) * sel * cf

            df = 0
            for t in range(max_t, -1, -1):
                df += Pf(io, no - 1, ic, nc, s, N, t, max_t, cache)
            d = ((nc - ic) / N) * df

            v = a + b + c + d
            v += (oos * (ic / nc) * s * Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, max_t, max_t, cache)) + (
                (ic / N) * s * Pf(io - 1, no - 1, ic, nc, s, N, max_t, max_t, cache)
            )  # Forcing success of cases b and c, respectively

            # cache[io, no, ic, nc, 0] = v
    if (nc >= 0) and (no >= 0) and (ic >= 0) and (io >= 0) and (ic <= nc) and (io <= no):
        cache[0][nc][no][ic, io] = v
    return v


def matrix_explicit(no=5, s=1/1000, N=1000, max_t=1):
    mtx = np.zeros((no+1, no+1))
    # cache = np.full((no+1, no+1, no+1, no+1, max_t+1), np.nan)
    cache = [[[np.full((i+1, j+1), np.nan) for j in range(0, no+1)] for i in range(0, no+1)] for t
             in range(0, max_t+1)]

    for nc in range(0, no+1):
        for ic in range(0, nc+1):
            p = hypergeom.pmf(ic, no, np.arange(0, no+1), nc)

            for io in range(0, no+1):
                mtx[:, io] += P0(io, no, ic, nc, s, N, max_t, cache) * p

    return mtx, cache

if __name__ == "__main__":
    from transition_probability_dynamic_failures import matrix

    np.set_printoptions(precision=2, linewidth=100)
    n = 3
    max_t = 2
    t_start = perf_counter()
    M1, cache = matrix_explicit(n, max_t=max_t)
    print("Took ", perf_counter() - t_start)
    t_start = perf_counter()
    M2, _, _ = matrix(n, max_t=max_t)
    print("Took ", perf_counter() - t_start)
    print("Last")
    print(M1)
    print(M2)
    diff = (M1 - M2) / M2
    mdiff = diff[~np.isnan(diff)]
    print(mdiff)
    print(np.max(np.abs(mdiff)))
