import numpy as np
from scipy.stats import hypergeom, binom


def basecases_t(io, no, ic, nc, s, t):
    if (
        (no < 0)
        or (nc < 0)
        or (io < 0)
        or (ic < 0)
        or (io > no)
        or (ic > nc)
        or (no > 0 and nc == 0)
    ):
        return 0  # TODO: check if nc> N is also a concern
    elif (io, no, ic, nc, t) == (0, 0, 0, 0, 0):
        return 1
    else:
        return None


def Pf(io, no, ic, nc, s, N, t):
    """The probability of not having had a transition after t failures"""
    v = basecases_t(io, no, ic, nc, s, t)
    if v is None:
        # t is the number of failures
        if t == 0:
            # We will build a recursion over the number of successfully drawn offspring n_0.
            # We need to start our recursion on
            v = P0(io, no, ic, nc, s, N, max_t=0)
        else:  # the curent number of failures was obtained from a previous number of failures
            oos = 1 - ((nc - 1) / N)  # out-of-sample
            # B and C correspond to the cases in P0
            b = oos * (ic / nc) * s * Pf(io, no, ic - 1, nc - 1, s, N, t - 1)

            c = (ic / N) * s * Pf(io, no, ic, nc, s, N, t - 1)

            v = b + c
    return v


def P0(io, no, ic, nc, s, N, max_t):
    """max_t is the maximum number of failures. """

    v = basecases_t(io, no, ic, nc, s, t=max_t)
    if v is None:  # Recursion on the number of
        oos = 1 - ((nc - 1) / N)  # out-of-sample
        sel = 1 - s

        # out of sample, ancestral
        af = sum(Pf(io, no - 1, ic, nc - 1, s, N, t) for t in range(max_t + 1))
        a = oos * ((nc - ic) / nc) * af

        # out of sample, derived
        bf = sum(Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, t) for t in range(max_t + 1))
        b = oos * (ic / nc) * sel * bf

        # in sample, derived
        cf = sum(Pf(io - 1, no - 1, ic, nc, s, N, t) for t in range(max_t + 1))
        c = (ic / N) * sel * cf

        # in sample, ancestral
        df = sum(Pf(io, no - 1, ic, nc, s, N, t) for t in range(max_t + 1))
        d = ((nc - ic) / N) * df

        v = a + b + c + d
        v += (oos * (ic / nc) * s * Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, max_t)) + (
            (ic / N) * s * Pf(io - 1, no - 1, ic, nc, s, N, max_t)
        )  # Forcing success of cases b and c, respectively
    return v


def matrix(no=5, s=1 / 1000, N=1000, max_t=1):
    mtx = np.zeros((no + 1, no + 1))
    for nc in range(0, no + 1):
        for ic in range(0, nc + 1):
            # vectorized over ip
            p = hypergeom.pmf(ic, no, np.arange(0, no + 1), nc)
            for io in range(0, no + 1):
                mtx[:, io] += P0(io, no, ic, nc, s, N, max_t) * p
    return mtx


def matrix_double_sample(n, N, s=0, max_t=1):
    mtx = np.zeros((2 * n + 1, n + 1))
    for ip in range(0, 2 * n + 1):
        for io in range(0, n + 1):
            for nc in range(0, 2 * n + 1):
                for ic in range(0, nc + 1):
                    mtx[ip, io] += P0(io, n, ic, nc, s, N, max_t) * hypergeom.pmf(
                        ic, 2 * n, ip, nc
                    )
    return mtx
