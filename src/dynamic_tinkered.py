import numpy as np
from scipy.stats import hypergeom, binom


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
        return None


def Pf(io, no, ic, nc, s, N, t, max_t):
    """The probability of not having had a transition after t failures"""

    v = basecases_t(io, no, ic, nc, s, t)
    #print("v in pf", v, "for params ",io, no, ic, nc, s, t)
    if v is None:
        # t is the number of failures
        if t <= 0:
            # We will build a recursion over the number of successfully drawn offspring n_0.
            # We need to start our recursion on
            v = P0(io, no, ic, nc, s, N, max_t=max_t)
            # I belive that the bug is here. By computing
            # P0 with max_t set to zero, we are forcing the
            # first draw to be successful. We wanted
            # to have P0 after zero attempts, not P0
            # after we forced the first attempt to be successful.

        else:  # the curent number of failures was obtained from a previous number of failures
            oos = 1 - ((nc - 1) / N)  # out-of-sample
            # B and C correspond to the cases in P0
            b = oos * (ic / nc) * s * Pf(io, no, ic - 1, nc - 1, s, N, t - 1, max_t)

            c = (ic / N) * s * Pf(io, no, ic, nc, s, N, t - 1, max_t)

            v = b + c
    return v


def P0(io, no, ic, nc, s, N, max_t, force_success=True):
    """max_t is the maximum number of failures. """

    v = basecases_t(io, no, ic, nc, s, t=max_t)
    if v is None:  # Recursion on what happened in the last lineage.
        oos = 1 - ((nc - 1) / N)  # out-of-sample
        sel = 1 - s

        # out of sample, ancestral
        af = sum(Pf(io, no - 1, ic, nc - 1, s, N, t, max_t) for t in range(max_t + 1))
        a = oos * ((nc - ic) / nc) * af

        # out of sample, derived
        bf = sum(Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, t, max_t) for t in range(max_t + 1))
        b = oos * (ic / nc) * sel * bf

        # in sample, derived
        cf = sum(Pf(io - 1, no - 1, ic, nc, s, N, t, max_t) for t in range(max_t + 1))
        c = (ic / N) * sel * cf

        # in sample, ancestral
        df = sum(Pf(io, no - 1, ic, nc, s, N, t, max_t) for t in range(max_t + 1))
        d = ((nc - ic) / N) * df
        # print("io, no, ic, nc", io, no, ic, nc)
        # print("a,b,c, d", a,b,c,d)
        v = a + b + c + d
        if force_success:
            v += (oos * (ic / nc) * s * Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, max_t, max_t)) + (
                (ic / N) * s * Pf(io - 1, no - 1, ic, nc, s, N, max_t, max_t)
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
