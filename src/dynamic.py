import numpy as np
from scipy.stats import hypergeom, binom


def basecases(io, no, ic, nc, s):
    if (no < 1) or (nc < 1) or (io < 0) or (ic < 0) or (io > no) or (ic > nc):
        return 0
    elif (io, no, ic, nc) == (1, 1, 1, 1):
        return 1 - s
    elif (io, no, ic, nc) == (0, 1, 0, 1):
        return 1
    elif (io, no, ic, nc) == (1, 1, 2, 2):
        return s
    elif (io, no, ic, nc) == (0, 1, 1, 2):
        return s / 2
    else:
        return None


# now only one failure
def Pf(io, no, ic, nc, s, N, t=1):
    v = basecases(io, no, ic, nc, s)
    if v is None:
        if t == 0:
            v = P0(io, no, ic, nc, s, N)
        else:
            a = s * ic / N * Pf(io, no, ic, nc, s, N, t - 1)
            oos = 1 - (nc - 1) / N  # out-of-sample
            b = oos * ic / nc * s * Pf(io, no, ic - 1, nc - 1, s, N, t - 1)
            v = a + b
    return v


def P0(io, no, ic, nc, s, N):
    v = basecases(io, no, ic, nc, s)
    if v is None:
        # TODO: give the terms better names
        oos = 1 - (nc - 1) / N  # out-of-sample
        sel = 1 - s

        af = Pf(io, no - 1, ic, nc - 1, s, N, t=0)
        a = oos * (nc - ic) / nc * af
        bf = Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, t=0)
        b = oos * ic / nc * sel * bf
        cf = Pf(io - 1, no - 1, ic, nc, s, N, t=0)
        c = ic / N * sel * cf
        df = Pf(io, no - 1, ic, nc, s, N, t=0)
        d = (nc - ic) / N * df
        v = a + b + c + d
    return v


def matrix(no=5, s=1 / 1000, N=1000):
    z = no + 1
    mtx = np.zeros((z, z))
    for nc in range(0, no + 1):
        for ic in range(0, nc + 1):
            # vectorized over ip
            p = hypergeom.pmf(ic, no, np.arange(0, no + 1), nc)
            for io in range(0, no + 1):
                mtx[:, io] += P0(io, no, ic, nc, s, N) * p
    return mtx
