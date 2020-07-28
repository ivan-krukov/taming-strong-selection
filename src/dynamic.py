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
def Pf(io, no, ic, nc, s, N, t):
    v = basecases(io, no, ic, nc, s)
    if v is None:
        if t == 0:
            v = P0(io, no, ic, nc, s, N)
        else:
            a = s * (ic / N) * Pf(io - 1, no, ic, nc, s, N, t - 1)
            oos = 1 - (nc - 1) / N  # out-of-sample
            b = oos * (ic / nc) * s * Pf(io - 1, no, ic - 1, nc - 1, s, N, t - 1)
            v = a + b
    return v


def P0(io, no, ic, nc, s, N, max_t=1):
    v = basecases(io, no, ic, nc, s)
    if v is None:
        oos = 1 - (nc - 1) / N  # out-of-sample
        sel = 1 - s

        af = sum(Pf(io, no - 1, ic, nc - 1, s, N, t=i) for i in range(max_t))
        a = oos * (nc - ic) / nc * af
        bf = sum(Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, t=i) for i in range(max_t))
        b = oos * ic / nc * sel * bf
        cf = sum(Pf(io - 1, no - 1, ic, nc, s, N, t=i) for i in range(max_t))
        c = ic / N * sel * cf
        df = sum(Pf(io, no - 1, ic, nc, s, N, t=i) for i in range(max_t))
        d = (nc - ic) / N * df
        v = a + b + c + d
        v += (oos * (ic / nc) * s * Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, max_t)) + (
            (ic / N) * s * Pf(io - 1, no - 1, ic, nc, s, N, max_t)
        )
    return v


def fold_last_entry(x, size):
    l = size - 1
    head = x[:l]
    tail = np.array(x[l:].sum())
    return np.concatenate((head, tail.reshape(1)))


def matrix(no=5, s=1 / 1000, N=1000, max_t=1, u=1):
    z = no + 1
    mtx = np.zeros((z, z))
    for nc in range(0, no + u):
        for ic in range(0, nc + 1):
            # vectorized over ip
            p = hypergeom.pmf(ic, no + (u - 1), np.arange(0, no + u), nc)
            q = p[:z]
            for io in range(0, no + 1):
                mtx[:, io] += P0(io, no, ic, nc, s, N, max_t) * q
    return mtx
