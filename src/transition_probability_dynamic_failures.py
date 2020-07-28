import numpy as np

def basecases(io, no, ic, nc, s, N):
    if (no < 1) or (nc < 1) or (io < 0) or (ic < 0) or (io > no) or (ic > nc):
        return 0
    elif (io, no, ic, nc) == (1, 1, 1, 1):
        return 1 - s
    elif (io, no, ic, nc) == (0, 1, 0, 1):
        return 1
    elif (io, no, ic, nc) == (1, 1, 2, 2):
        return s
    elif (io, no, ic, nc) == (0, 1, 1, 2):
        return s/2
    else:
        return None

def Pf(io, no, ic, nc, s, N):
    v = basecases(io, no, ic, nc, s, N)
    if v is None:
        in_sample  = s * ic/N * Pf(io-1, no-1, ic, nc, s, N)
        out_sample = s * (1 - (nc-1)/N) * ic/nc * Pf(io-1, no-1, ic-1, nc-1, s, N)
        return in_sample + out_sample
    else:
        return v


def P(io, no, ic, nc, s, N):
    v = basecases(io, no, ic, nc, s, N)
    if v is None:
        a = (1 - ((nc - 1) / N)) * ((nc-ic)/N) * P(io, no-1, ic, nc-1, s, N)
        b = (1 - ((nc - 1) / N)) * (ic/nc) * (1-s) * P(io-1, no-1, ic-1, nc-1, s, N)
        c = (ic/N) * (1-s) * P(io-1, no-1, ic, nc, s, N)
        d = ((nc-ic) / N) * P(io, no-1, ic, nc, s, N)
        v = a + b + c + d
    return v
