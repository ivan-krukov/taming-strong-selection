import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
import scipy.sparse as sr

def Qn(io, no, ic, nc, N, cache, n):
    
    if (io, no, ic, nc) == (1, 1, 1, 1):
        v = 1
    elif (io, no, ic, nc) == (0, 1, 0, 1):
        v = 1
    elif (no < 1) or (nc < 1) or (io < 0) or (ic < 0) or (io > no) or (ic > nc) or (no < nc) or (io < ic):
        v = 0
    else:
        v = cache[io, no, ic, nc]
        
        
        if np.isnan(v):
            
            Qn(io, no-1, ic, nc-1, N, cache, n)
            Q1a = 0 if Q == 0 else (1 - ((nc-1)/N)) * (nc-ic)/nc * Q
            
            Qn(io-1, no-1, ic-1, nc-1, N, cache, n)
            Q1d = 0 if Q == 0 else (1 - ((nc-1)/N)) * ic/nc * Q
            
            Qn(io, no-1, ic, nc, N, cache, n)
            Q2a = 0 if Q == 0 else (nc-ic)/N * Q

            Qn(io-1, no-1, ic, nc, N, cache, n)
            Q2d = 0 if Q == 0 else (ic/N) * Q

            v = Q1a + Q1d + Q2a + Q2d
        cache[io, no, ic, nc] = v
        
    return v


def Qs(io, no, ic, nc, N, s, cache):
    if (io, no, ic, nc) == (1, 1, 1, 1):
        v = 1 - s
    elif (io, no, ic, nc) == (0, 1, 0, 1):
        v = 1
    elif (io, no, ic, nc) == (1, 1, 2, 2):
        v = s
    elif (io, no, ic, nc) == (0, 1, 1, 2):
        v = s / 2
    elif (no < 1) or (nc < 1) or (io < 0) or (ic < 0) or (io > no) or (ic > nc) or (no < nc) or (io < ic):
        v = 0
    else:
        v = cache[io, no, ic, nc]

        if np.isnan(v):
            
            Q = Qs(io, no-1, ic, nc-1, N, s, cache)
            Q1a = 0 if Q == 0 else (1 - ((nc-1)/N)) * (nc-ic)/nc * Q

            Q = Qs(io-1, no-1, ic-1, nc-1, N, s, cache)
            Q1d = 0 if Q == 0 else (1 - ((nc-1)/N)) * ic/nc * (1-s) * Q

            Q = Qs(io, no-1, ic, nc, N, s, cache)
            Q2a = 0 if Q == 0 else (nc-ic)/N * Q

            Q = Qs(io-1, no-1, ic, nc, N, s, cache)
            Q2d = 0 if Q == 0 else (ic/N) * (1-s) * Q

            Q = Qs(io, no-1, ic-1, nc-2, N, s, cache)
            Q3a = 0 if Q == 0 else (1 - ((nc-2)/N)) * (ic/nc) * s * (1 - ((nc-1)/N)) * Q

            Q = Qs(io-1, no-1, ic-2, nc-2, N, s, cache)
            Q3d = 0 if Q == 0 else (1 - ((nc-2)/N)) * (ic/nc) * s * ((ic-1)/(nc-1)) * Q
            
            Q = Qs(io, no-1, ic, nc, N, s, cache)
            Q4a = 0 if Q == 0 else (ic/N) * s * ((nc-ic)/N) * Q
            
            Q = Qs(io-1, no-1, ic, nc, N, s, cache)
            Q4d = 0 if Q == 0 else (ic/N) * s * ((ic-1)/N) * Q
            
            Q = Qs(io, no-1, ic, nc-1, N, s, cache)
            Q5a = 0 if Q == 0 else (ic/N) * s * (1-((nc-1)/N)) * (nc-ic)/nc * Q
            
            Q = Qs(io-1, no-1, ic-1, nc-1, N, s, cache)
            Q5d = 0 if Q == 0 else (ic/N) * s * (1-((nc-1)/N)) * ic/nc * Q
            
            Q = Qs(io, no-1, ic-1, nc-1, N, s, cache)
            Q6a = 0 if Q == 0 else (1-((nc-1)/N)) * ic/nc * s * (nc - ic) / N * Q
            
            Q = Qs(io-1, no-1, ic-1, nc-1, N, s, cache)
            Q6d = 0 if Q == 0 else (1-((nc-1)/N)) * ic/nc * s * (ic-1) / N * Q
            
            v = Q1a + Q1d + Q2a + Q2d + Q3a + Q3d + Q4a + Q4d + Q5a + Q5d + Q6a + Q6d
            
        cache[io, no, ic, nc] = v
        
    return v

def matrix_dumb(n, N):
    mtx = np.zeros((n + 1, n + 1))
    cache = np.full((n+1, n+1, n+1, n+1), np.nan)
    # cache = [np.full((i+1, i+1, i+1), np.nan) for i in range(n + 1)]
    for ip in range(0, n + 1):
        for io in range(0, n + 1):
            q = 0
            for nc in range(0, n+1):
                for ic in range(0, nc+1):
                    q += Q(io, n, ic, nc, N, cache, n) * hypergeom.pmf(ic, n, ip, nc)
            mtx[ip, io] = q
    return mtx


def matrix_tangle(n, N):
    mtx = np.zeros((n + 1, n + 1))
    # cache = [np.full((n+1, n+1, n+1), np.nan) for i in range(0, n+1)]
    cache = np.full((n+1, n+1, n+1, n+1), np.nan)
    # cache = sr.lil_matrix(((n+1)**2, (n+1)**2))
    for nc in range(0, n + 1):  # This can be (1, n+1)
        for ic in range(0, nc + 1):
            # vectorized over ip
            p = hypergeom.pmf(ic, n, np.arange(0, n+1), nc)
            for io in range(0, n + 1):
                mtx[:, io] += Q(io, n, ic, nc, N, cache, n) * p
    return mtx, cache

def matrix_selection(n, N, s=0):
    mtx = np.zeros((n + 1, n + 1))
    # cache = [np.full((n+1, n+1, n+1), np.nan) for i in range(0, n+1)]
    cache = np.full((n+1, n+1, n+1, n+1), np.nan)
    for nc in range(0, n + 1):  # This can be (1, n+1)
        for ic in range(0, nc + 1):
            # vectorized over ip
            p = hypergeom.pmf(ic, n, np.arange(0, n+1), nc)
            for io in range(0, n + 1):
                mtx[:, io] += Qs(io, n, ic, nc, N, s, cache) * p
    return mtx, cache



Q_new, cache = matrix_tangle(5, 100)
# Q_old = matrix_dumb(5, 100)
# Q_new.sum(axis=1), np.allclose(Q_new - Q_old, 0), np.sum(np.isfinite(cache)) / np.prod(cache.shape)




"""if (io, no, ic, nc) == (1, 1, 1, 1):
        v = 1 - s
    elif (io, no, ic, nc) == (0, 1, 0, 1):
        v = 1
    elif (io, no, ic, nc) == (2, 2, 1, 1):
        v = s * (1 - s)
    elif (io, no, ic, nc) == (2, 2, 2, 2):
        v = (1-s)**2
    elif (io, no, ic, nc) == (1, 2, 0, 1):
        v = s/2
    elif (io, no, ic, nc) == (1, 2, 1, 2):
        v = 1 - s
    elif (io, no, ic, nc) == (0, 2, 0, 2):
        v = 1
    elif (io, no, ic, nc) == (1, 2, 1, 2):
        v = (1 - s)
    elif (no < 2) or (nc < 2) or (io < 0) or (ic < 0) or (io > no) or (ic > nc) or (no < nc) or (io < ic):
        v = 0"""

"Done"
