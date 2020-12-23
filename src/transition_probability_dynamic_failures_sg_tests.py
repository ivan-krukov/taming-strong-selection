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
            # should this be multiplied by s
            v += (oos * (ic / nc) * s * Pf(io - 1, no - 1, ic - 1, nc - 1, s, N, max_t, max_t, cache_P0, cache_Pf)) + (
                (ic / N) * s * Pf(io - 1, no - 1, ic, nc, s, N, max_t, max_t, cache_P0, cache_Pf)
            )  # Forcing success of cases b and c, respectively

    cache_P0[io, no, ic, nc] = v
    return v


#@jit(nopython=True)
def Pf_smartcache(io, no, ic, nc, s, N, t, max_t, cache_P0_previous, cache_P0_current, cache_Pf_previous, cache_Pf_current):
    """The probability of not having had a transition after t failures"""

    #print("in pf", io, no, ic, nc, t)
    #print(cache_Pf_current.shape)
    if io<0 or no<0 or ic < 0 or nc < 0 or t <0 or io>no or ic>nc :
        v = 0
    else:
        #print("shape in pf", cache_Pf_current.shape)
        v = cache_Pf_current[io, ic, nc, t]

    # print("v in pf", v, "for params ",io, no, ic, nc, s, t)
    if np.isnan(v):
        v = basecases_t(io, no, ic, nc, s, t)
        #print("v in pf", v, "for params ", io, no, ic, nc, s, t)
        if np.isnan(v):

            # t is the number of failures
            if t <= 0:
                # We will build a recursion over the number of successfully drawn offspring n_0.
                # We need to start our recursion on
                v = P0_smartcache(io, no, ic, nc, s, N, max_t, cache_P0_previous, cache_P0_current,
                                  cache_Pf_previous, cache_Pf_current)

            else:  # the curent number of failures was obtained from a previous number of failures
                oos = 1 - ((nc - 1) / N)  # out-of-sample
                # B and C correspond to the cases in P0
                b = oos * (ic / nc) * s * Pf_smartcache(io, no, ic - 1, nc - 1, s, N, t - 1, max_t, cache_P0_previous,
                                                        cache_P0_current, cache_Pf_previous, cache_Pf_current)

                c = (ic / N) * s * Pf_smartcache(io, no, ic, nc, s, N, t - 1, max_t, cache_P0_previous, cache_P0_current,
                                      cache_Pf_previous, cache_Pf_current)
                v = b + c


            cache_Pf_current[io, ic, nc, t] = v

    return v


#@jit(nopython=True)
def P0_smartcache(io, no, ic, nc, s, N, max_t, cache_P0_previous, cache_P0_current,
                                                                cache_Pf_previous, cache_Pf_current):
    """max_t is the maximum number of failures. """
    assert (len(cache_P0_current.shape)==3 and
            len(cache_Pf_current.shape)==4), "current cache sizes incorrect"
    # print("in P0", io, no, ic, nc, cache_P0_current.shape )
    v = cache_P0_current[io, ic, nc]
    #if cache_Pf_previous is not np.nan:
    #    print("shape in P0", cache_Pf_previous.shape )
    if np.isnan(v):  # Recursion on what happened in the last lineage.
        v = basecases_t(io, no, ic, nc, s, t=max_t)
        if np.isnan(v):
            oos = 1 - ((nc - 1) / N)  # out-of-sample
            sel = 1 - s

            # out of sample, ancestral
            af = 0
            for t in range(max_t, -1, -1):
                af += Pf_smartcache(io, no - 1, ic, nc - 1, s, N, t, max_t, np.nan, cache_P0_previous, np.nan,
                                     cache_Pf_previous)
            a = oos * ((nc - ic) / nc) * af

            # out of sample, derived
            bf = 0
            for t in range(max_t, -1, -1):
                bf += Pf_smartcache(io - 1, no - 1, ic - 1, nc - 1, s, N, t, max_t, np.nan, cache_P0_previous, np.nan,
                                    cache_Pf_previous)
            b = oos * (ic / nc) * sel * bf

            # in sample, derived
            cf = 0
            for t in range(max_t, -1, -1):
                cf += Pf_smartcache(io - 1, no - 1, ic, nc, s, N, t, max_t, np.nan, cache_P0_previous, np.nan,
                                    cache_Pf_previous)
            c = (ic / N) * sel * cf

            # in sample, ancestral
            df = 0
            for t in range(max_t, -1, -1):
                df += Pf_smartcache(io, no - 1, ic, nc, s, N, t, max_t, np.nan, cache_P0_previous, np.nan,
                                    cache_Pf_previous)
            d = ((nc - ic) / N) * df

            v = a + b + c + d
            # should this be multiplied by s
            v += (oos * (ic / nc) * s * Pf_smartcache(io - 1, no - 1, ic - 1, nc - 1, s, N, max_t, max_t, np.nan,
                                                      cache_P0_previous, np.nan, cache_Pf_previous)) + (
                    (ic / N) * s * Pf_smartcache(io - 1, no - 1, ic, nc, s, N, max_t, max_t, np.nan,
                                                 cache_P0_previous, np.nan, cache_Pf_previous)
            )  # Forcing success of cases b and c, respectively

            cache_P0_current[io, ic, nc] = v
    return v # This does not need to return!



def matrix(no=5, s=1 / 1000, N=1000, max_t=1):
    mtx = np.zeros((no + 1, no + 1)) # mtx = Q in text (i_o,i_p)
    cache_P0 = np.full((no+1, no+1, no+1, no+1), np.nan) # p0 = T_0 in text; arguments (i_o,n_o,i_p,n_p)
    cache_Pf = np.full((no+1, no+1, no+1, no+1, max_t+1), np.nan) # pf = T_r (i_o,n_o,i_p,n_p,r)
    for nc in range(0, no + 1): # n_c is n_p in text
        for ic in range(0, nc + 1):
            # vectorized over ip
            p = hypergeom.pmf(ic, no, np.arange(0, no + 1), nc)
            for io in range(0, no + 1):
                mtx[:, io] += P0(io, no, ic, nc, s, N, max_t, cache_P0, cache_Pf) * p
    return mtx, cache_P0, cache_Pf


def matrix_smart_cache(no=5, s=1 / 1000, N=1000, max_t=1):
    mtx = np.zeros((no + 1, no + 1)) # mtx = Q in text (i_o,i_p)
    #cache_P0 = np.full((no+1, no+1, no+1, no+1), np.nan) # p0 = T_0 in text; arguments (i_o,n_o,i_p,n_p)
    #cache_Pf = np.full((no+1, no+1, no+1, no+1, max_t+1), np.nan) # pf = T_r (i_o,n_o,i_p,n_p,r)
    cache_P0_previous = np.full((1, no+1, no+1),np.nan) # arguments (i_o,i_p,n_p)
    cache_P0_current = np.full((2, no+1, no+1), np.nan)
    cache_Pf_previous = np.full((1, no+1, no+1, max_t+1),np.nan) # n_o=0; arguments (i_o,i_p,n_p, r)
    cache_Pf_current = np.full((2, no+1, no+1, max_t+1 ), np.nan) # n_o=1
    for nop in range(1, no+1):
        for npp in range(1,no+1): # This might be too much looping
            for iop in range(0,nop+1):
                for ipp in range(0, npp + 1):
                    print("nop,npp,iop,ipp, v", nop,npp,iop,ipp,
                    P0_smartcache(iop, nop, ipp, npp, s, N, max_t, cache_P0_previous, cache_P0_current,
                                                                cache_Pf_previous, cache_Pf_current))

        #clear cache for nop'==nop-1 that we wil no longer need
        cache_P0_previous = cache_P0_current
        cache_Pf_previous = cache_Pf_current
        cache_P0_current = np.full((nop+2,no+2,no+2),np.nan)
        cache_Pf_current = np.full((nop+2,no+2,no+2, max_t+1),np.nan) # +2 because we need to account for the increment
        # in the loop.

    #for nc in range(0, no + 1): # n_c is n_p in text
    #    for ic in range(0, nc + 1):
    #        # vectorized over ip
    #        p = hypergeom.pmf(ic, no, np.arange(0, no + 1), nc)
    #        for io in range(0, no + 1):
    #            mtx[:, io] += P0(io, no, ic, nc, s, N, max_t, cache_P0, cache_Pf) * p
    #return mtx, cache_P0, cache_Pf
    return cache_P0_previous