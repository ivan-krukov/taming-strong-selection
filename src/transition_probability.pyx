# distutils: sources = c/transition_probability.c
# distutils: include_dirs = c/

from libc.stdlib cimport malloc, free
cimport transition_probability
cimport numpy as np
import numpy as np


cpdef build_cache_matrices(int no, int k, int N, double s, int max_t):

    # Allocate space for the intermediates
    cdef Tile** cache = <Tile** > malloc((max_t+1) * sizeof(Tile*))
    for i in range(0, max_t+1):
        cache[i] = tile_new(no+1+k, no+1+k)

    # Invoke the C-side cache builder - heavy lifting
    for nc in range(0, no+1+k):
        for ic in range(0, nc+1):
            for io in  range(0, no+1+k):
                P0(io, no, ic, nc, s, N, max_t, cache)


    # Output list, and copy the results
    cdef list output = []
    cdef Matrix* m
    for nc in range(0, no+1+k):
        out = np.zeros((nc+1, no+1))
        m = cache[0].data[(cache[0].cols * nc) + no]
        # Copy the data
        for i in range(m.rows):
            for j in range(m.cols):
                out[i, j] = m.data[(m.cols * i) + j]
        output.append(out)

    # Deallocate intemediate storage
    for i in range(0, max_t+1):
        tile_del(cache[i])
    free(cache)

    return output


#--------------------------------------------
# Jackknife extrapolations :
# used for the moment closure under selection
# to extrapolate the Phi_(n+1) and Phi_(n+2)
# from the Phi_n.
#--------------------------------------------

def python2round(float f):
    if round(f + 1) - round(f) != 1:
        return f + abs(f) / f * 0.5
    return round(f)

# The choice i' in n samples that best approximates the frequency of \i/(n + 1) is i*n / (n + 1)
cpdef int index_bis(int i, int n):
    return int(min(max(python2round(i * n / float(n+1)), 2), n-2))

# Compute the order 3 Jackknife extrapolation coefficients for 1 jump (Phi_n -> Phi_(n+1))
cpdef np.ndarray[np.float64_t, ndim = 2] calcJK13(int n):
    cdef np.ndarray[np.float64_t, ndim = 2] J = np.zeros((n, n-1))
    cdef int i
    cdef int ibis
    for i in range(n):
        ibis = index_bis(i + 1, n) - 1
        J[i, ibis] = -(1.+n) * ((2.+i)*(2.+n)*(-6.-n+(i+1.)*(3.+n))-2.*(4.+n)*(-1.+(i+1.)*(2.+n))*(ibis+1.)
                  +(12.+7.*n+n**2)*(ibis+1.)**2) / (2.+n) / (3.+n) / (4.+n)
        J[i, ibis - 1] = (1.+n) * (4.+(1.+i)**2*(6.+5.*n+n**2)-(i+1.)*(14.+9.*n+n**2)-(4.+n)*(-5.-n+2.*(i+1.)*(2.+n))*(ibis+1.)
                    +(12.+7.*n+n**2)*(ibis+1.)**2) / (2.+n) / (3.+n) / (4.+n) / 2.
        J[i, ibis + 1] = (1.+n) * ((2.+i)*(2.+n)*(-2.+(i+1.)*(3.+n))-(4.+n)*(1.+n+2.*(i+1.)*(2.+n))*(ibis+1.)
                    +(12.+7.*n+n**2)*(ibis+1.)**2) / (2.+n) / (3.+n) / (4.+n) / 2.
    return J

# Compute the order 3 Jackknife extrapolation coefficients for 2 jumps (Phi_n -> Phi_(n+2))
cpdef np.ndarray[np.float64_t, ndim = 2] calcJK23(int n):
    cdef np.ndarray[np.float64_t, ndim = 2] J = np.zeros((n + 1, n - 1))
    cdef int i
    cdef int ibis
    for i in range(n + 1):
        ibis = index_bis(i + 1, n) - 1
        if i == n - 1 or i == n:
            ibis = n - 3
        J[i, ibis] = -(1.+n) * ((2.+i)*(2.+n)*(-9.-n+(i+1.)*(3.+n))-2.*(5.+n)*(-2.+(i+1.)*(2.+n))*(ibis+1.)
                  +(20.+9.*n+n**2)*(ibis+1.)**2) / (3.+n) / (4.+n) / (5.+n)
        J[i, ibis - 1] = (1.+n) * (12.+(1.+i)**2*(6.+5.*n+n**2)-(i+1.)*(22.+13.*n+n**2)-(5.+n)*(-8.-n+2.*(i+1.)*(2.+n))*(ibis+1.)
                    +(20.+9.*n+n**2)*(ibis+1.)**2) / (3.+n) / (4.+n) / (5.+n) / 2.
        J[i, ibis + 1] = (1.+n) * ((2.+i)*(2.+n)*(-4.+(i+1.)*(3.+n))-(5.+n)*(n+2.*(i+1.)*(2.+n))*(ibis+1.)
                    +(20.+9.*n+n**2)*(ibis+1.)**2) / (3.+n) / (4.+n) / (5.+n) / 2.
    return J
