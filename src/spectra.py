import numpy as np
import numpy.ma as ma
import moments

import numpy.linalg as la
from scipy.special import binom as choose
from scipy import integrate
from scipy.stats import hypergeom
from wright_fisher import wright_fisher_haploid

@np.vectorize
def afs_inf_sites(x, N, s=0, u=1e-8):
    """Equations 9.18 and 9.23 in Ewens, 2004
    hacky haploid version
    This assumes a diploid model"""
    theta = 2 * N * u
    if s == 0:
        return theta / x
    else:
        alpha = 2 * N * s
        y = 1 - x
        return (theta / (x * y)) * ((np.exp(alpha * y) - 1) / (np.exp(alpha) - 1))


# binomial projection
def projection_fun(x, i, n, N, s=0, u=1e-8):
    # this can likely be done better in log space
    return (
        choose(n, i)
        * np.power(x, i)
        * np.power(1 - x, n - i)
        * afs_inf_sites(x, N, s, u)
    )


def binomial_projection_full(n, N, s=0, u=1e-8):
    integ = np.zeros(n - 1)
    for i in range(1, n):
        z = integrate.quad(projection_fun, 0, 1, args=(i, n, N, s, u), epsabs=1e-12)
        integ[i - 1] = z[0]
    return integ



def moments_fs(n, N, s, mu):
    # Note that we want the haploid model, so we only take a half
    pi = moments.LinearSystem_1D.steady_state_1D(n, N=1, gamma=N*s, theta=4*N*mu)
    fs = moments.Spectrum(pi)
    return fs[1:-1]/2


def wright_fisher_sfs(N, s, mu=0):
    w = wright_fisher_haploid(N, s)
    I = np.eye(N - 1)
    z = np.zeros(N - 1)
    z[0] = (N) * mu
    pi = la.solve((w[1:-1, 1:-1] - I).T, -z)
    return pi

def hypergeom_projection_mtx(N, n):
    rn = np.arange(0, n+1)
    rN = np.arange(0, N+1)
    return np.array([hypergeom(N, i, n).pmf(rn) for i in rN])
