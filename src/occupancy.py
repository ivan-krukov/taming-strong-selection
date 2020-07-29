import numpy as np
from scipy.special import binom, logsumexp, gammaln

# Occupancy stuff
def reduced_occupancy(N):
    rocc = np.full((N+1,N+1), -np.inf)

    for n in range(1, N+1):
        rocc[1,n] = (1-n) * np.log(n)
    for n in range(2, N+1):
        rocc[n,n] = (n-1) * np.log(1-(1/n)) + rocc[n-1, n-1]

    for i in range(2, N):
        for k in range(2, N-i+2):
            n = k + i - 1
            l1 = np.log(k) - np.log(n-k) + rocc[k, n-1]
            l2 = rocc[k-1, n-1]
            rocc[k,n] = ((n-1) * np.log(1-(1/n))) + np.logaddexp(l1, l2)

    return rocc


def log_occupancy(k, n, M, rocc):
    const = n * (np.log(n) - np.log(M))
    
    i = np.arange(0, k)
    s = np.sum(np.log(n - i) - np.log(M - i))
    return rocc[k, n] + const - s


def log_occupancy_gamma(k, n, M, rocc):
    a = n * (np.log(n) - np.log(M))
    b = gammaln(M+1) - gammaln(M-k+1)
    c = gammaln(n+1) - gammaln(n-k+1)
    return rocc[k, n] + a + b - c
    

def log_occupancy_vec(n, M, rocc):
    occ = np.zeros(n)
    const = n * (np.log(n) - np.log(M))
    s = 0
    for i in range(0, n):
        s += np.log(n-i) - np.log(M-i)
        occ[i] = rocc[i+1, n] + const - s

    return occ


def binomln(n, k):
    return gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)


def dist_num_anc(a, n, x, s, M, rocc):
    """Distribution of the number of ancestors
    a - number of ancestors - random variable
    n - sample size
    x - derived allele frequency
    s - selection coefficient
    M - population size
    rocc - reduced occupancy distribution cache"""
    p = 0
    xs = x * s
    for i in range(a, M + 1):
        
        occ = log_occupancy_gamma(a, i, M, rocc)
        
        sel = binomln(i-1, n-1) + (n * np.log(1-xs)) + ((i-n) * np.log(xs))
        
        if occ < -100:
            break    
        lp = occ + sel
        
        p += np.exp(lp)
        
    return p
