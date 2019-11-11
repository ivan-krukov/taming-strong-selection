import numpy as np
from scipy.special import factorial, binom
from functools import lru_cache


def rule_asc_rest(n, m=None):
    """
    Generate the partitions of integer `n`.
    If `m` is provided, generate the partitions with elements no greater that `m`.
    Author: Jerome Kelleher"""
    a = [0 for i in range(n + 1)]
    k = 1
    a[1] = n
    while k != 0:
        x = a[k - 1] + 1
        y = a[k] - 1
        k -= 1
        while (x <= y):
            a[k] = x
            y -= x
            k += 1
        a[k] = x + y
        if (m is None) or (x + y <= m):
            yield a[:k + 1]


def doublet_partitions(n):
    a = n
    b = 0
    while a >= 0:
        yield ([1] * a) + ([2] * b)
        a -= 2
        b += 1

def doublet_partitions_rest(n, m = None):
    if m is None:
        m = (n + 1) // 2
    a = n
    b = 0
    while (a >= 0) and (b < m):
        yield ([1] * a) + ([2] * b)
        a -= 2
        b += 1

def multinomial(params):
    """
    Implementation by Reiner Martin
    https://stackoverflow.com/questions/46374185/does-python-have-a-function-which-computes-multinomial-coefficients
    """
    res, i = 1, 1
    for a in params:
        for j in range(1, a + 1):
            res *= i
            res //= j
            i += 1
    return res


def transition_probability(n, N, k, d, restrict=None, kingman=False):
    """
    Calculate the transition probability in a finite sample of size `n` from `k` to `k+d` derived alleles
    `N` is the effective population size.
    `restrict` limits the calculation to events of that order: `2` corresponds to coalescence of two lineages.
    By default, all transitions are included"""

    if kingman:
        P = list(doublet_partitions_rest(k + d, 2))
        Q = list(doublet_partitions_rest(n - k - d, 2))
    else:
        P = list(rule_asc_rest(k + d, restrict)) if (k + d) > 0 else [[]]
        Q = list(rule_asc_rest(n - k - d, restrict)) if (n - k - d) > 0 else [[]]

    s = 0

    for p in P:
        for q in Q:
            a = n - len(p) - len(q)
            b = n - k - len(q)
            if (a >= 0) and (b >= 0):
                coal = (np.power(1 / N, a) *
                        np.product((N - np.arange(len(p) + len(q))) / N))  # coalescence term
                mult = ((multinomial(p) / np.product(factorial(np.bincount(p)))) *
                        (multinomial(q) / np.product(factorial(np.bincount(q)))))  # Multiplicity
                evnt = binom(a, b) / binom(n, k)  # Pr(event | partitions)
                s += coal * mult * evnt

    return binom(n, k + d) * s


@lru_cache(maxsize=20)
def coalescent_matrix(n, N, restrict=None, kingman=False):
    """
    Construct a transition probability matrix of size `n`, with population size `N`
    """
    Q = np.zeros((n + 1, n + 1))
    for i in range(0, n + 1):
        for j in range(0, n + 1):
            d = j - i
            Q[i, j] = transition_probability(n, N, i, d, restrict, kingman)
    return Q


def kingman_simple(n, N):
    Q = np.zeros((n + 1, n + 1))
    const = np.product((N - np.arange(1, n - 1)) / N)
    for i in range(0, n + 1):
        if i - 1 >= 0:
            Q[i, i-1] = i * (n - i) * const / (2 * N)
        Q[i, i] = const * (1 - (n - 1)/N)
        if i + 1 <= n:
            Q[i, i+1] = i * (n - i) * const / (2 * N)
    return Q
        
