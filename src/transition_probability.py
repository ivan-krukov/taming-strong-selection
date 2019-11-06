import numpy as np
from scipy.special import factorial, binom


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


def transition_probability(n, N, k, d, restrict=None):
    """
    Calculate the transition probability in a finite sample of size `n` from `k` to `k+d` derived alleles
    `N` is the effective population size.
    `restrict` limits the calculation to events of that order: `2` corresponds to coalescence of two lineages.
    By default, all transitions are included"""

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


def coalescent_matrix(n, N, restrict=None):
    """
    Construct a transition probability matrix of size `n`, with population size `N`
    """
    Q = np.zeros((n + 2, n + 2))
    for i in range(0, n + 2):
        for j in range(0, n + 2):
            d = j - i
            Q[i, j] = transition_probability(n + 1, 1000, i, d, 2)
    return Q
