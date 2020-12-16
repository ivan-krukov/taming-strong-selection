#!/usr/bin/env python3
"Testing the different mean equations - 10 and 11 in main text"

import matplotlib.pyplot as plt
import numpy as np


def mean_exact(n, N, s):
    "Exact mean of the lineages lost"
    z = 1 - (1 / N)
    a = N * (z ** n)
    b = ((1 - s) / (1 - (s * z))) ** n
    return N - (a * b) - n


def mean_apx(n, N, s):
    "Approximate mean of the lineages lost"
    return (n * s) - (n * (n - 1) / (2 * N))


N = 10_000
n = 100
s = 10 / N

n_range = np.arange(0, 100)
s_range = np.linspace(0, 10 / N)


fig, ax = plt.subplots(ncols=2)

ax[0].plot(n_range, mean_exact(n_range, N, s), label="exact")
ax[0].plot(n_range, mean_apx(n_range, N, s), label="approximate", ls="--")
ax[0].set_xlabel('sample size')

ax[1].plot(s_range, mean_exact(n, N, s_range))
ax[1].plot(s_range, mean_apx(n, N, s_range), ls="--")
ax[1].set_xlabel('selection')

fig.legend()
plt.show()
