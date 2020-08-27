import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

def approx_mean(n, N, s):
    return (s * n)/(1 - s) + (N * (1 - np.power(1-(1/N), n)))

def approx_std_dev(n, N, s):
    return np.sqrt(N*(np.power(1 - 1/N,n) + np.power(1 - 2/N,n)*(N-1) - np.power(1-1/N, 2*n)*N) + (n*s)/np.power(1-s,2))

N = 2000
n = 200
percentiles = [0.5, 0.9, 0.99, 0.999]

quantile= [[] for _ in percentiles]
for i,p in enumerate(percentiles):
    for Ns in range(0, 50):
        kwargs = dict(n=n, N=N, s=Ns/N)
        z = norm(approx_mean(**kwargs), approx_std_dev(**kwargs))
        quantile[i].append(z.ppf(p))

with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots()
    ax.axhline(y=200, color="k", ls=":")
    for i,p in enumerate(percentiles):
        ax.plot(quantile[i], label=p)


    ax.legend(frameon=False, title="Percentile")
    ax.set(xlabel="Ns", ylabel="Sample size")
    fig.tight_layout()
    fig.savefig("quantile.pdf")


