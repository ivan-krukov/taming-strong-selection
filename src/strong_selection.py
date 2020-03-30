import numpy as np
import matplotlib.pyplot as plt
from transition_probability_selection import matrix_selection
import moments
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy.linalg as la
from scipy.special import binom as choose
from scipy import integrate
import seaborn as sns
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
        y = 1-x
        return (theta / (x*y)) * ((np.exp(alpha*y) - 1) / (np.exp(alpha) - 1))

# binomial projection
def projection_fun(x, i, n, N, s=0, u=1e-8):
    # this can likely be done better in log space
    return choose(n, i) * np.power(x, i) * np.power(1-x, n-i) * afs_inf_sites(x, N, s, u)


def binomial_projection_full(n, N, s=0, u=1e-8):
    integ = np.zeros(n-1)
    for i in range(1, n):
        z = integrate.quad(projection_fun, 0, 1, args=(i, n, N, s, u))
        integ[i-1] = z[0]
    return integ


tmp_store = Path.cwd() / Path("../data")

N_range = [200, 2000]
n = 200
mu = 1e-8
z = np.zeros(n-1)
z[0] = n * mu                   # Forward mutation
# z[-1] = n * mu                  # Backward mutation
I = np.eye(n-1)


ns_range = [0, 50]


mtxs = [[] for _ in N_range]
frequency_spectra = [[] for _ in N_range]
for i, N in enumerate(tqdm(N_range)):
    for j, Ns in enumerate(tqdm(ns_range)):
        mtx_pkl = tmp_store / Path(f"mtx_n_{n}_Ns_{Ns}_N_{N}.pypkl")
        if not mtx_pkl.exists():
            M, _ = matrix_selection(n, N, Ns/N)
            with open(mtx_pkl, "wb") as pkl:
                pickle.dump(M, pkl)
        else:
            with open(mtx_pkl, "rb") as pkl:
                M = pickle.load(pkl)
        mtxs[i].append(M)

        # solve for equilibrium
        pi = la.solve((M[1:-1,1:-1]-I).T, -z)
        frequency_spectra[i].append(pi)


def moments_fs(n, N, s):
    fs = moments.LinearSystem_1D.steady_state_1D(n, gamma=N*s)
    fs = moments.Spectrum(fs)
    return fs[1:-1]

def normalize(x):
    return np.array(x) / sum(x)

def wright_fisher_sfs(N, s, mu=0):
    w = wright_fisher_haploid(N, s)
    I = np.eye(N-1)
    z = np.zeros(N-1)
    z[0] = (N) * mu
    pi = la.solve((w[1:-1,1:-1]-I).T, -z)
    return pi


# plt.semilogy(moments_fs(n, N, -ns_range[-1]/N), label="Moments")
fig_store = Path.cwd() / Path("../fig")

with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

    N = N_range[-1]
    s = ns_range[-1] / N

    ax[0].semilogy(normalize(frequency_spectra[-1][-1]), label="Numeric")
    ax[0].semilogy(normalize(moments_fs(n, N, -s)), ls="--", label="Moments")
    ax[0].semilogy(normalize(binomial_projection_full(n, N, s)), ls=":", label="Diffusion")
    ax[0].set(title=f"n={n}, N={N}, Ns={ns_range[-1]}")
    ax[0].set(xlim=(-1,50), ylim=(1e-15, 2))


    N = N_range[0]
    s = ns_range[-1] / N

    ax[1].semilogy(normalize(frequency_spectra[0][-1]), label="Numeric (this study)")
    ax[1].semilogy(normalize(moments_fs(n, N, -s)), ls="--", label="Moments")
    
    ax[1].semilogy(normalize(binomial_projection_full(n, N, s)), ls=":", label="Diffusion")
    ax[1].semilogy(normalize(wright_fisher_sfs(N, -s, mu)), ls="--", label="Full Wright-Fisher")
    ax[1].set(title=f"n={n}, N={N}, Ns={ns_range[-1]}")

    ax[1].set(xlim=(-1,50), ylim=(1e-15, 2))
    ax[1].legend(frameon=False)

    fig.tight_layout()

    fig.savefig(fig_store / Path("strong_selection.pdf"), dpi=300)
