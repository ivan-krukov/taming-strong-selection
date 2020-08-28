import numpy as np
import matplotlib.pyplot as plt
#from transition_probability_selection import matrix_selection_more_contributors
import moments
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy.linalg as la
from scipy.special import binom as choose
from scipy import integrate
from indemmar import plot_and_legend
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
        z = integrate.quad(projection_fun, 0, 1, args=(i, n, N, s, u))
        integ[i - 1] = z[0]
    return integ


tmp_store = Path("data")

N_range = [1000, 200]
n = 100
mu = 1e-8
z = np.zeros(n - 1)
z[0] = n * mu  # Forward mutation
# z[-1] = n * mu                  # Backward mutation
I = np.eye(n - 1)

ns_range = [0, 50]


mtxs = [[] for _ in N_range]
frequency_spectra = [[] for _ in N_range]
for i, N in enumerate(tqdm(N_range)):
    for j, Ns in enumerate(tqdm(ns_range)):
        mtx_pkl = tmp_store / Path(f"mtx_n_{n}_Ns_{Ns}_N_{N}.pypkl")
        # if not mtx_pkl.exists():
        #     M, _ = matrix_selection_more_contributors(n, N, Ns / N)
        #     with open(mtx_pkl, "wb") as pkl:
        #         pickle.dump(M, pkl)
        # else:
        with open(mtx_pkl, "rb") as pkl:
            M = pickle.load(pkl)
        mtxs[i].append(M)

        # solve for equilibrium
        pi = la.solve((M[1:-1, 1:-1] - I).T, -z)
        frequency_spectra[i].append(pi)


def moments_fs(n, N, s):
    pi = moments.LinearSystem_1D.steady_state_1D(n, gamma=N * s)
    fs = moments.Spectrum(pi)
    return fs[1:-1]


def normalize(x):
    return np.array(x) / sum(x)


def wright_fisher_sfs(N, s, mu=0):
    w = wright_fisher_haploid(N, s)
    I = np.eye(N - 1)
    z = np.zeros(N - 1)
    z[0] = (N) * mu
    pi = la.solve((w[1:-1, 1:-1] - I).T, -z)
    return pi


plot_letters = list("ABCD")
with plot_and_legend(
    fname="fig/strong_selection_four_panel.pdf",
    legend_title="Model",
    ncol=2,
    nrow=2,
    figsize=(10, 6),
) as (fig, ax):

    for i, N in enumerate(N_range):
        for j, Ns in enumerate(ns_range):
            s = Ns / N
            a = ax[j][i]
            numeric = frequency_spectra[i][j]
            a.semilogy(normalize(numeric), label="Numeric")

            moments_solution = moments_fs(n, N, -s)
            a.semilogy(normalize(moments_solution), label="Moments")

            diffusion = binomial_projection_full(n, N, s)
            a.semilogy(normalize(diffusion), ls="--", label="Diffusion")

            if N == n:
                wright_fisher = wright_fisher_sfs(N, -s, mu)
                a.semilogy(normalize(wright_fisher), ls="--", label="Wright-Fisher")

            a.set(title=f"n={n}, N={N}, Ns={Ns}")
            idx = (j * 2) + i
            a.text(
                -0.05, 1.05, plot_letters[idx], fontweight="bold", transform=a.transAxes
            )
