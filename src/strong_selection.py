import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
#from transition_probability_selection import matrix_selection_more_contributors
import moments
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy.linalg as la
from scipy.special import binom as choose
from scipy import integrate
from scipy.stats import hypergeom
from indemmar import plot_and_legend
from wright_fisher import wright_fisher_haploid
import seaborn as sns


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

def hypergeom_projection_mtx(N, n):
    rn = np.arange(0, n+1)
    rN = np.arange(0, N+1)
    return np.array([hypergeom(N, i, n).pmf(rn) for i in rN])

def relative_error(values, truth):
    return (values - truth) / truth

if __name__ == "__main__":
    tmp_store = Path("data")

    N_range = [1000, 500, 100]
    n = 100
    mu = 1e-8
    z = np.zeros(n - 1)
    z[0] = n * mu  # Forward mutation
    I = np.eye(n - 1)

    ns_range = [0, 1, 5, 10, 50, 99]


    J = moments.Jackknife.calcJK13(n+1)
    frequency_spectra = {N: {ns: None for ns in ns_range} for N in N_range}
    for i, N in enumerate(tqdm(N_range)):
        for j, Ns in enumerate(tqdm(ns_range)):
            mtx_store = tmp_store / Path(f"mtx_n_{n}_Ns_{Ns}_N_{N}_J_1.txt")
            M = np.loadtxt(mtx_store)
            # with open(mtx_pkl, "rb") as pkl:
            #     M = pickle.load(pkl)

            # solve for equilibrium
            tmp = (M[1:-1, 1:-1] - I).T
            pi = la.solve(tmp, -z)
            frequency_spectra[N][Ns] = pi

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.8)
    plot_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    with plot_and_legend(
            fname="fig/strong_selection_six_panel.pdf",
            ncol=3,
            nrow=6,
            figsize=(15, 12),
            legend_side="bottom",
            legend_ncol=3
    ) as (fig, ax):

        for i, N in enumerate(N_range):
            for j, Ns in enumerate(ns_range):
                s = Ns / N
                a = ax[j][i]

                wright_fisher = wright_fisher_sfs(N, -s, mu)
                H = hypergeom_projection_mtx(N, n)[1:-1, 1:-1]
                wf_n = normalize(wright_fisher @ H)
                large_v = wf_n > 1e-12


                # numeric = relative_error(normalize(frequency_spectra[N][Ns]), wf_n)
                # assert(len(numeric) == n-1)
                # moments_solution = relative_error(normalize(moments_fs(n, N, -s)), wf_n)
                # assert(len(moments_solution) == n-1)
                # diffusion = relative_error(normalize(binomial_projection_full(n, N, s)), wf_n)
                # assert(len(diffusion) == n-1)
                plot_range = np.arange(1, n)

                plt_kw = dict(ls="", marker=".", markersize=5)
                # a.plot(plot_range, ma.masked_array(numeric, ~large_v), label="This study", **plt_kw)
                # a.plot(plot_range, ma.masked_array(moments_solution, ~large_v), label="Moments", **plt_kw)
                # a.plot(plot_range, ma.masked_array(diffusion, ~large_v), label="Diffusion approximation")
                a.semilogy(plot_range, normalize(frequency_spectra[N][Ns]), label='This study',
                           **plt_kw)
                a.semilogy(plot_range, normalize(moments_fs(n, N, -s)), label='Moments', **plt_kw)
                a.semilogy(plot_range, normalize(binomial_projection_full(n, N, s)), label='Diffusion')
                a.semilogy(plot_range, normalize(wf_n), label='Wright-Fisher')

                a.set(title=f"n={n}, N={N}, Ns={Ns}")
                idx = (j * len(N_range)) + i
                a.text(
                    -0.05, 1.05, plot_letters[idx], fontweight="bold", transform=a.transAxes
                )

        ax[0][0].set(ylabel="Relative error")
        ax[1][0].set(ylabel="Relative error", xlabel="Allele count")
        ax[1][1].set(xlabel="Allele count")
        fig.suptitle("Relative error to the exact Wright-Fisher AFS")
