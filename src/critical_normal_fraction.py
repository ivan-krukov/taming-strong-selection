import numpy as np
from numpy.ma import masked_array
from scipy.stats import norm
from scipy.optimize import fsolve, root
from normal_approximation import approx_mean, approx_std_dev
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def critical_sample_size(n, s, N, q=0.99):
    """Required sample size for all lineages to be contained with confidence `q`"""
    kwargs = dict(n=n, N=N, s=s)
    z = norm(approx_mean(**kwargs), approx_std_dev(**kwargs))
    return z.ppf(q)

def solve_critical(n, s, N, q=0.99):
    """Solving for the critical sample size for a given Ns"""
    return 1 - (critical_sample_size(n, s, N, q) / n)

def solve_boundary(s, N, l=10, q=0.99):
    nstar = fsolve(solve_critical, N/20, (s, N, q))
    return ((nstar**2) / (2 * N)) - l

if __name__ == "__main__":
    plot_rc = {"legend.title_fontsize":14}
    Ns = np.arange(1/2, 50)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc=plot_rc)
    fig, ax = plt.subplots(figsize=(9,6))

    for N in np.array([1_000, 1_500, 2_000, 5_000, 10_000]):
        nstar = np.array([fsolve(solve_critical, N/20, (x/N, N, 0.99)) for x in Ns])
        mst = masked_array(nstar, nstar**2 / (2*N) <= 10)
        ax.plot(Ns, mst / N, label=f"{N}")

    # plot were the normal approximation breaks down
    N_range = np.geomspace(500, 11_000, 50)
    lmb = 10 # lambda
    s_crit = np.array([fsolve(solve_boundary, 0.02, (N, lmb)) for N in N_range]).reshape(-1)
    n_crit = np.sqrt(2 * lmb * N_range)
    ax.plot(s_crit * N_range, n_crit / N_range, color="k", ls="--")
    
    ax.legend(title="N")

    ax.set(xlabel=r"$Ns$", ylabel=r"$n_c / N$",
        title="Fraction of the population to be closed under normal approximation")
    fig.savefig("fig/critical_normal_fraction.pdf")

