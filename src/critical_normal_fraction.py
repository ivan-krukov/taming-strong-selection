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
    lmb = 10 # lambda
    Ns = np.arange(10, 100)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc=plot_rc)
    fig, ax = plt.subplots(figsize=(9,6))

    for i, N in enumerate([1_000, 2_500, 5_000, 10_000]):
        nstar = np.array([fsolve(solve_critical, N/20, (x/N, N, 0.99)) for x in Ns])
        mask = (nstar ** 2) / (2 * N) <= lmb
        negmask = ~mask
        # Induce overlap, for plotting
        first_neg = np.where(negmask)[0][0]
        mask[first_neg-1] = False
        
        right = masked_array(nstar, mask)
        left  = masked_array(nstar, negmask)
        ax.plot(Ns, right / N, label=f"{N}", linewidth=3, color=f"C{i}")
        ax.plot(Ns, left / N, linewidth=1, color=f"C{i}", ls='--')

    # plot were the normal approximation breaks down
    N_range = np.geomspace(500, 50_000, 50)
    # s_crit = np.array([fsolve(solve_boundary, 0.01, (N, lmb)) for N in N_range]).reshape(-1)
    n_crit = np.sqrt(2 * lmb * N_range)
    # ax.plot(s_crit * N_range, n_crit / N_range, color="k", ls="--")
    
    ax.legend(title="N")

    ax.set(xlabel=r"$Ns$", ylabel=r"$n_c / N$",
        title="Fraction of the population to be closed under normal approximation")
    fig.savefig("fig/critical_normal_fraction.pdf")

