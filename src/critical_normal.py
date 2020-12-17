import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from normal_approximation import approx_mean, approx_std_dev
import matplotlib.pyplot as plt
import seaborn as sns

def critical_sample_size(n, s, N, q=0.99):
    """Required sample size for all lineages to be contained with confidence `q`"""
    kwargs = dict(n=n, N=N, s=s)
    z = norm(approx_mean(**kwargs), approx_std_dev(**kwargs))
    return z.ppf(q)

def solve_critical(n, s, N, q=0.95):
    """Solving for the critical sample size for a given Ns"""
    return 1 - (critical_sample_size(n, s, N, q) / n)

if __name__ == "__main__":
    plot_rc = {"legend.title_fontsize":14}
    N = 1000
    Ns = np.arange(1/2, 50)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc=plot_rc)
    fig, ax = plt.subplots(figsize=(9,6))

    for q in [0.999, 0.99, 0.9, 0.5]:
        nstar = [fsolve(solve_critical, 50, (x/N, N, q)) for x in Ns]
        ax.plot(Ns, nstar, label=f"{np.round(q*100, 3)}%")
    ax.legend(title="Confidence")

    ax.set(xlabel="Ns", ylabel="Critical sample size",
        title="Critical sample size required for closure - normal approximation")
    fig.savefig("fig/critical_normal.pdf")

