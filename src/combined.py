import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from indemmar import plot_and_legend
from occupancy import dist_num_anc, reduced_occupancy
from normal_approximation import approx_mean, approx_std_dev
from critical_normal import critical_sample_size, solve_critical

plot_rc = {"legend.title_fontsize":16}
sns.reset_defaults()
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2.0, rc=plot_rc)

rocc = reduced_occupancy(1000)
N = 1000
n = 200
x = 1

fig, ax = plt.subplots(ncols=2, figsize=(15,6))
rmin, rmax = n-40, n+5

for i,Ns in enumerate([1 / 1000, 5, 10, 50]):
    s = Ns / N
    n_range = np.arange(rmin, rmax+1)
    dist = [dist_num_anc(a, n, x, s, N, rocc) for a in n_range]
    kwargs = dict(n=n, N=N, s=Ns/N)
    z = norm(approx_mean(**kwargs), approx_std_dev(**kwargs))
    ax[0].plot(n_range, dist, ls="", marker="o", color=f"C{i}")
    ax[0].plot(n_range, z.pdf(n_range), label=Ns, color=f"C{i}")

ax[0].legend(title="Ns", loc="upper left")

shared_args = dict(
    xlabel="Number of contributing lineages",
    xlim=(rmin, rmax),
    ylim=(-0.005, 0.12),
    xticks=list(range(rmin, rmax, 5)),
)
ax[0].set(ylabel="Probability", **shared_args)
ax[0].text(-0.05, 1.05, "A", fontweight='bold', transform=ax[0].transAxes)
ax[0].fill_between([n, rmax], [-10, -10], [10, 10], color="red", alpha=0.05)

Ns_range = np.arange(1/2, 50)
for i, q in enumerate([0.99, 0.95, 0.9, 0.5]):
    nstar = [fsolve(solve_critical, 50, (Ns/N, N, q)) for Ns in Ns_range]
    ax[1].plot(Ns_range, nstar, label=f"{int(q*100)}%", color="C0", alpha=q)

ax[1].set(xlabel="Ns", ylabel="Critical sample size")
ax[1].legend(title="Confidence", loc="upper left")
ax[1].text(-0.05, 1.05, "B", fontweight='bold', transform=ax[1].transAxes)

fig.tight_layout()
fig.savefig("fig/combined.pdf")
