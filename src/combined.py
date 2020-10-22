import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.ma import masked_array
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

lmb = 10 # lambda
Ns = np.arange(10, 100)
for i, N in enumerate([1_000, 2_500, 5_000, 10_000]):
    nstar = np.array([fsolve(solve_critical, N/20, (x/N, N, 0.99)) for x in Ns])
    mask = (nstar ** 2) / (2 * N) <= lmb
    negmask = ~mask
    # Induce overlap, for plotting
    first_neg = np.where(negmask)[0][0]
    mask[first_neg-1] = False

    right = masked_array(nstar, mask)
    left  = masked_array(nstar, negmask)
    ax[1].plot(Ns, right / N, label=f"{N}", linewidth=3, color=f"C{i}")
    ax[1].plot(Ns, left / N, linewidth=1, color=f"C{i}", ls='--')

ax[1].legend(title="N", loc="upper left")
ax[1].set(xlabel=r"$Ns$", ylabel=r"$n_c / N$")
ax[1].text(-0.05, 1.05, "B", fontweight='bold', transform=ax[1].transAxes)

fig.tight_layout()
fig.savefig("fig/combined.pdf")
