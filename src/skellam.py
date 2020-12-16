"""Skellam approximation figure"""

import numpy as np
from scipy.stats import skellam
import matplotlib.pyplot as plt
import seaborn as sns
from occupancy import dist_num_anc, reduced_occupancy

plot_rc = {"legend.title_fontsize": 16}
sns.reset_defaults()
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2.0, rc=plot_rc)

rocc = reduced_occupancy(1000)
N = 1000
n = 20
x = 1

fig, ax = plt.subplots(figsize=(6, 6))
rmin, rmax = max(1, n - 10), n + 10

for i, Ns in enumerate([1 / 1000, 5, 10, 50]):
    s = Ns / N
    n_range = np.arange(rmin, rmax + 1)
    # Exact distribution
    dist = [dist_num_anc(a, n, x, s, N, rocc) for a in n_range]
    kwargs = dict(n=n, N=N, s=Ns / N)
    ax.plot(n_range, dist, ls="", marker="o", color=f"C{i}")

    # Skellam
    mu1 = n * s
    mu2 = n * (n - 1) / (2 * N)
    s = skellam(mu1, mu2)
    s_range = np.arange(-n, rmax)
    # ATTN: note that +n is added to the support of the distribution here
    ax.plot(s_range + n, s.pmf(s_range), label=Ns, ls="--", color=f"C{i}")

ax.legend(title="Ns", loc="upper left")

shared_args = dict(
    xlabel="Number of contributing lineages",
    xlim=(rmin, rmax),
    ylim=(-0.005, 1),
    xticks=list(range(rmin, rmax, 5)),
)
ax.set(ylabel="Probability", title="Skellam approximation", **shared_args)
ax.fill_between([n, rmax], [-10, -10], [10, 10], color="red", alpha=0.05)

fig.tight_layout()
fig.savefig("fig/skellam.pdf")
