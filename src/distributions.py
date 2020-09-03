import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from indemmar import plot_and_legend
from occupancy import dist_num_anc, reduced_occupancy

plot_rc = {"legend.title_fontsize":16}
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2.0, rc=plot_rc)

rocc = reduced_occupancy(1000)
N = 1000
n = 200
x = 1

with plot_and_legend(
    fname="fig/distributions.pdf", legend_title="Ns", ncol=2, figsize=(12, 6)
) as (fig, ax):

    rmin, rmax = n-25, n+5
    for Ns in [1 / 1000, 5, 10, 50]:
        s = Ns / N
        n_range = np.arange(rmin, rmax+1)
        dist = np.zeros(rmax-rmin+1)
        cuml = np.zeros(rmax-rmin+1)
        for (i, a) in enumerate(n_range):
            y = dist_num_anc(a, n, x, s, N, rocc)
            if i > 0:
                cuml[i] = cuml[i - 1]

            cuml[i] += y
            dist[i] = y

        ax[0].plot(n_range, dist, ls="-", marker="o", label=int(Ns))
        ax[1].plot(n_range, cuml, ls="-", marker="o")

    shared_args = dict(
        xlabel="Number of contributing lineages",
        xlim=(rmin, rmax),
        ylim=(-0.05, 1.05),
        xticks=list(range(rmin, rmax, 5)),
    )
    ax[0].set(ylabel="Probability", **shared_args)
    ax[0].text(-0.05, 1.05, "A", fontweight='bold', transform=ax[0].transAxes)
    ax[1].set(ylabel="Cumulative Probability", **shared_args)
    ax[1].text(-0.05, 1.05, "B", fontweight='bold', transform=ax[1].transAxes)

    for i in range(2):
        ax[i].fill_between([n, rmax], [-10, -10], [10, 10], color="red", alpha=0.05)
