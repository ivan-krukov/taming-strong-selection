import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from indemmar import plot_and_legend
from occupancy import dist_num_anc, reduced_occupancy

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

rocc = reduced_occupancy(1000)
N = 1000
n = 20
x = 1

with plot_and_legend(
    fname="fig/distributions.pdf", legend_title="Ns", ncol=2, figsize=(12, 6)
) as (fig, ax):

    for Ns in [1 / 1000, 1, 10, 50]:
        s = Ns / N
        n_range = np.arange(10, 31)
        dist = np.zeros(21)
        cuml = np.zeros(21)
        for (i, a) in enumerate(n_range):
            y = dist_num_anc(a, n, x, s, N, rocc)
            if i > 0:
                cuml[i] = cuml[i - 1]

            cuml[i] += y
            dist[i] = y

        ax[0].plot(n_range, dist, ls="", marker="o", label=int(Ns))
        ax[1].plot(n_range, cuml, ls="", marker="o")

    shared_args = dict(
        xlabel="Number of contributing lineages",
        xlim=(10.5, 30.5),
        ylim=(-0.05, 1.05),
        xticks=list(range(10, 31, 2)),
    )
    ax[0].set(ylabel="Probability", **shared_args)
    ax[0].text(-0.05, 1.05, "A", fontweight='bold', transform=ax[0].transAxes)
    ax[1].set(ylabel="Cumulative Probability", **shared_args)
    ax[1].text(-0.05, 1.05, "B", fontweight='bold', transform=ax[1].transAxes)

    for i in range(2):
        ax[i].fill_between([0, 20], [-10, -10], [10, 10], color="red", alpha=0.2)

    # fig.suptitle(f"N={N}")
