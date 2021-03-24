import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
import pickle
from indemmar import plot_and_legend
from spectra import *
import seaborn as sns

from argparse import ArgumentParser


def relative_error(values, truth):
    return (values - truth) / truth


def normalize(x):
    return np.array(x) / sum(x)


if __name__ == "__main__":
    tmp_store = Path("data")

    parser = ArgumentParser()

    parser.add_argument("--mu", type=float, default=1e-8)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--jackknife", "-j", default=5, type=int)
    parser.add_argument(
        "--N-range", nargs="+", action="extend", type=int, required=True
    )
    parser.add_argument(
        "--ns-range", nargs="+", action="extend", type=int, required=True
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    n = args.n
    mu = args.mu
    j = args.jackknife
    z = np.zeros(n - 1)
    z[0] = n * mu  # Forward mutation
    I = np.eye(n - 1)

    J = moments.Jackknife.calcJK13(n + 1)
    frequency_spectra = {N: {ns: None for ns in args.ns_range} for N in args.N_range}
    for i, N in enumerate(tqdm(args.N_range)):
        for j, Ns in enumerate(tqdm(args.ns_range)):
            mtx_store = tmp_store / Path(f"q_mat_{N}_{Ns}_{n}_3_{j}.txt")
            M = np.loadtxt(mtx_store)
            # solve for equilibrium
            tmp = (M[1:-1, 1:-1] - I).T
            pi = la.solve(tmp, -z)
            frequency_spectra[N][Ns] = pi

    plot_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.8)
    with plot_and_legend(
        fname=args.output,
        ncol=len(args.N_range),
        nrow=len(args.ns_range),
        figsize=(5 * len(args.N_range), 3 * len(args.ns_range)),
        legend_side="bottom",
        legend_ncol=3,
    ) as (fig, ax):

        for i, N in enumerate(args.N_range):
            for j, Ns in enumerate(args.ns_range):
                s = Ns / N
                a = ax[j][i]

                wright_fisher = wright_fisher_sfs(N, -s, mu)
                H = hypergeom_projection_mtx(N, n)[1:-1, 1:-1]
                wf_n = wright_fisher @ H
                large_v = wf_n > 1e-12

                numeric = relative_error(frequency_spectra[N][Ns], wf_n)
                assert len(numeric) == n - 1
                moments_solution = relative_error(moments_fs(n, N, -s, mu), wf_n)
                assert len(moments_solution) == n - 1
                diffusion = relative_error(binomial_projection_full(n, N, s), wf_n)
                assert len(diffusion) == n - 1

                plot_range = np.arange(1, n)

                plt_kw = dict(ls="", marker=".", markersize=5)
                a.plot(
                    plot_range,
                    ma.masked_array(numeric, ~large_v),
                    label="This study",
                    **plt_kw,
                )
                a.plot(
                    plot_range,
                    ma.masked_array(moments_solution, ~large_v),
                    label="Moments",
                    **plt_kw,
                )
                a.plot(
                    plot_range,
                    ma.masked_array(diffusion, ~large_v),
                    label="Diffusion approximation",
                )

                a.set(title=f"n={n}, N={N}, Ns={Ns}")
                idx = (j * len(args.N_range)) + i
                a.text(
                    -0.05,
                    1.05,
                    plot_letters[idx],
                    fontweight="bold",
                    transform=a.transAxes,
                )
                a.ticklabel_format(style="plain", useOffset=False)

        ax[0][0].set(ylabel="Relative error")
        ax[1][0].set(ylabel="Relative error", xlabel="Allele count")
        ax[1][1].set(xlabel="Allele count")
        fig.suptitle("Relative error to the exact Wright-Fisher AFS")
