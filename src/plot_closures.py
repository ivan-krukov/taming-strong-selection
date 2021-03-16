import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
from indemmar import plot_and_legend
from pathlib import Path

data_store = Path.cwd() / Path("data")
fig_store = Path.cwd() / Path("fig")

matrix_sets = []
Ns_range = [1, 10, 50]
n_range = np.arange(10, 140+5, 5)
k_range = [0, 1, 2, 3]
N = 1000

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

with plot_and_legend(fname=fig_store / Path('missing.pdf'), legend_title="Ns", ncol=len(k_range),
        figsize=(12, 6)) as (fig, ax):
    for Ns in Ns_range:
        for k in k_range:
            missing_p = []
            for n in n_range:
                M = np.loadtxt(data_store / Path(f"q_mat_{N}_{Ns}_{n}_3_{k}.txt"))
                missing_p.append(1 - (M[-1,:].sum()))
            ax[k].semilogy(n_range, missing_p, label=f"{Ns}", ls="", marker="o")
            ax[k].set(title=f"Jackknife order {k}")
            ax[k].set(xlabel="Number of offspring, $n_o$")
    ax[0].set(ylabel="Probability")
