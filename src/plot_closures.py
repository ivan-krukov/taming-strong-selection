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
N = 1000

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

with plot_and_legend(fname=fig_store / Path('missing.pdf'), legend_title="Ns") as (fig, ax):
    for Ns in Ns_range:
        missing_p = []
        for n in n_range:
            with open(data_store / Path(f"mtx_n_{n}_Ns_{Ns}_N_{N}.pypkl"), "rb") as pkl:
                M = pickle.load(pkl)
                missing_p.append(1 - (M[-1,:].sum()))
        ax.semilogy(n_range, missing_p, label=Ns, ls="", marker="o")
        
