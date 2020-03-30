import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
from indemmar import plot_and_legend
from pathlib import Path

data_store = Path.cwd() / Path("../data")
fig_store = Path.cwd() / Path("../fig")

matrix_sets = []
for ns in ["1.0", "5.0", "10.0"]:
    with open(data_store / Path(f"mtxs_{ns}.pypkl"), "rb") as pkl:
        matrix_sets.append(pickle.load(pkl))

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

with plot_and_legend(fname=fig_store / Path('missing.pdf'), legend_title="Ns") as (fig, ax):
    for ns, mtxs in zip([1,5,10], matrix_sets):
        closures = []
        for m in mtxs:
            closures.append(1-m[-1,:].sum())
            
        ax.semilogy(np.arange(10, 210, 10), closures, linestyle="", marker="o", label=ns)


    ax.set(xlabel="Sample size", ylabel="Probability of missing lineages", title="N=1000")

        
