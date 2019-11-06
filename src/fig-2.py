from src.transition_probability import coalescent_matrix
from src.allele_age import allele_age, diffusion_allele_age

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

N = 1000

Q_full = coalescent_matrix(20, N)
Q_2way = coalescent_matrix(20, N, 2)
Q_2way.sum(axis=1)

a_full = allele_age(Q_full[1:-1, 1:-1], 1)
a_2way = allele_age(Q_2way[1:-1, 1:-1], 1)
a_diff = diffusion_allele_age(np.linspace(0, 1, 19))

with sns.plotting_context("paper", font_scale=2):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    r = np.arange(1, 20)
    ax.plot(r, a_full[0], label="Full TPM")
    ax.plot(r, a_2way[0], label="2-way TPM")
    ax.plot(r[1:-1], a_diff[1:-1] * N, label="Diffusion approximation")

    ax.set(xlabel="Starting number of copies", ylabel="Allele age")
    ax.legend()
    fig.savefig("fig/fig-2.png")
