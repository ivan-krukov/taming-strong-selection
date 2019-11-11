from transition_probability import coalescent_matrix, kingman_simple
from allele_age import allele_age, diffusion_allele_age
from wright_fisher import wright_fisher

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

N = 100
n = 20
r = np.arange(1, n)
wf_r = (r/n*N).astype(int)

print("Building full TPM...")
Q_full = coalescent_matrix(n, N)
a_full = allele_age(Q_full[1:-1, 1:-1], 1)

print("Building 2-way TPM...")
Q_2way = coalescent_matrix(n, N, 2)
Q_2way /= Q_2way.sum(axis=1)
a_2way = allele_age(Q_2way[1:-1, 1:-1], 1)

print("Building WF TPM...")
Q_wfsh = wright_fisher(N)
a_wfsh = allele_age(Q_wfsh[1:-1, 1:-1], int(1/n*N))

print("Bulding continuous approximate TPM...")
#TODO: this does not work as expected
# we expect this to approach Q_full with thinner slices
slce_f = 1000
Q_slce = Q_2way.copy()
s_diag = Q_slce.diagonal()
Q_slce[range(n + 1), range(n + 1)] = 0
Q_slce /= slce_f
Q_slce[range(n + 1), range(n + 1)] = 1 - Q_slce.sum(axis=1)
a_slce = allele_age(Q_slce[1:-1, 1:-1], 1)

print("Building restricted TPM...")
Q_king = kingman_simple(n, N)/50
Q_king /= Q_king.sum(axis=1)
a_king = allele_age(Q_king[1:-1, 1:-1], 1)

a_diff = diffusion_allele_age(r/n)


with sns.plotting_context("paper", font_scale=2), sns.axes_style("whitegrid"), sns.color_palette("Set2"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
    ax.plot(r, a_full[0], label=f"Full, n={n}, N={N}", ls="", marker=7)
    ax.plot(r, a_2way[0], label=f"2-way, n={n}", ls="", marker=7)
    ax.plot(r[1:-1], a_diff[1:-1] * N, label="Diffusion approximation")
    ax.plot(r, a_slce[0]/slce_f, label="2-way, slices", marker=6, ls="")
    ax.plot(wf_r/N*n, a_wfsh[0][wf_r], label=f"Wright-Fisher, N={N}")
    ax.plot(r, a_king[0], label=f"Single transiton only, n={n}")
    ax.set(xlabel="Observed allele frequency", ylabel="Allele age")
    ax.legend()
    # edit x axis to perxent format
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:,.0%}'.format(x/n) for x in vals])
    # plot
    fig.suptitle("Allele age as a function of observed copies")
    fig.savefig("fig/fig-2.png")
