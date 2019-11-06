from src.transition_probability import coalescent_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


Q_full = coalescent_matrix(20, 1_000)
Q_2way = coalescent_matrix(20, 1_000, 2)
Q_2way.sum(axis=1)

with sns.plotting_context("paper", font_scale=2):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    ax[0].matshow(np.log(Q_full))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set(title="Full transition probability matrix")

    ax[1].matshow(np.log(Q_2way))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set(title="2-way coalescence only")

    fig.suptitle("Transition probability matrices of finite size")

    fig.savefig("fig/fig-1.png")
