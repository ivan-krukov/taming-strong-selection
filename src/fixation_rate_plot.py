import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

df = pd.read_csv(sys.argv[1])
N = int(sys.argv[2])

out_file = sys.argv[3]

genome_size = 3e9
mutation_rate = 1e-8

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.8)

fig, ax = plt.subplots(ncols=3, figsize=(15, 6))

df["diffusion_load"] = (
    df.diffusion * df.dfe * N * genome_size * mutation_rate * (-df.selection)
)
df["wright_fisher_load"] = (
    df.wright_fisher * df.dfe * N * genome_size * mutation_rate * (-df.selection)
)

df = df[df.selection > -10]

ax[0].plot(df.selection, df.diffusion, label="Diffusion")
ax[0].plot(df.selection, df.wright_fisher, label="Wright-Fisher")
ax[0].set_title("Probability of fixation \n of a single variant in the genome")
ax[0].text(-0.1, 1.1, "A", fontweight="bold", transform=ax[0].transAxes)
ax[0].legend()

ax[1].plot(df.selection, df.diffusion_load, label="Diffusion")
ax[1].plot(df.selection, df.wright_fisher_load, label="Wright-Fisher")
# ax[1].axhline(1 / N, color="red", ls="--")
ax[1].set_title("Fixation load")
ax[1].text(-0.1, 1.1, "B", fontweight="bold", transform=ax[1].transAxes)
# ax[1].legend()

ax[2].plot(df.selection, (df.diffusion - df.wright_fisher) / df.wright_fisher)
# ax[2].axhline(1 / N, color="red", ls="--")
ax[2].set_title("Relative error of fixation probabilities")
ax[2].text(-0.1, 1.1, "C", fontweight="bold", transform=ax[2].transAxes)

# fig.suptitle(
#     f"Rate of fixation in diffusion vs. Wright-Fisher, N={N}. "
#     # + "Red line - 1 fixation per genome per N generations"
# )
fig.savefig(out_file, dpi=300)
