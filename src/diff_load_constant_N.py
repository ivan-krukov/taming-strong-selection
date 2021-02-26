# Here, we are calculating the excess coalescent laod that is bround by the difference in models.
from spectra import *
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    tmp_store = Path("data")
    n_range = [50, 100, 200]
    N = 1000
    mu = 1e-8

    ns_range = [0, 1, 5, 10, 20, 50]

    frequency_spectra = {n: {ns: None for ns in ns_range} for n in n_range}

    for i, n in enumerate(n_range):
        plot_range = np.arange(1, n)
        for j,Ns in enumerate(ns_range):
            # mtx_store = tmp_store / Path(f"mtx_n_{n}_Ns_{Ns}_N_{N}_J_1.txt")
            mtx_store = tmp_store / Path(f"q_mat_{N}_{Ns}_{n}_3_5.txt")
            z = np.zeros(n - 1)
            z[0] = n * mu  # Forward mutation
            I = np.eye(n - 1)
            M = np.loadtxt(mtx_store)
            # solve for equilibrium
            tmp = (M[1:-1, 1:-1] - I).T
            pi = la.solve(tmp, -z)
            frequency_spectra[n][Ns] = pi
            numeric = (pi)
            s = Ns / N

            num_load = numeric @ plot_range * s / n
            diffusion = (binomial_projection_full(n, N, s))
            coal_load = diffusion @ plot_range * s / n
            excess_load = (coal_load - num_load) / num_load * 100

            print(f"${n}$ & ${N}$ & ${Ns}$ & ${num_load:#.3e}$ & ${coal_load:#.3e}$ & ${excess_load:#.3f}\%$ \\\\")
