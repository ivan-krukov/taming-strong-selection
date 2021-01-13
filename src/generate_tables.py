from transition_probability_jackknife import matrix_jackknife
from pathlib import Path
import numpy as np
from tqdm import tqdm

N_range = [1000, 100, 200, 500]
Ns_range = [0, 1, 5, 10, 50, 99]
max_t = 4
n_range = [50, 100]
tmp_store = Path("data")

for n in tqdm(n_range):
    for N in tqdm(N_range):
        for Ns in tqdm(Ns_range):
           s = Ns / N
           print(n, N, Ns)
           mtx_txt = tmp_store / Path(f"mtx_n_{n}_Ns_{Ns}_N_{N}_J_1.txt")
           if not mtx_txt.exists():
               R = matrix_jackknife(n, k=1, N=N, s=s, max_t=max_t)
               np.savetxt(mtx_txt, R)
