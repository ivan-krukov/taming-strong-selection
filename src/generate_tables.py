from transition_probability_dynamic_failures import matrix
from pathlib import Path
import pickle
from tqdm import tqdm

N_range = [1000, 100, 200]
Ns_range = [0, 10, 50]
max_t = 4
n = 100
tmp_store = Path("data")

for N in tqdm(N_range):
    for Ns in tqdm(Ns_range):
       s = Ns / N

       mtx_pkl = tmp_store / Path(f"mtx_n_{n}_Ns_{Ns}_N_{N}.pypkl")
       if not mtx_pkl.exists():
           R, _, _ = matrix(n, N=N, s=s, max_t=max_t)
           
           with open(mtx_pkl, "wb") as pkl:
               pickle.dump(R, pkl)
