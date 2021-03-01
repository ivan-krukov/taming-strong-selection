# Comparing fixation rates between diffusion model and the exact Wright-Fisher

from wright_fisher import wright_fisher
import numpy as np
from numpy.linalg import solve
from scipy.stats import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

def pfix_wf(N, s):
    wf = wright_fisher(2*N, s)
    wf_Q = wf[1:-1, 1:-1]
    wf_R = wf[1:-1, -1]
    I = np.eye(wf_Q.shape[0])
    B1 = solve(I-wf_Q, wf_R)

    return B1[0]

def pfix_df(N, s):
    return (1 - np.exp(-s)) / (1 - np.exp(-2*N*s))

N = int(sys.argv[1])
Ns_range = np.arange(-0.1, -20, -0.1)

# Eyre-Walker, Woolfit, Phelps, Genetics 2006
dfe = gamma(1/0.23)

print("selection,diffusion,wright_fisher,dfe")
for Ns in tqdm(Ns_range):
    # diffusion
    df = pfix_df(N, Ns/N)
    # Wright-Fisher
    wf = pfix_wf(N, Ns/N)
    # Difstribution of fitness effects
    fe = dfe.pdf(-Ns)
    print(Ns, df, wf, fe, sep=",")
