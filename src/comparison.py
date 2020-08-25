from transition_probability_dynamic_failures import matrix,P0,Pf
from transition_probability_selection import matrix_selection
import numpy as np
from time import perf_counter

np.set_printoptions(precision=2, linewidth=70)

Ne = 1e3
s = 1e-2
n = 100
max_t=1

print("Computing the old verion...")
t_start = perf_counter()
M_old, cP = matrix_selection(
    n, Ne, s
)  # The second return is a cache of pre-built matrices
print("Took ", perf_counter() - t_start)

print("Computing the new version...")
# This will be slow with large n
t_start = perf_counter()
M_new, cP0, cPf = matrix(n, s=s, N=Ne, max_t=max_t)
print("Took ", perf_counter() - t_start)

print("absolute differences")
print(M_new - M_old)
print("relative differences")

relative_diff = (M_new - M_old)/M_old
print(relative_diff)

print("old")
print(M_old)

print("new")
print(M_new)

print("Expectedd order of difference: ", s/Ne)

material_differences = relative_diff[~np.isnan(relative_diff)]

print("Max relative difference: ", np.max(np.abs(material_differences)))

# print(cP0)
# print(cPf)

print(np.count_nonzero(cP0) / cP0.size, np.count_nonzero(cPf) / cPf.size)
print(np.count_nonzero(cP) / cP.size)
print(cP0.size, cPf.size)
