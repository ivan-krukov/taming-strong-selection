from transition_probability_dynamic_failures import matrix,P0,Pf
from transition_probability_selection import matrix_selection
import numpy as np
from time import perf_counter

np.set_printoptions(precision=3, linewidth=100)

Ne = 1e3
s = 1e-2
n = 100
max_t=1

print("Computing the old verion...")
t_start = perf_counter()
M_old, _ = matrix_selection(
    n, Ne, s
)  # The second return is a cache of pre-built matrices
print("Took ", perf_counter() - t_start)

print("Computing the new version...")
# This will be slow with large n
t_start = perf_counter()
M_new = matrix(n, s=s, N=Ne, max_t=max_t)
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

# print(P0(1, 2, 1, 2, s, Ne, max_t))

# print("Pf(1, 1, 1, 1, s, Ne, max_t)")
# print(Pf(1, 1, 1, 1, s, Ne, 1, 1))


# print("Pf(1, 1, 1, 1, s, Ne, 0)")
# print(Pf(1, 1, 1, 1, s, Ne, 0, 0))

# print("P0(1, 1, 1, 1, s, Ne, 0)")
# print(P0(1, 1, 1, 1, s, Ne, 0, 0))

print("Expectedd order of difference: ", s/Ne)

material_differences = relative_diff[~np.isnan(relative_diff)]

print("Max relative difference: ", np.max(np.abs(material_differences)))

