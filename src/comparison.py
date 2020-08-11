from dynamic_tinkered import matrix
from transition_probability_selection import matrix_selection

Ne = 1000
s = 1e-2
n = 1

print("Computing the old verion...")
M_old, _ = matrix_selection(
    n, Ne, s
)  # The second return is a cache of pre-built matrices

print("Computing the new version...")
# This will be slow with large n
M_new = matrix(n, s=s, N=Ne, max_t=1)

print("absolute differences")
print(M_new - M_old)
print("relative differences")
print((M_new - M_old)/M_old)

print("old")
print(M_old)

