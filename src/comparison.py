from dynamic_tinkered import matrix,P0,Pf
from transition_probability_selection import matrix_selection

Ne = 10**10
s = 1e-2
n = 2
max_t=1

print("Computing the old verion...")
M_old, _ = matrix_selection(
    n, Ne, s
)  # The second return is a cache of pre-built matrices

print("Computing the new version...")
# This will be slow with large n
M_new = matrix(n, s=s, N=Ne, max_t=max_t)

print("absolute differences")
print(M_new - M_old)
print("relative differences")
print((M_new - M_old)/M_old)

print("new")
print(M_new)

print(P0(1, 2, 1, 2, s, Ne, max_t))

print("Pf(1, 1, 1, 1, s, Ne, max_t)")
print(Pf(1, 1, 1, 1, s, Ne, 1))


print("Pf(1, 1, 1, 1, s, Ne, 0)")
print(Pf(1, 1, 1, 1, s, Ne, 0))

print("P0(1, 1, 1, 1, s, Ne, 0)")
print(P0(1, 1, 1, 1, s, Ne, 0))