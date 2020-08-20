import numpy as np
def two_by_two(s, N):

    case = [0 for _ in range(8 + 1)]

    case[1] = (1/2) *     (1-s) *         (1 - (1/N))
    case[2] = (1/2) * s * (1-s) * (1/N) * (1 - (1/N))
    case[3] = (1/2) *     (1-s) *         (1 - (1/N))
    case[4] = (1/2) * s * (1-s) * (1/N) * (1 - (1/N))
    case[5] = (1/2) * s * (1-s) * (1/N) * (1 - (1/N))
    case[5] = (1/2) * s * (1-s) * (1/N) * (1 - (1/N))
    case[6] = (1/2) * s * (1-s) * (1/N) * (1 - (1/N)) * (1/N) * s
    case[7] = (1/2) * s * (1-s) * (1/N) * (1 - (1/N))
    case[8] = (1/2) * s * (1-s) * (1/N) * (1 - (1/N)) * (1/N) * s

    return sum(case)

def two_by_two_new_simplitied(s, N):
    return (1-s) * (1 - (1/N)) * (1 + (s/N))**2

def two_by_two_old_simplitied(s, N):
    return (1-s) * (1 - (1/N)) * (1 + (s/N))

N = 1e5
s = 1e-2
n = 2
x = two_by_two(s=s, N=N)
y = two_by_two_new_simplitied(s=s, N=N)
z = two_by_two_old_simplitied(s=s, N=N)

from dynamic_tinkered import matrix, P0, matrix_nop
from transition_probability_selection import matrix_selection, matrix_selection_nop, Qs

M_new = matrix_nop(n, s=s, N=N, max_t=1) #matrix(n, s=s, N=N, max_t=1)
M_old, _ = matrix_selection_nop(n, s=s, N=N) #matrix(n, s=s, N=N, max_t=1)

cache =  np.full((2*n+1, 2*n+1, 2*n+1, 2*n+1), np.nan)
M_old_rect = Qs(1,2,1,2,s=s,N=N,cache=cache)
# M_old, _ = matrix_selection(n, N, s)
# print(x, y, z)
# print(M_old[1,1], M_new[1,1])
# print(y - M_old[1,1], z - M_new[1,1])
print(y - M_new[1,1])
print(z - M_old[1,1])
print(z - M_old_rect)
