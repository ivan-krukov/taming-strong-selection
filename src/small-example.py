import numpy as np
def two_by_two(s, N):

    case = [0 for _ in range(8 + 1)]

    case[1] = (1/2) *     (1-s) *         (1 - (1/N))
    print("C1 ", case[1])
    case[2] = (1/2) * s         * (1/N) * (1 - (1/N))
    print("C2 ", case[2])
    case[3] = (1/2) *     (1-s) *         (1 - (1/N))
    print("C3 ", case[3])
    case[4] = (1/2) * s * (1-s) * (1/N) * (1 - (1/N))
    print("C4 ", case[4])
    case[5] = (1/2) * s         * (1/N) * (1 - (1/N))
    print("C5 ", case[5])
    case[6] = (1/2) * s         * (1/N) * (1 - (1/N)) * (1/N) * s
    print("C6 ", case[6])
    case[7] = (1/2) * s * (1-s) * (1/N) * (1 - (1/N))
    print("C7 ", case[7])
    case[8] = (1/2) * s         * (1/N) * (1 - (1/N)) * (1/N) * s
    print("C8 ", case[8])

    return case, sum(case)

def two_by_two_new_simplitied(s, N):
    return (1-s) * (1 - (1/N)) * (1 + (s/N))**2

def two_by_two_old_simplitied(s, N):
    return (1-s) * (1 - (1/N)) * (1 + (s/N))

N = 1e1
s = 2e-1
n = 2
cases, x = two_by_two(s=s, N=N)
# y = two_by_two_new_simplitied(s=s, N=N)
# z = two_by_two_old_simplitied(s=s, N=N)

from transition_probability_dynamic_failures import matrix, P0
from transition_probability_selection import matrix_selection, Qs

M_new = matrix(n, s=s, N=N, max_t=1) #matrix(n, s=s, N=N, max_t=1)
M_old, _ = matrix_selection(n, s=s, N=N) #matrix(n, s=s, N=N, max_t=1)

cache =  np.full((2*n+1, 2*n+1, 2*n+1, 2*n+1), np.nan)
M_old_rect = Qs(1,2,1,2,s=s,N=N,cache=cache)
print(M_new - M_old)

