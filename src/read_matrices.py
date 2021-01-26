import numpy as np
from argparse import ArgumentParser
from scipy.stats import hypergeom
import sys
# from pathlib import Path

def hypergeom_projection(N, n):
    rN = np.arange(0, N+1)
    rn = np.arange(0, n+1)
    return np.array([hypergeom(N, i, n).pmf(rn) for i in rN])

def python2round(f):
    if round(f + 1) - round(f) != 1:
        return f + abs(f) / f * 0.5
    return round(f)

def index_bis(i, n):
    return int(min(max(python2round(i * n / float(n+1)), 2), n-2))

def calcJK13(n):
    J = np.zeros((n, n-1))
    for i in range(n):
        ibis = index_bis(i + 1, n) - 1
        J[i, ibis] = -(1.+n) * ((2.+i)*(2.+n)*(-6.-n+(i+1.)*(3.+n))-2.*(4.+n)*(-1.+(i+1.)*(2.+n))*(ibis+1.)
                  +(12.+7.*n+n**2)*(ibis+1.)**2) / (2.+n) / (3.+n) / (4.+n)
        J[i, ibis - 1] = (1.+n) * (4.+(1.+i)**2*(6.+5.*n+n**2)-(i+1.)*(14.+9.*n+n**2)-(4.+n)*(-5.-n+2.*(i+1.)*(2.+n))*(ibis+1.)
                    +(12.+7.*n+n**2)*(ibis+1.)**2) / (2.+n) / (3.+n) / (4.+n) / 2.
        J[i, ibis + 1] = (1.+n) * ((2.+i)*(2.+n)*(-2.+(i+1.)*(3.+n))-(4.+n)*(1.+n+2.*(i+1.)*(2.+n))*(ibis+1.)
                    +(12.+7.*n+n**2)*(ibis+1.)**2) / (2.+n) / (3.+n) / (4.+n) / 2.
    return J


if __name__ == '__main__':
    parser = ArgumentParser('convert matrices from transition_probability_explicit')
    parser.add_argument('--sample-size', '-n', type=int)
    parser.add_argument('--jackknife', '-j', type=int)
    parser.add_argument('output_file')

    args = parser.parse_args()
    n = args.sample_size
    k = args.jackknife

    # txt = Path(args.input_file).read_text()
    txt = sys.stdin.read()
    T = list()

    chunks = txt.split('---')
    for i, chunk in enumerate(chunks):
        if not chunk.isspace():
            t = np.fromstring(chunk, sep=' ')
            T.append(t.reshape((i+1, n+1)))

    M = np.zeros((n+1, n+1))
    assert np.all(T[0] == 0)

    # first prt of the sum - hypergeometric down
    for i in range(0, n+1):
        h = hypergeom_projection(n, i)
        M += h @ T[i]

    # second part - jackknife up
    J = np.eye(n+1)
    for i in range(n+1, n+k+1):
        J = calcJK13(i+1) @ J
        M += J.T @ T[i]

    np.savetxt(args.output_file, M)
