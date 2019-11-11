import numpy as np
import numpy.linalg as la


def allele_age(Q, starting=1, mean_only=False):
    """
    Calculate mean and standard deviation of allele age and variance for a sample of `n` chromosomes, having observed `observed` copies
    """
    p = starting - 1            # starting index

    I = np.eye(Q.shape[0])
    ImQ = I - Q

    e = np.zeros_like(Q[0])
    e[p] = 1

    M1 = la.solve(ImQ.T, e)
    M2 = la.solve(ImQ.T, M1)
    M3 = la.solve(ImQ.T, M2)

    mu_1 = (M2 @ Q) / M1
    A = Q @ (I + Q)
    mu_2 = M3 @ A / M1

    if mean_only:
        return mu_1
    else:
        return mu_1, np.sqrt(mu_2 - np.power(mu_1, 2))


def diffusion_allele_age(p):
    """
    Diffusion approximation of allele age, starting at allele frequency `p`
    """
    return ((-2 * p) / (1 - p)) * np.log(p)
