import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
import seaborn as sns
from occupancy import dist_num_anc, reduced_occupancy

def approx_mean(n, N, s):
    # return (s * n)/(1 - s) + (N * (1 - np.power(1-(1/N), (n*(1+s))))) - (n*s)
    k = 1 - (1/N)
    return N * (1 - np.power( (k*(1-s)) / (1 - k*s), n))

def approx_std_dev(n, N, s):
    # return np.sqrt(N*(np.power(1 - 1/N,n) + np.power(1 - 2/N,(n))*(N-1) - np.power(1-1/N, 2*(n))*N) + (n*s)/np.power(1-s,2) )
    pow = np.power
    k = 1 - (1/N)
    k2 = pow(k, 2)
    kb = 1 - (2/N)
    z = 1 - s
    first = pow(k, 2*n) * pow(N, 2) * (pow(z/(1-(k2*s)),n) - pow(z/(1-(k*s)),2*n))
    first = max(first, 0)
    second =  (N-1) * pow(kb*z/(1-(kb*s)), n)
    second += pow(k*z / (1 - (k*s)), n)
    second -= N * pow(k2*z / (1-(k2*s)), n)
    second = max(second, 0)
    return np.sqrt(first + N * second)

if __name__ == "__main__":
    N = 1000
    n = 200
    ## percentiles = [0.5, 0.9, 0.99, 0.999]
    plot_range = np.arange(n-30, n+20)
    rocc = reduced_occupancy(1000)
    x = 1

    with sns.plotting_context("paper", font_scale=1.5):
        fig, ax = plt.subplots()
        # ax.axhline(y=200, color="k", ls=":")
        for (i,Ns) in enumerate([1/1000, 5, 10, 50]):
            s = Ns / N
            dist = np.zeros(len(plot_range))
            for (j, a) in enumerate(plot_range):
                dist[j] = dist_num_anc(a, n, x, s, N, rocc)

            kwargs = dict(n=n, N=N, s=Ns/N)
            z = norm(approx_mean(**kwargs), approx_std_dev(**kwargs))
            ax.plot(plot_range, dist, ls="", marker="o", color=f"C{i}")
            ax.plot(plot_range, z.pdf(plot_range), label=Ns, color=f"C{i}")
            #ax.fill_between(plot_range, z.pdf(plot_range), where=plot_range > z.ppf(0.99), alpha=0.5, color=f"C{i}")

        ax.legend(frameon=False, title="Ns")
        ax.set(ylabel="Probability", xlabel="Sample size", title=f"N={N}, n={n}")
        fig.tight_layout()
        fig.savefig("fig/normal-approximation.pdf")
