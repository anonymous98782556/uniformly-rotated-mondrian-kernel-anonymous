import matplotlib.pylab as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import sys

from config import *
from mondrian_kernel import Mondrian_kernel_features, uniformly_rotated_Mondrian_kernel_features
from random_binning import random_binning_features
from utils import initialize_plotting, Fourier_features, remove_chartjunk, tableau20

# TODO: this now only works for 2D data
def run_and_plot_experiment_convergence(X, lifetime, M_max, num_sweeps):
    """ Compares kernel approximation error of different random feature schemes (Fourier features,
    	random binning, Mondrian features, uniformly rotated Mondrian features) to their limiting 
        kernel with the given lifetime.

        Note that this now only works for 2D data as the uniformly rotated Mondrian kernel was
        explicitly calculated.
    """

    # fix random seed
    np.random.seed(seed)

    # precompute NxN kernel (Gram) matrix
    K = np.exp(- lifetime * squareform(pdist(X, 'cityblock')))
    # precompute the 2x2 uniformly rotated kernel matrix
    def func(val, lifetime):
        x_span = np.linspace(0, np.pi / 4, num=10000)
        y_span = 4 / np.pi * np.exp(-1 * lifetime * np.sqrt(2) * np.cos(x_span) * val)
        return np.trapezoid(y_span, x_span)
    mfunc = np.vectorize(func)
    K_UR = mfunc(squareform(pdist(X, 'euclidean')), lifetime)

    # compute maximum kernel approximation error as number of features increases
    def get_errors(get_features, lifetime, R_max, scheme_name, K):
        errors = [[] for _ in range(R_max)]
        for repeat in range(num_sweeps):
            # obtain fresh features
            Z, feature_from_repetition = get_features(X, lifetime, R_max)
            for R in range(1, R_max+1):
                # save maximum kernel approximation error
                fs = feature_from_repetition < R
                E = abs(Z[:, fs].dot(np.transpose(Z[:, fs])) / R - K)
                errors[R-1].append(np.max(E))
            sys.stdout.write("\r%s %d / %d" % (scheme_name, repeat+1, num_sweeps))
            sys.stdout.flush()
        sys.stdout.write("\n")
        return map(np.mean, errors), map(np.std, errors)

    FF_error_avg, FF_error_std = get_errors(Fourier_features, lifetime, int(M_max/2), 'Fourier features', K)
    RB_error_avg, RB_error_std = get_errors(random_binning_features, lifetime, M_max, 'random binning', K)
    MK_error_avg, MK_error_std = get_errors(Mondrian_kernel_features, lifetime, M_max, 'Mondrian kernel', K)
    UR_error_avg, UR_error_std = get_errors(uniformly_rotated_Mondrian_kernel_features, lifetime, M_max, 
                                            'uniformly rotated', K_UR)

    # plot
    fig = plt.figure(num=1, figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    remove_chartjunk(ax)
    
    ax.yaxis.grid(which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    ax.set_xlabel('$M$ (nonzero features per data point)')
    ax.set_ylabel('maximum absolute error')
    ax.set_ylim((0, 1.2))
    ax.errorbar(range(1, M_max+1, 2), list(FF_error_avg), yerr=list(FF_error_std), marker='^', linestyle='',
                markeredgecolor=tableau20(0), color=tableau20(0), label='Fourier features')
    ax.errorbar(range(1, M_max+1, 1), list(RB_error_avg), yerr=list(RB_error_std), marker='o', linestyle='',
                markeredgecolor=tableau20(6), color=tableau20(6), label='random binning')
    ax.errorbar(range(1, M_max+1, 1), list(MK_error_avg), yerr=list(MK_error_std), marker='v', linestyle='',
                markeredgecolor=tableau20(4), color=tableau20(4), label='Mondrian kernel')
    ax.errorbar(range(1, M_max+1, 1), list(UR_error_avg), yerr=list(UR_error_std), marker='*', linestyle='',
                markeredgecolor=tableau20(2), color=tableau20(2), label='uniformly rotated')
    ax.legend(frameon=False, loc='best')

def experiment_1_limiting_kernel_convergence():
    # obtain data
    D = 2                       # input dimension
    N = 100                     # number of sampling locations
    X = np.random.rand(N, D)    # sample N datapoints uniformly at random from unit interval/square/cube/hypercube

    # run experiment and plot results
    lifetime = 10.0
    M_max = 50                  # maximum value of M to sweep until
    num_sweeps = 5              # sweeps through M (repetitions of the experiment)
    run_and_plot_experiment_convergence(X, lifetime, M_max, num_sweeps)

def main():
    initialize_plotting()
    experiment_1_limiting_kernel_convergence()
    plt.show()

if __name__ == "__main__":
    main()