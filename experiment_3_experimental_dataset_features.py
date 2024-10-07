import matplotlib.pylab as plt
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
import sys

from config import *
from mondrian_kernel import Mondrian_kernel_features, uniformly_rotated_Mondrian_kernel_features
from random_binning import random_binning_features
from utils import load_CPU, Fourier_features, initialize_plotting, remove_chartjunk, exact_regression_test_error, tableau20

# constant from binary search
lifetime_uniform_rotated = 2.5e-7

def run_and_plot_experiment_convergence_testerror(X, y, X_test, y_test, M_max, lifetime, delta, num_sweeps, exact_Laplace=False):
    """ Compares approximation error of different random feature schemes (Fourier features,
        random binning, Mondrian features, uniformly rotated Mondrian kernel) via test set error.
    """

    # fix random seed
    np.random.seed(seed)

    N, D = np.shape(X)
    X_all = np.array(np.r_[X, X_test])

    if exact_Laplace:
        K_all = scipy.exp(- lifetime * squareform(pdist(X_all, 'cityblock')))
        h = np.linalg.solve(K_all[:N, :N] + delta * np.identity(N), y - np.mean(y))
        y_test_hat = np.mean(y) + np.transpose(K_all[:N, N:]).dot(h)
        Laplace_error_test = 100.0 * np.linalg.norm(y_test_hat - y_test) / np.linalg.norm(y_test)
        print(Laplace_error_test)

    # compute RMSE as the number of features increases
    def get_errors(get_features, lifetime, R_max, scheme_name):
        errors = [[] for _ in range(R_max)]
        for sweep in range(num_sweeps):
            # obtain fresh features
            Z, feature_from_repetition = get_features(X_all, lifetime, R_max)
            for R in range(1, R_max+1):
                # save maximum kernel approximation error
                fs = feature_from_repetition < R
                Z_train = Z[:N, fs] / np.sqrt(R)
                Z_test = Z[N:, fs] / np.sqrt(R)
                error_test = exact_regression_test_error(Z_train, y, Z_test, y_test, delta)
                errors[R-1].append(error_test)
            sys.stdout.write("\r%s %d / %d" % (scheme_name, sweep + 1, num_sweeps))
            sys.stdout.flush()
        sys.stdout.write("\n")
        return map(np.mean, errors), map(np.std, errors)

    FF_error_avg, FF_error_std = get_errors(Fourier_features, lifetime, int(M_max/2), 'Fourier features')
    RB_error_avg, RB_error_std = get_errors(random_binning_features, lifetime, M_max, 'random binning')
    MK_error_avg, MK_error_std = get_errors(Mondrian_kernel_features, lifetime, M_max, 'Mondrian kernel')
    UR_error_avg, UR_error_std = get_errors(uniformly_rotated_Mondrian_kernel_features, lifetime_uniform_rotated, 
                                            M_max, 'uniformly rotated')

    # plot error against # M of non-zero features
    fig = plt.figure(num=3, figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    remove_chartjunk(ax)
    ax.yaxis.grid(which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    ax.set_xlabel('$M$ (non-zero features per data point)')
    ax.set_ylabel('relative test set error [$\\%$]')
    ax.errorbar(range(1, M_max+1, 2), list(FF_error_avg), yerr=list(FF_error_std), marker='^',
                markeredgecolor=tableau20(0), ls='', color=tableau20(0), label='Fourier features')
    ax.errorbar(range(1, M_max+1, 1), list(RB_error_avg), yerr=list(RB_error_std), marker='o',
                markeredgecolor=tableau20(6), ls='', color=tableau20(6), label='random binning')
    ax.errorbar(range(1, M_max+1, 1), list(MK_error_avg), yerr=list(MK_error_std), marker='v',
                markeredgecolor=tableau20(4), ls='', color=tableau20(4), label='Mondrian kernel')
    ax.errorbar(range(1, M_max+1, 1), list(UR_error_avg), yerr=list(UR_error_std), marker='*',
                markeredgecolor=tableau20(2), ls='', color=tableau20(2), label='uniformly rotated')
    # add exact Laplace kernel regression test set RMSE (on CPU this is 3.12% with lifetime 1.0)
    ax.axhline(y=3.12, color='black', lw=2)
    ax.legend(frameon=False)

def experiment_3_experimental_dataset_features():
    np.random.seed(0)

    # load CPU data
    X, y, X_test, y_test = load_CPU()

    # run experiment and plot results
    M_max = 50
    lifetime = 1e-6       # value used by Rahimi & Recht
    delta = 1e-4          # value used by Rahimi & Recht
    num_repeats = 5       # number of experiment repetitions to get error bars
    run_and_plot_experiment_convergence_testerror(X, y, X_test, y_test, M_max, lifetime, delta, num_repeats)

def main():
    initialize_plotting()
    experiment_3_experimental_dataset_features()
    plt.show()

if __name__ == "__main__":
    main()