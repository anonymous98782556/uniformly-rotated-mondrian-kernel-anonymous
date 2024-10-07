import matplotlib.pylab as plt
import numpy as np
import sys

from config import *
from mondrian_kernel import evaluate_all_lifetimes
from random_binning import evaluate_random_binning
from utils import load_CPU, SGD_regression_test_error, select_width, initialize_plotting, remove_chartjunk, tableau20

def evaluate_fourier_features(X, y, X_test, y_test, M, lifetime, delta):
    # get Fourier features
    D = X.shape[1]
    omega = lifetime * np.random.standard_cauchy(size=(D, M))
    Z = np.c_[np.cos(X.dot(omega)), np.sin(X.dot(omega))]
    Z_test = np.c_[np.cos(X_test.dot(omega)), np.sin(X_test.dot(omega))]

    SGD_epochs = 10
    error_test = SGD_regression_test_error(Z, y, Z_test, y_test, delta, SGD_epochs)
    sys.stdout.write('\rFF lg_lifetime = %.3f; error_test = %.2f%%   ' % (np.log2(lifetime), error_test))
    sys.stdout.flush()
    return error_test

def experiment_4_experimental_dataset_time():
    """ Plots test error performance as a function of computational time on the CPU data set. For the Mondrian kernel and
        uniformly rotated Mondrian kernel, all lifetime values from 0 up to a terminal lifetime are swept through. For 
        Fourier features and random binning a binary search procedure is employed to find good lifetime parameter 
        values, with an initial expansion phase.
    """

    # fix random seed
    np.random.seed(seed)

    # load data
    X, y, X_test, y_test = load_CPU()

    # set parameters
    M = 350                     # number of Mondrian trees to use
    lifetime_max = 1e-6         # terminal lifetime
    delta = 0.0001              # ridge regression delta

    # set up plot
    fig = plt.figure(num=4, figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    remove_chartjunk(ax)

    ax.yaxis.grid(True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    ax.set_xlabel('computational time [$s$]')
    ax.set_ylabel('validation set relative error [$\\%$]')
    ax.set_xscale('log')
    ax.set_ylim((0, 25))

    # Fourier features
    runtimes, errors = select_width(X, y, X_test, y_test, M, delta, evaluate_fourier_features, 100)
    ax.scatter(runtimes, errors, marker='^', color=tableau20(0), label='Fourier features')
    # random binning
    runtimes, errors = select_width(X, y, X_test, y_test, M, delta, evaluate_random_binning, 50)
    ax.scatter(runtimes, errors, marker='o', color=tableau20(6), label='random binning')
    # Mondrian kernel
    res = evaluate_all_lifetimes(X, y, X_test, y_test, M, lifetime_max, delta, mondrian_kernel=True)
    ax.scatter(res['runtimes'], res['kernel_test'], marker='.', color=tableau20(4), label='Mondrian kernel')
    # uniformly rotated Mondrian kernel
    res = evaluate_all_lifetimes(X, y, X_test, y_test, M, lifetime_max, delta, mondrian_kernel=True, uniformly_rotated=True)
    ax.scatter(res['runtimes'], res['kernel_test'], marker='*', color=tableau20(2), label='uniformly rotated')
    ax.legend(frameon=False, loc='best')

if __name__ == "__main__":
    initialize_plotting()
    experiment_4_experimental_dataset_time()
    plt.show()
