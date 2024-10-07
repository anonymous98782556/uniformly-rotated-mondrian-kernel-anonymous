import matplotlib.pylab as plt
import numpy as np

from config import *
from mondrian_kernel import evaluate_all_lifetimes
from utils import initialize_plotting, construct_data_synthetic_uniformly_rotated, remove_chartjunk, tableau20

def experiment_2_ground_truth_lifetime():
    """ Simulates data from a Gaussian process prior with a uniformly rotated Mondriran limited kernel of known lifetime 
        (inverse width). Then the uniformly rotated Mondrian kernel procedure for evaluating all lifetimes from 0 up to 
        a terminal lifetime is run on this simulated dataset and the results are plotted, showing how accurately the ground 
        truth inverse kernel width could be recovered.
    """

    # fix random seed
    np.random.seed(seed)

    # synthetize data from uniformly rotated Mondrian limiting kernel
    D = 2
    lifetime = 10.00
    noise_var = 0.01 ** 2
    N_train = 500
    N_validation = 500
    N_test = 500
    X, y, X_test, y_test = construct_data_synthetic_uniformly_rotated(D, lifetime, noise_var, N_train, N_validation + N_test)

    # Mondrian kernel lifetime sweep parameters
    M = 50
    lifetime_max = lifetime * 3
    delta = noise_var   # prior variance 1

    res = evaluate_all_lifetimes(X, y, X_test, y_test, M, lifetime_max, delta, mondrian_kernel=True, uniformly_rotated=True, validation=True)
    lifetimes = res['times']
    error_train = res['kernel_train'] 
    error_validation = res['kernel_validation']
    error_test = res['kernel_test'] 
    error_train = [num / 100.0 for num  in error_train]
    error_validation = [num / 100.0 for num  in error_validation]
    error_test = [num / 100.0 for num  in error_test]

    # set up plot
    fig = plt.figure(num=2, figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    remove_chartjunk(ax)

    ax.set_title('$M = %d$, $\\mathcal{D}$ = synthetic ($D = 2$, $N = N_{val} = N_{test}=%d$)' % (M, N_validation))
    ax.set_xscale('log')
    ax.yaxis.grid(which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    ax.set_xlabel('lifetime $\\lambda$')
    ax.set_ylabel('relative error [$\\%$]')

    ax.plot(lifetimes, error_train, drawstyle="steps-post", ls='-', color=tableau20(15), label='train')
    ax.plot(lifetimes, error_test, drawstyle="steps-post", ls='-', color=tableau20(4), label='test')
    ax.plot(lifetimes, error_validation, drawstyle="steps-post", ls='-', lw=2, color=tableau20(2), label='validation')
    ax.legend(frameon=False)
    
    # plot ground truth and estimate
    ax.axvline(x=10, ls=':', color='black')
    i = np.argmin(error_validation)
    lifetime_hat = lifetimes[i]
    print('lifetime_hat = %.3f' % lifetime_hat)
    ax.plot([lifetime_hat, lifetime_hat], [0, error_validation[i]], ls='dashed', lw=2, color=tableau20(2))
    ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, lifetime_hat], labels=["$10^{-2}$", "$10^{-1}$", "$10^0$", "$\\lambda_0$", "$\\hat{\\lambda}$"])
    ax.set_xlim((1e-2, lifetime_max))
    
def main():
    initialize_plotting()
    experiment_2_ground_truth_lifetime()
    plt.show()

if __name__ == "__main__":
    main()