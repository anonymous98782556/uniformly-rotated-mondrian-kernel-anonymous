import matplotlib.pylab as plt

from utils import initialize_plotting
from experiment_1_limiting_kernel_convergence import experiment_1_limiting_kernel_convergence
from experiment_2_ground_truth_lifetime import experiment_2_ground_truth_lifetime
from experiment_3_experimental_dataset_features import experiment_3_experimental_dataset_features
from experiment_4_experimental_dataset_time import experiment_4_experimental_dataset_time

if __name__ == "__main__":
    initialize_plotting()
    print("Experiment 1: experiment_convergence_kernelerror")
    experiment_1_limiting_kernel_convergence()
    print("\nExperiment 2: experiment_synthetic")
    experiment_2_ground_truth_lifetime()
    print("\nExperiment 3: experiment_convergence_testerror")
    experiment_3_experimental_dataset_features()
    print("\nExperiment 4: experiment_CPU")
    experiment_4_experimental_dataset_time()
    plt.show()