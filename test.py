import numpy as np
from preferentialBO.src.preferential_bayesian_opt import *

model_name = "Gibbs"
first_acq = "current_best"
second_acq = "EI"  # for HB-EI
# second_acq = "UCB" # for HB-UCB

# seed for reproducibility
np.random.seed(0)

kernel_lengthscale_bounds = np.array([[1, 1], [10, 10],])
kernel = GPy.kern.RBF(input_dim=2, ARD=True)

# bayesian optimizer
optimizer = PreferentialBO_HallucinationBeliever(
    # Left d (=2) columns and right d columns express winners and losers, respectively.
    X=np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),  # i.e., (0, 1) > (2, 3) and (1, 2) > (3, 4)
    x_bounds=np.array([[0, 0], [10, 10]]),  # input domain X = [0, 10]^2
    kernel=kernel,
    kernel_bounds=kernel_lengthscale_bounds,
    noise_std=1e-2,
    GPmodel=model_name,
)
# marginal likelihood maximization using Laplace approximation
optimizer.GPmodel.model_selection()

# for the continuous X
x1, x2 = optimizer.next_input(first_acq=first_acq, second_acq=second_acq)

# X_all is numpy array (N \times d) for the pool setting
# x1, x2 = optimizer.next_input_pool(first_acq=first_acq, second_acq=second_acq, X=X_all)

print("Selected duel is {} {}".format(x1, x2))

# if x1 > x2, add the new training duel
optimizer.update(X_win=x1, X_loo=x2)
