This page provides the python code for preferential Bayesian optimization (BO), which includes experiments in [[1]](https://arxiv.org/abs/2302.01513).  
If you use this codes for research, consider citing [1].

# Environment
* Linux
    * We ran the experiments on CentOS 6.9.
    * We confirmed the scripts running on Ubuntu 20.04 (latest LTS), but for this environment, we have not confirmed that the result of the paper can be completely reproduced (the difference in OS may produce a slight change in the results).
* Python 3.9.0
* All packages are in requirements.txt

# Instruction

* We implement Gibbs sampling-based preferential Gaussian process regression (GPR) and the preferential BO methods called hallucination believer (HB) with EI and UCB.

* Organization of the codes:
    * EP_Implementation.md: we provide the details of implementation of EP for GPR.
    * examples: notebooks for GPR and preferential BO
        * GPmodel_comparison: make the illustrative example in the paper (Fig1. in [1])
        * PreferentialBO: perform preferential BO using our package.
    * experiments: the scripts to reproduce the experiments in the paper [1].
        * In an experiment with respect to MCMC, we use the code from [2,3,4]. See README in the folder named ``experiments'' for details.
    * src: main modules for preferential BO and GPR for preferential learning.
    * test_functions: benchmark functions.

# Minimum Example
```python
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
```

# Reference
[1]: [Takeno, S., Nomura, M., Karasuyama, M., Towards Practical Preferential Bayesian Optimization with Skew Gaussian Processes, arXiv:2302.01513, 2023.](https://arxiv.org/abs/2302.01513)

[2]: Benavoli, A., Azzimonti, D., and Piga, D. Skew Gaussian processes for classification. Machine Learning, 109(9): 1877–1902, 2020.

[3]: Benavoli, A., Azzimonti, D., and Piga, D. Preferential Bayesian optimisation with skew Gaussian processes. In Proceedings of the Genetic and Evolutionary Computation Conference Companion, pp. 1842–1850, 2021.

[4]: Benavoli, A., Azzimonti, D., and Piga, D. A unified framework for closed-form nonparametric regression, classification, preference and mixed problems with skew Gaussian processes. Machine Learning, 110(11):3095–3133, 2021.