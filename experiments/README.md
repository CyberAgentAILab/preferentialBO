# Instruction

* Methods
    * HB-EI (proposed)
    * HB-UCB (proposed)
    * LA-EI
    * EP-EI
    * EP-MUC
    * DuelTS
    * DuelUCB
    * EIIG

*  For all the benchmark function experiments:
    * For 10 parallel experiments of different seeds (0 ~ 9):
        * python preferential_bayesopt_exp.py method_name Benchmark_name 0 -1 1
    * If you run an experiment with one specific seed:
        * python preferential_bayesopt_exp.py method_name Benchmark_name 0 seed 1
    * For the plots of the experimental results for regret:
        * python plot_results_benchmark.py

* For other comparisons:
    * An illustrative example shown in Figure 1 can be reproduced in the notebook named GPmodel_comparison.ipynb.
    * For comparisons between MCMC, LA, and EP (Figure 2 (a)):
        * python compare_PGPmodels.py Benchmark_name
    * For comparison between Gibbs sampling and LinESS (Figure 3 and Table 1):
        * python make_trace_plots.py Benchmark_name
        * For the experiments of LinESS, we use the code from https://github.com/benavoli/SkewGP [1,2,3] in the folder ``Benavoli_code.''
    * For comparison of MC estimators (Figure 2 (b)):
        * python compare_MCestimator.py Benchmark_name
    * For comparison of marginal likelihood approximation only for Benchmark_name with input dimension d = 2 (Figure 5 in Appendix):
        * python compare_marginal_likelihood.py Benchmark_name


# Reference

[1]: Benavoli, A., Azzimonti, D., and Piga, D. Skew Gaussian processes for classification. Machine Learning, 109(9): 1877–1902, 2020.

[2]: Benavoli, A., Azzimonti, D., and Piga, D. Preferential Bayesian optimisation with skew Gaussian processes. In Proceedings of the Genetic and Evolutionary Computation Conference Companion, pp. 1842–1850, 2021.

[3]: Benavoli, A., Azzimonti, D., and Piga, D. A unified framework for closed-form nonparametric regression, classification, preference and mixed problems with skew Gaussian processes. Machine Learning, 110(11):3095–3133, 2021.