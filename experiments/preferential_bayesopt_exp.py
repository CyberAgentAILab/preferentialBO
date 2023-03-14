# -*- coding: utf-8 -*-
import sys
import os
import signal
import time
import concurrent.futures
import pickle

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# matplotlib.use("Agg")
import GPy

from preferentialBO.test_functions import test_functions
from preferentialBO.src.preferential_gp_regression import *
from preferentialBO.src.preferential_bayesian_opt import *


def main(params):
    (func_name, BO_method, initial_seed, function_seed, MAX_NUM_WORKER) = params
    print(params)

    test_func = eval("test_functions." + func_name)()
    if "SynFun" in func_name:
        test_func.func_sampling(seed=function_seed)
    x_bounds = test_func.bounds
    input_dim = test_func.d
    interval_size = x_bounds[1] - x_bounds[0]
    kernel_bounds = np.array([interval_size * 1e-1, interval_size / 2.0])
    X_all = None

    # seed for reproducibility
    np.random.seed(initial_seed)

    # First setting
    num_duels = 3 * input_dim
    noise_std = 1e-2

    X_init = (
        np.random.rand(2 * num_duels, input_dim) * (x_bounds[1] - x_bounds[0])
        + x_bounds[0]
    )
    y_init = test_func.values(X_init)

    X_init = X_init.reshape(num_duels, 2, input_dim)
    y_init = y_init.reshape(num_duels, 2, 1)
    sort_idx = np.argsort(y_init, axis=1)

    X_init_sort = np.take_along_axis(X_init, sort_idx, axis=1)[:, ::-1, :].reshape(
        num_duels, 2 * input_dim
    )

    kernel = GPy.kern.RBF(
        input_dim=input_dim,
        lengthscale=0.5 * (kernel_bounds[1] - kernel_bounds[0]) + kernel_bounds[0],
        variance=1,
        ARD=True,
    )
    results_path = func_name + "_results/"
    if "SynFun" in func_name:
        kernel = GPy.kern.RBF(
            input_dim=input_dim,
            lengthscale=test_func.ell * np.ones(input_dim),
            variance=1,
            ARD=True,
        )
        func_name = (
            func_name
            + "_ell="
            + str(test_func.ell)
            + "-d="
            + str(test_func.d)
            + "-seed"
            + str(test_func.seed)
        )
    # else:
    #     with open(results_path + 'kernel_params/rbf_kernel_params.pickle', 'rb') as f:
    #         kernel_param = pickle.load(f)
    #     kernel = GPy.kern.RBF(input_dim=input_dim, lengthscale= kernel_param, variance=1, ARD=True)
    # print("Ideal kernel params:", kernel_param)

    results_path = (
        func_name + "_results/" + BO_method + "/seed=" + str(initial_seed) + "/"
    )
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Brochu et al., 2010
    if BO_method == "LA-EI":
        model_name = "Laplace"
        first_acq = "max_mean"
        second_acq = "EI"
    # Sui et al., 2017
    elif BO_method == "EP-KSS":
        model_name = "EP"
        first_acq = "TS"
        second_acq = "TS"
    elif BO_method == "EP-EI":
        model_name = "EP"
        first_acq = "max_mean"
        second_acq = "EI"
    # proposed
    elif BO_method == "HB-EI":
        model_name = "Gibbs"
        first_acq = "current_best"
        second_acq = "EI"
    # proposed
    elif BO_method == "HB-UCB":
        model_name = "Gibbs"
        first_acq = "current_best"
        second_acq = "UCB"
    # proposed
    elif BO_method == "HB-BVEI":
        model_name = "Gibbs"
        first_acq = "current_best"
        second_acq = "BVEI"
    # proposed
    elif BO_method == "HB-BVUCB":
        model_name = "Gibbs"
        first_acq = "current_best"
        second_acq = "BVUCB"
    # proposed
    elif BO_method == "HB-MUC":
        model_name = "Gibbs"
        first_acq = "current_best"
        second_acq = "MUC"
    # check behavior
    elif BO_method == "MCMC-EI":
        model_name = "Gibbs"
        first_acq = "current_best"
        second_acq = "EI"
    # check behavior
    elif BO_method == "MCMC-BVEI":
        model_name = "Gibbs"
        first_acq = "current_best"
        second_acq = "BVEI"
    elif BO_method == "EP-MUC":
        model_name = "EP"
        first_acq = "max_mean"
        second_acq = "MUC"
    # Benavoli et al., 2020
    elif BO_method == "DuelTS":
        model_name = "Gibbs"
        first_acq = "current_best"
        second_acq = "TS"
    # Benavoli et al., 2020, 2021
    elif BO_method == "DuelUCB":
        model_name = "Gibbs"
        first_acq = "current_best"
        second_acq = "UCB"
    # Benavoli et al., 2020, 2021
    elif BO_method == "EIIG":
        model_name = "Gibbs"
        first_acq = "current_best"
        second_acq = "EIIG"
    # # Nielsen et al., 2014
    # elif BO_method=="BVEI":
    #     model_name = "Laplace"
    #     first_acq = "max_mean"
    #     second_acq = "BVEI"
    # Fauvel and Chalk, 2021
    # # Gonzalez et al., 2017
    # elif BO_method == "DualTS":
    #     model_name = "Laplace"
    #     first_acq = "TS"
    #     second_acq = "US"
    # Benavoli et al., 2020
    elif BO_method == "ESSDuelTS":
        model_name = "ESS"
        first_acq = "current_best"
        second_acq = "TS"
    # Benavoli et al., 2020, 2021
    elif BO_method == "ESSDuelUCB":
        model_name = "ESS"
        first_acq = "current_best"
        second_acq = "UCB"
    # Benavoli et al., 2020, 2021
    elif BO_method == "ESSEIIG":
        model_name = "ESS"
        first_acq = "current_best"
        second_acq = "EIIG"
    elif BO_method == "LArandom":
        model_name = "Laplace"
        first_acq = "random"
        second_acq = "random"
    elif BO_method == "GSrandom":
        model_name = "Gibbs"
        first_acq = "random"
        second_acq = "random"
    else:
        print("Specified method is not implemented")
        exit()

    # bayesian optimizer
    if model_name in ["Laplace", "EP"]:
        optimizer = PreferentialBO_GaussApprox(
            X_init_sort, x_bounds, kernel, kernel_bounds, noise_std, GPmodel=model_name
        )
    elif "DuelUCB" in BO_method or "EIIG" in BO_method or "MCMC" in BO_method:
        optimizer = PreferentialBO_MCMC(
            X_init_sort, x_bounds, kernel, kernel_bounds, noise_std, GPmodel=model_name
        )
    elif "Gibbs" == model_name:
        optimizer = PreferentialBO_HallucinationBeliever(
            X_init_sort, x_bounds, kernel, kernel_bounds, noise_std, GPmodel=model_name
        )

    if "MCMC" in BO_method:
        optimizer.GPmodel.thinning = 10
        optimizer.GPmodel.sample_size = 10000

    if "SynFun" not in func_name:
        # tuning hyperparameters
        optimizer.GPmodel.model_selection()

    Regret_list = list()
    computation_time_list = [0]

    ITR_MAX = 110
    if "SynFun" in func_name:
        ITR_MAX = 205

    if "random" in first_acq:
        X = (
            np.random.rand(2 * ITR_MAX, input_dim) * (x_bounds[1] - x_bounds[0])
            + x_bounds[0]
        )
        X = np.c_[X[:ITR_MAX, :], X[ITR_MAX:, :]]

    for i in range(ITR_MAX):
        print("-------------------------------------")
        print(str(i) + "th iteration")
        print("-------------------------------------")

        # compute next input
        start = time.time()
        if "random" in BO_method:
            x1, x2 = np.atleast_2d(X[i, :input_dim]), np.atleast_2d(X[i, input_dim:])
        else:
            if X_all is None:
                x1, x2 = optimizer.next_input(
                    first_acq=first_acq, second_acq=second_acq
                )
            else:
                x1, x2 = optimizer.next_input_pool(
                    first_acq=first_acq, second_acq=second_acq, X=X_all
                )

        print("Duel is {} {}".format(x1, x2))

        # evaluate preference
        new_output = test_func.values(np.r_[x1, x2])
        if new_output[0] > new_output[1]:
            optimizer.update(X_win=x1, X_loo=x2)
        else:
            optimizer.update(X_win=x2, X_loo=x1)

        # tuning hyperparameters
        if (i + 1) % 10 == 0 and "SynFun" not in func_name:
            optimizer.GPmodel.model_selection()

        computational_time = time.time() - start

        # compute recommendation point (without time measurement)
        if first_acq in ["current_best", "max_mean"]:
            inference_point = x1
        else:
            print("not implemented first acq")
            exit()

        Regret_list.append(test_func.values(np.atleast_2d(inference_point)).ravel())

        # save results
        with open(results_path + "Regret.pickle", "wb") as f:
            pickle.dump(np.array(Regret_list), f)

        with open(results_path + "computation_time.pickle", "wb") as f:
            pickle.dump(np.array(computation_time_list), f)

        # add computational time
        computation_time_list.append(computational_time)

        # output intermediate result
        if i % 5 == 0:
            print(
                "time {}, function value at recommendation point {}".format(
                    np.sum(computation_time_list), Regret_list[-1]
                )
            )

    with open(results_path + "obtained_X.pickle", "wb") as f:
        pickle.dump(optimizer.GPmodel.X, f)


if __name__ == "__main__":
    args = sys.argv
    BO_method = args[1]
    test_func = args[2]
    initial_seed = np.int(args[3])
    function_seed = np.int(args[4])
    parallel_num = np.int(args[5])
    options = [option for option in args if option.startswith("-")]

    os.environ["OMP_NUM_THREADS"] = "1"
    NUM_WORKER = 10
    if "SynFun" in test_func:
        NUM_WORKER = 10

    # When seed = -1, experiments of seed of 0-9 is done for parallel
    # When other seed is set, experiments of the seed is done
    if function_seed >= 0:
        if initial_seed >= 0:
            main((test_func, BO_method, initial_seed, function_seed, parallel_num))
            exit()

        function_seeds = [function_seed]
        initial_seeds = np.arange(NUM_WORKER).tolist()
    else:
        print("not implemented for multiple function seeds")
        # function_seeds = np.arange(NUM_WORKER).tolist()
        # if initial_seed < 0:
        #     initial_seeds = np.arange(NUM_WORKER).tolist()
        # else:
        #     initial_seeds = [initial_seed]

    params = list()
    for f_seed in function_seeds:
        for i_seed in initial_seeds:
            params.append((test_func, BO_method, i_seed, f_seed, parallel_num))
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKER) as executor:
        results = executor.map(main, params)
