# -*- coding: utf-8 -*-
# import os
# import sys
# import time
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib

matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from scipy.stats import norm
from scipy import special
from scipy import optimize
from scipy.stats import qmc
from scipy.special import owens_t

from .preferential_gp_regression import *


class PreferentialBO_core:
    __metaclass__ = ABCMeta

    def __init__(self, X, x_bounds, kernel, kernel_bounds, noise_std, GPmodel="Gibbs"):
        self.input_dim = np.shape(X)[1] // 2
        self.bounds = x_bounds
        self.bounds_list = x_bounds.T.tolist()
        self.initial_points_sampler = qmc.Sobol(d=self.input_dim, seed=0)

        if GPmodel == "Gibbs":
            self.GPmodel = PreferentialGP_Gibbs(
                X=X, kernel=kernel, kernel_bounds=kernel_bounds, noise_std=noise_std
            )
        elif GPmodel == "ESS":
            self.GPmodel = PreferentialGP_ESS(
                X=X, kernel=kernel, kernel_bounds=kernel_bounds, noise_std=noise_std
            )
        elif GPmodel == "Laplace":
            self.GPmodel = PreferentialGP_Laplace_test(
                X=X, kernel=kernel, kernel_bounds=kernel_bounds, noise_std=noise_std
            )
        elif GPmodel == "EP":
            self.GPmodel = PreferentialGP_EP(
                X=X, kernel=kernel, kernel_bounds=kernel_bounds, noise_std=noise_std
            )
        else:
            print("Specified model named {} is not implemented.".format(GPmodel))
            exit()

    def GP_mean(self, X):
        X = np.atleast_2d(X)
        mean, _ = self.GPmodel.predict(X)
        return -mean

    def posteriori_maximum(self):
        x0s = (
            self.initial_points_sampler.random(n=50 * self.input_dim)
            * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )

        x_max, _ = minimize(self.GP_mean, x0s, self.bounds_list)

        return x_max

    def update(self, X_win, X_loo):
        self.GPmodel.add_data(X_win, X_loo)


class PreferentialBO_GaussApprox(PreferentialBO_core):
    def __init__(
        self, X, x_bounds, kernel, kernel_bounds, noise_std, GPmodel="Laplace"
    ):
        super().__init__(X, x_bounds, kernel, kernel_bounds, noise_std, GPmodel)
        self.noise_constant = 2.0 * self.GPmodel.noise_std**2

    def next_input_pool(self, first_acq, second_acq, X):
        self.GPmodel.inference()
        X = np.atleast_2d(X)

        if first_acq == "max_mean":
            acquisition_values = self.GP_mean(X)
        elif first_acq == "TS":
            acquisition_values = self.TS(X)
        else:
            print(
                "Specified acquisition function named {} is not implemented".format(
                    first_acq
                )
            )
            exit()
        next_input1 = np.atleast_2d(X[np.argmin(acquisition_values)])
        X = X[~np.all(X == next_input1, axis=1), :]

        # preprocessing_for_second_acq()
        if second_acq == "EI":
            acquisition_values = self.EI(X)
        elif second_acq == "BVEI":
            acquisition_values = self.BVEI(X)
        elif second_acq == "MUC":
            acquisition_values = self.MUC(X)
        else:
            print(
                "Specified acquisition function named {} is not implemented".format(
                    second_acq
                )
            )
            exit()
        next_input2 = np.atleast_2d(X[np.argmin(acquisition_values)])

        return next_input1, next_input2

    def next_input(self, first_acq, second_acq):
        self.GPmodel.inference()
        x0s = (
            self.initial_points_sampler.random(n=20 * self.input_dim)
            * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )

        if first_acq == "max_mean":
            mean_train = self.GP_mean(self.GPmodel.flatten_X)
            min_idx = np.argmin(mean_train)
            next_input1 = self.GPmodel.flatten_X[min_idx]
            f_min1 = mean_train[min_idx]
            # next_input1, f_min1 = minimize(self.GP_mean, x0s, self.bounds_list)
        elif first_acq == "TS":
            self.GPmodel.sample(sample_size=1)
            next_input1, f_min1 = minimize(self.TS, x0s, self.bounds_list)
        else:
            print(
                "Specified acquisition function named {} is not implemented".format(
                    first_acq
                )
            )
            exit()
        x0s = np.r_[np.atleast_2d(next_input1), x0s]

        # preprocessing_for_second_acq()
        if second_acq == "EI":
            self.current_best_mean = -f_min1
            next_input2, f_min2 = minimize(self.EI, x0s, self.bounds_list)
        elif second_acq == "BVEI":
            self.x_1 = np.atleast_2d(next_input1)
            next_input2, f_min2 = minimize(self.BVEI, x0s, self.bounds_list)
        elif second_acq == "MUC":
            self.x_1 = np.atleast_2d(next_input1)
            next_input2, f_min2 = minimize(self.MUC, x0s, self.bounds_list)
        elif second_acq == "TS":
            self.GPmodel.sample(sample_size=1)
            next_input2, f_min2 = minimize(self.TS, x0s, self.bounds_list)
        else:
            print(
                "Specified acquisition function named {} is not implemented".format(
                    second_acq
                )
            )
            exit()

        # print(np.hstack(self.GPmodel.predict(np.r_[np.atleast_2d(next_input2), self.x_1], full_cov=True)))
        print("optimized acquisition function value:", -1 * f_min1, -1 * f_min2)

        return np.atleast_2d(next_input1), np.atleast_2d(next_input2)

    def TS(self, X):
        X = np.atleast_2d(X)
        sample_path_value = self.GPmodel.evaluate_sample(X)
        return -sample_path_value

    def EI(self, X):
        X = np.atleast_2d(X)
        mean, var = self.GPmodel.predict(X)
        std = np.sqrt(var)
        Z = (mean - self.current_best_mean) / std
        return -((Z * std) * normcdf(Z) + std * normpdf(Z)).ravel()

    """
    Only for X \in \RR^{1, d}
    """

    def BVEI(self, X):
        X = np.atleast_2d(X)
        mean, cov = self.GPmodel.predict(np.r_[X, self.x_1], full_cov=True)

        A = np.c_[[1, -1]]
        std = np.sqrt(A.T @ cov @ A)
        if std == 0:
            return np.array([0])
        else:
            mean = A.T @ mean
            Z = mean / std
            return -((Z * std) * normcdf(Z) + std * normpdf(Z)).ravel()

    def MUC(self, X):
        X = np.atleast_2d(X)
        mean, cov = self.GPmodel.predict(np.r_[X, self.x_1], full_cov=True)

        A = np.c_[[-1, 1]]
        var = A.T @ cov @ A / self.noise_constant
        mean = A.T @ mean / np.sqrt(self.noise_constant)

        tmp = mean / np.sqrt(1 + var)
        mean_win = normcdf(tmp)
        var_win = mean_win * (1 - mean_win) - 2 * owens_t(
            tmp, 1.0 / np.sqrt(1 + 2 * var)
        )
        return -var_win


class PreferentialBO_HallucinationBeliever(PreferentialBO_core):
    def __init__(self, X, x_bounds, kernel, kernel_bounds, noise_std, GPmodel="Gibbs"):
        super().__init__(X, x_bounds, kernel, kernel_bounds, noise_std, GPmodel)
        self.current_best = None
        self.noise_constant = 2.0 * self.GPmodel.noise_std**2
        self.iteration = 1
        self.beta_root = 2

    def next_input_pool(self, first_acq, second_acq, X):
        X = np.atleast_2d(X)

        if first_acq == "current_best":
            next_input1 = self.current_best
        else:
            self.GPmodel.inference(sample_size=1)
            if first_acq == "UCB":
                acquisition_values = self.UCB(X)
            elif first_acq == "TS":
                self.GPmodel.sample(sample_size=1)
                acquisition_values = self.TS(X)
            else:
                print(
                    "Specified acquisition function named {} is not implemented".format(
                        first_acq
                    )
                )
                exit()
            next_input1 = np.atleast_2d(X[np.argmin(acquisition_values)])
        X = X[~np.all(X == next_input1, axis=1), :]

        if second_acq == "UCB":
            self.GPmodel.inference(sample_size=1)
            acquisition_values = self.UCB(X)
        elif first_acq == "TS":
            self.GPmodel.inference(sample_size=1)
            self.GPmodel.sample(sample_size=1)
            acquisition_values = self.TS(X)
        else:
            print(
                "Specified acquisition function named {} is not implemented".format(
                    second_acq
                )
            )
            exit()
        next_input2 = np.atleast_2d(X[np.argmin(acquisition_values)])

        return next_input1, next_input2

    def next_input(self, first_acq, second_acq):
        x0s = (
            self.initial_points_sampler.random(n=20 * self.input_dim)
            * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )

        self.GPmodel.inference(sample_size=1)
        if first_acq == "current_best":
            if self.current_best is None:
                mean_train, _ = self.GPmodel.one_sample_conditioned_predict(
                    self.GPmodel.flatten_X
                )
                max_idx = np.argmax(mean_train)
                self.current_best = np.atleast_2d(self.GPmodel.flatten_X[max_idx])

            next_input1 = self.current_best
            f_min1 = 0
        # elif first_acq=="UCB":
        #     self.GPmodel.inference(sample_size=1)
        #     next_input1, f_min1 = minimize(self.UCB, x0s, self.bounds_list)
        # elif first_acq=="TS":
        #     self.GPmodel.inference(sample_size=1)
        #     self.GPmodel.sample(sample_size=1)
        #     next_input1, f_min1 = minimize(self.TS, x0s, self.bounds_list)
        else:
            print(
                "Specified acquisition function named {} is not implemented".format(
                    first_acq
                )
            )
            exit()

        x0s = np.r_[np.atleast_2d(self.current_best), x0s]
        self.x_1 = np.atleast_2d(next_input1)
        if second_acq == "UCB":
            next_input2, f_min2 = minimize(self.UCB, x0s, self.bounds_list)
        elif second_acq == "TS":
            self.GPmodel.sample(sample_size=1)
            next_input2, f_min2 = minimize(self.TS, x0s, self.bounds_list)
        elif second_acq == "EI":
            self.current_best_mean, _ = self.GPmodel.one_sample_conditioned_predict(
                next_input1
            )
            next_input2, f_min2 = minimize(self.EI, x0s, self.bounds_list)
        elif second_acq == "MUC":
            next_input2, f_min2 = minimize(self.MUC, x0s, self.bounds_list)
        elif second_acq == "BVUCB":
            next_input2, f_min2 = minimize(self.BVUCB, x0s, self.bounds_list)
        elif second_acq == "BVEI":
            next_input2, f_min2 = minimize(self.BVEI, x0s, self.bounds_list)
        else:
            print(
                "Specified acquisition function named {} is not implemented".format(
                    second_acq
                )
            )
            exit()

        print("optimized acquisition function value:", -1 * f_min2)

        return np.atleast_2d(next_input1), np.atleast_2d(next_input2)

    def TS(self, X):
        X = np.atleast_2d(X)
        sample_path_value = self.GPmodel.evaluate_sample(X)
        return -sample_path_value

    def UCB(self, X):
        X = np.atleast_2d(X)
        mean, var = self.GPmodel.one_sample_conditioned_predict(X)
        std = np.sqrt(var)
        return -(mean + self.beta_root * std).ravel()

    def EI(self, X):
        X = np.atleast_2d(X)
        mean, var = self.GPmodel.one_sample_conditioned_predict(X)
        if var <= 0:
            print(mean, var)
            exit()
        std = np.sqrt(var)
        Z = (mean - self.current_best_mean) / std
        return -((Z * std) * normcdf(Z) + std * normpdf(Z)).ravel()

    def PI(self, X):
        X = np.atleast_2d(X)
        mean, var = self.GPmodel.one_sample_conditioned_predict(X)
        return (self.f_max - mean) / np.sqrt(var)

    def update(self, X_win, X_loo):
        self.current_best = X_win
        self.iteration += 1
        super().update(X_win, X_loo)

    def MUC(self, X):
        X = np.atleast_2d(X)
        mean, cov = self.GPmodel.one_sample_conditioned_predict(
            np.r_[X, self.x_1], full_cov=True
        )

        A = np.c_[[-1, 1]]
        var = A.T @ cov @ A / self.noise_constant
        mean = A.T @ mean / np.sqrt(self.noise_constant)

        tmp = mean / np.sqrt(1 + var)
        mean_win = normcdf(tmp)
        var_win = mean_win * (1 - mean_win) - 2 * owens_t(
            tmp, 1.0 / np.sqrt(1 + 2 * var)
        )
        return -var_win

    def BVUCB(self, X):
        X = np.atleast_2d(X)
        mean, cov = self.GPmodel.one_sample_conditioned_predict(
            np.r_[X, self.x_1], full_cov=True
        )

        A = np.c_[[1, -1]]
        var = A.T @ cov @ A
        if var <= 0:
            var = 0
        mean = A.T @ mean

        UCB = mean + self.beta_root * np.sqrt(var)
        return -UCB.ravel()

    def BVEI(self, X):
        X = np.atleast_2d(X)
        mean, cov = self.GPmodel.one_sample_conditioned_predict(
            np.r_[X, self.x_1], full_cov=True
        )

        A = np.c_[[1, -1]]
        var = A.T @ cov @ A
        mean = A.T @ mean
        if var <= 0:
            return 0

        std = np.sqrt(var)
        Z = (mean - 0) / std
        return -((Z * std) * normcdf(Z) + std * normpdf(Z)).ravel()


class PreferentialBO_MCMC(PreferentialBO_core):
    def __init__(self, X, x_bounds, kernel, kernel_bounds, noise_std, GPmodel="Gibbs"):
        super().__init__(X, x_bounds, kernel, kernel_bounds, noise_std, GPmodel)
        self.current_best = None
        # hyperparameter for EIIG
        self.k = 0.1
        # hyperparameter for UCB
        self.gamma = 0.95
        self.idx = int(self.GPmodel.sample_size * self.gamma - 1)

    def next_input_pool(self, first_acq, second_acq, X):
        X = np.atleast_2d(X)

        self.GPmodel.inference()
        if first_acq == "current_best":
            next_input1 = self.current_best
        else:
            print(
                "Specified acquisition function named {} is not implemented".format(
                    first_acq
                )
            )
            exit()
        X = X[~np.all(X == next_input1, axis=1), :]

        if second_acq == "EIIG":
            self.GPmodel.sample()
            acquisition_values = self.EIIG(X)
        # elif second_acq=="TS":
        #     self.GPmodel.inference(sample_size=1)
        #     self.GPmodel.sample(sample_size=1)
        #     acquisition_values = self.TS(X)
        elif second_acq == "UCB":
            self.GPmodel.sample()
            acquisition_values = self.UCB(X)
        elif second_acq == "EI":
            self.GPmodel.sample()
            acquisition_values = self.EI(X)
        else:
            print(
                "Specified acquisition function named {} is not implemented".format(
                    second_acq
                )
            )
            exit()
        next_input2 = np.atleast_2d(X[np.argmin(acquisition_values)])

        return next_input1, next_input2

    def next_input(self, first_acq, second_acq):
        x0s = (
            self.initial_points_sampler.random(n=20 * self.input_dim)
            * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )

        self.GPmodel.inference()
        if first_acq == "current_best":
            if self.current_best is None:
                train_mean = self.GPmodel.mean(self.GPmodel.flatten_X)
                max_idx = np.argmax(train_mean)
                self.current_best = np.atleast_2d(self.GPmodel.flatten_X[max_idx])
            next_input1 = self.current_best
        else:
            print(
                "Specified acquisition function named {} is not implemented".format(
                    first_acq
                )
            )
            exit()

        x0s = np.r_[np.atleast_2d(self.current_best), x0s]
        if second_acq == "EIIG":
            self.GPmodel.sample()
            next_input2, f_min2 = minimize(self.EIIG, x0s, self.bounds_list)
        # elif second_acq=="TS":
        #     self.GPmodel.inference(sample_size=1)
        #     self.GPmodel.sample(sample_size=1)
        #     next_input2, f_min2 = minimize(self.TS, x0s, self.bounds_list)
        elif second_acq == "UCB":
            self.GPmodel.sample()
            next_input2, f_min2 = minimize(self.UCB, x0s, self.bounds_list)
        elif second_acq == "EI":
            self.current_best_mean = self.GPmodel.mean(next_input1)
            next_input2, f_min2 = minimize(self.EI, x0s, self.bounds_list)
        elif second_acq == "BVEI":
            self.x_1 = np.atleast_2d(next_input1)
            next_input2, f_min2 = minimize(self.BVEI, x0s, self.bounds_list)
        else:
            print(
                "Specified acquisition function named {} is not implemented".format(
                    second_acq
                )
            )
            exit()

        print("optimized acquisition function value:", -1 * f_min2)
        return np.atleast_2d(next_input1), np.atleast_2d(next_input2)

    def TS(self, X):
        X = np.atleast_2d(X)
        sample_path_value = self.GPmodel.evaluate_sample(X)
        return -sample_path_value

    def EI(self, X):
        mean, var = self.GPmodel.v_conditional_predict(X)
        std = np.sqrt(var)
        Z = (mean - self.current_best_mean) / std
        EI = np.mean((Z * std) * normcdf(Z) + std * normpdf(Z), axis=1).ravel()
        return -EI

    def BVEI(self, X):
        X = np.atleast_2d(X)
        mean, cov = self.GPmodel.v_conditional_predict(
            np.r_[X, self.x_1], full_cov=True
        )

        A = np.c_[[1, -1]]
        var = A.T @ cov @ A
        mean = A.T @ mean
        if var <= 0:
            return 0

        std = np.sqrt(var)
        Z = (mean - 0) / std
        return -np.mean((Z * std) * normcdf(Z) + std * normpdf(Z)).ravel()

    def UCB(self, X):
        X = np.r_[np.atleast_2d(X), np.atleast_2d(self.current_best)]
        sample_path_value = self.GPmodel.evaluate_sample(X)
        diff_samples = sample_path_value[0, :] - sample_path_value[1, :]

        UCB = np.partition(diff_samples, self.idx)[self.idx]
        return -UCB

    def EIIG(self, X):
        X = np.r_[np.atleast_2d(X), np.atleast_2d(self.current_best)]
        sample_path_value = self.GPmodel.evaluate_sample(X)
        diff_samples = sample_path_value[0, :] - sample_path_value[1, :]
        cdf_samples = normcdf(diff_samples / (np.sqrt(2) * self.GPmodel.noise_std))

        def h(prob):
            # if prob \approx 0 or 1, h(p) \approx 0
            prob = prob[np.logical_and(prob < 1, 0 < prob)]
            if np.size(prob) == 0:
                return 0
            else:
                one_minus_prob = 1 - prob
                return -prob * np.log(prob) - one_minus_prob * np.log(one_minus_prob)

        EI = np.mean(cdf_samples)
        IG = h(EI) - np.mean(h(cdf_samples))
        if EI == 0:
            return -(self.k * -32 + IG)
        else:
            return -(self.k * np.log(EI) + IG)

    def update(self, X_win, X_loo):
        self.current_best = X_win
        super().update(X_win, X_loo)


##################################################################################################
# util functions
##################################################################################################

root_two = np.sqrt(2)


def normcdf(x):
    return 0.5 * (1 + special.erf(x / root_two))


def normpdf(x):
    pdf = np.zeros(np.shape(x))
    small_x_idx = np.abs(x) < 50
    pdf[small_x_idx] = np.exp(-x[small_x_idx] ** 2 / 2) / (np.sqrt(2 * np.pi))
    return pdf


def minimize(func, start_points, bounds, jac="2-point"):
    x = np.copy(start_points)
    func_values = list()

    for i in range(np.shape(x)[0]):
        res = optimize.minimize(
            func,
            x0=x[i],
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxfun": 50},
            jac=jac,
        )
        func_values.append(res["fun"])
        x[i] = res["x"]

    min_index = np.argmin(func_values)
    return x[min_index], func_values[min_index]
