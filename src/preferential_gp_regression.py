# -*- coding: utf-8 -*-
import copy

import numpy as np

# from matplotlib import pyplot as plt
# from scipy.stats import truncnorm, mvn
from scipy.stats import norm, multivariate_normal
from scipy.stats import multivariate_normal
from scipy import optimize
from tqdm import tqdm
import GPy
from scipy import special
from scipy.stats import qmc

maxfeval = 3 * 1e2
jitter = 1e-8


class PreferentialGP_Laplace:
    """
    X_sort in \RR^{#duels \times 2 input_dim}: left side x is winner, right side x is looser

    Wei Chu and Zoubin Ghahramani. Preference learning with Gaussian processes. In Proceedings of the 22nd international conference on Machine learning, pages 137–144, 2005b.

    """

    def __init__(self, X, kernel, kernel_bounds, noise_std=1e-2):
        self.input_dim = np.shape(X)[1] // 2
        self.num_duels = np.shape(X)[0]
        self.noise_std = noise_std
        self.kernel = kernel
        self.kernel_bounds = kernel_bounds

        self.X = X
        self.flatten_X = np.r_[X[:, : self.input_dim], X[:, self.input_dim :]]
        self.winner_idx = np.arange(self.num_duels)
        self.looser_idx = np.arange(self.num_duels, 2 * self.num_duels)

        assert (
            np.shape(np.unique(self.flatten_X, axis=1))[0] == 2 * self.num_duels
        ), "Input has same duel points, so the current implementation for gradient of objective cannot be considered correctly"

        self.flatten_K_inv = None
        self.hessian_indicator = np.r_[
            np.c_[np.eye(self.num_duels), -1 * np.eye(self.num_duels)],
            np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)],
        ]
        self.initial_points_sampler = qmc.Sobol(d=self.input_dim, seed=0)

    def inference(self):
        # For numerical stability, we add a small value at diagonal elements
        self.flatten_K = self.kernel.K(self.flatten_X) + jitter * np.eye(
            2 * self.num_duels
        )
        self.flatten_K_inv = np.linalg.inv(self.flatten_K)
        self.f_map, self.covariance_map, Lambda = self.Laplace_inference()
        self.lambda_inv = np.linalg.inv(Lambda + jitter * np.eye(2 * self.num_duels))

    def inverse_mills_ratio(self, z):
        z_cdf = normcdf(z)
        cdf_nonzero_idx = z_cdf > 0
        # inverse mills ratio \approx -z if z <<<0
        inverse_mills_ratio = -z
        inverse_mills_ratio[cdf_nonzero_idx] = normpdf(z[cdf_nonzero_idx]) / (
            z_cdf[cdf_nonzero_idx]
        )
        return inverse_mills_ratio

    def Laplace_inference(self):
        new_f_map = np.zeros(2 * self.num_duels)
        z = np.zeros(self.num_duels)
        new_obj_val = np.inf

        for i in range(100):
            old_f_map = np.copy(new_f_map)
            old_obj_val = np.copy(new_obj_val)

            z = (old_f_map[self.winner_idx] - old_f_map[self.looser_idx]) / (
                np.sqrt(2) * self.noise_std
            )
            inverse_mills_ratio = self.inverse_mills_ratio(z)

            gradient = self._objective_gradient(old_f_map, inverse_mills_ratio)
            hess = self._objective_hessian(z, inverse_mills_ratio)
            update = np.linalg.solve(hess, np.c_[gradient]).ravel()

            new_f_map = old_f_map - update
            new_obj_val = self._objective(new_f_map, z)
            if np.max(np.abs(new_f_map - old_f_map)) <= 1e-5:
                break

            if np.any(np.abs(new_f_map) > 5):
                return np.nan, np.nan, np.nan

            z_tmp = (new_f_map[self.winner_idx] - new_f_map[self.looser_idx]) / (
                np.sqrt(2) * self.noise_std
            )

        assert i < 100, "error in Laplace approximation may be large"
        f_map = np.c_[new_f_map]
        # beta = self.flatten_K_inv @ f_map
        z = (f_map[self.winner_idx] - f_map[self.looser_idx]) / (
            np.sqrt(2) * self.noise_std
        )
        inverse_mills_ratio = self.inverse_mills_ratio(z)

        Lambda = self._lambda(z, inverse_mills_ratio)
        covariance_map = np.linalg.inv(self.flatten_K_inv + Lambda)

        return f_map, covariance_map, Lambda

    def _objective(self, f, z):
        S = -np.sum(norm.logcdf(z)) + np.c_[f].T @ self.flatten_K_inv @ np.c_[f] / 2.0
        return S.ravel()

    def _objective_gradient(self, f, inverse_mills_ratio):
        tmp_gradient = inverse_mills_ratio / (np.sqrt(2) * self.noise_std)
        first_term_gradient = np.r_[-1 * tmp_gradient, tmp_gradient]
        second_term_gradient = self.flatten_K_inv @ np.c_[f]
        return first_term_gradient + second_term_gradient.ravel()

    def _objective_hessian(self, z, inverse_mills_ratio):
        Lambda = self._lambda(z, inverse_mills_ratio)
        return self.flatten_K_inv + Lambda

    def _lambda(self, z, inverse_mills_ratio):
        tmp_hessian = (inverse_mills_ratio**2 + z * inverse_mills_ratio) / (
            2 * self.noise_std**2
        )
        hessian_first_term = (
            self.hessian_indicator * np.c_[np.r_[tmp_hessian, tmp_hessian]]
        )
        return hessian_first_term

    def predict(self, X, full_cov=False):
        cov_train_test = self.kernel.K(self.flatten_X, X)
        posterior_mean = cov_train_test.T @ self.flatten_K_inv @ self.f_map

        if full_cov:
            cov_test = self.kernel.K(X, X)
            posterior_cov = (
                cov_test
                - cov_train_test.T
                @ np.linalg.inv(self.flatten_K + self.lambda_inv)
                @ cov_train_test
            )
            return posterior_mean, posterior_cov
        else:
            var_test = self.kernel.Kdiag(X)
            posterior_var = var_test - np.einsum(
                "ij,jk,ki->i",
                cov_train_test.T,
                np.linalg.inv(self.flatten_K + self.lambda_inv),
                cov_train_test,
            )

            return posterior_mean, np.c_[posterior_var]

    def minus_log_likelihood(self, kernel_params=None, jac=False):
        if kernel_params is None:
            kernel = copy.copy(self.kernel)
        else:
            kernel_params = np.atleast_2d(kernel_params)
            kernel = GPy.kern.RBF(
                input_dim=self.input_dim, lengthscale=kernel_params, ARD=True
            )
        # if self.flatten_K_inv is not None:
        #     copy_flatten_K_inv = copy.copy(self.flatten_K_inv)
        # else:
        #     copy_flatten_K_inv = None

        flatten_K = kernel.K(self.flatten_X) + jitter * np.eye(2 * self.num_duels)
        # flatten_K = kernel.K(self.flatten_X)
        self.flatten_K_inv = np.linalg.inv(flatten_K)
        f_map, _, Lambda = self.Laplace_inference()

        if np.any(np.isnan(f_map)):
            return np.nan

        z = (f_map[self.winner_idx] - f_map[self.looser_idx]) / (
            np.sqrt(2) * self.noise_std
        )
        # inverse_mills_ratio = self.inverse_mills_ratio(z)
        flatten_K_Lambda = flatten_K @ Lambda
        minus_log_likelihood = (
            self._objective(f_map, z)
            + 0.5 * np.log(np.linalg.det(flatten_K_Lambda + np.eye(2 * self.num_duels)))
        )[0]

        return minus_log_likelihood

    def model_selection(self, num_start_points=10):
        x = self.initial_points_sampler.random(n=num_start_points - 1) * (
            self.kernel_bounds[1] - self.kernel_bounds[0]
        )
        x = np.r_[x, np.atleast_2d(self.kernel.lengthscale.values)]

        def wrapper_minus_log_likelihood(kernel_params):
            return self.minus_log_likelihood(kernel_params=kernel_params, jac=False)

        func_values = list()
        for i in range(np.shape(x)[0]):
            res = optimize.minimize(
                wrapper_minus_log_likelihood,
                x0=x[i],
                bounds=np.array(self.kernel_bounds).T.tolist(),
                method="L-BFGS-B",
                options={"ftol": 0.1},
                jac="2-point",
            )
            func_values.append(res["fun"])
            x[i] = res["x"]

        min_index = np.argmin(func_values)
        print("Selected kernel lengthscales: {}".format(x[min_index]))
        self.kernel = GPy.kern.RBF(
            input_dim=self.input_dim, lengthscale=x[min_index], ARD=True
        )
        # self.inference()

    def sample(self, sample_size=1):
        self.RFF = RFF_RBF(
            input_dim=self.input_dim, lengthscales=self.kernel.lengthscale.values
        )
        self.coefficient = np.random.randn(self.RFF.basis_dim, sample_size)

        f_sample = (
            np.linalg.cholesky(self.covariance_map)
            @ np.random.randn(2 * self.num_duels, sample_size)
            + self.f_map
        )
        flattenX_transform = self.RFF.transform(self.flatten_X)
        f_prior_flattenX = flattenX_transform @ self.coefficient

        self.K_inv_f_sample = self.flatten_K_inv @ (f_sample - f_prior_flattenX)
        pass

    def evaluate_sample(self, X):
        K_X_flattenX = self.kernel.K(X, self.flatten_X)

        X_transform = self.RFF.transform(X)
        f_X_samples_prior = X_transform @ self.coefficient
        f_X_samples = f_X_samples_prior + K_X_flattenX @ self.K_inv_f_sample
        return f_X_samples

    def add_data(self, X_win, X_loo):
        X_win = np.atleast_2d(X_win)
        X_loo = np.atleast_2d(X_loo)
        assert np.shape(X_win) == np.shape(
            X_loo
        ), "Shapes of winner and looser in added data do not match"
        self.X = np.r_[self.X, np.c_[X_win, X_loo]]
        self.num_duels = self.num_duels + np.shape(X_win)[0]
        self.flatten_X = np.r_[self.X[:, : self.input_dim], self.X[:, self.input_dim :]]
        self.winner_idx = np.arange(self.num_duels)
        self.looser_idx = np.arange(self.num_duels, 2 * self.num_duels)
        self.hessian_indicator = np.r_[
            np.c_[np.eye(self.num_duels), -1 * np.eye(self.num_duels)],
            np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)],
        ]

        assert (
            np.shape(np.unique(self.flatten_X, axis=1))[0] == 2 * self.num_duels
        ), "Input has same duel points, so the current implementation for gradient of objective cannot be considered correctly"


class PreferentialGP_EP(PreferentialGP_Laplace):
    """
    X_sort in \RR^{#duels \times 2 input_dim}: left side x is winner, right side x is looser
    """

    def __init__(self, X, kernel, kernel_bounds, noise_std=1e-2):
        self.input_dim = np.shape(X)[1] // 2
        self.num_duels = np.shape(X)[0]
        self.noise_std = noise_std
        self.kernel = kernel
        self.kernel_bounds = kernel_bounds

        # for LP inference
        self.winner_idx = np.arange(self.num_duels)
        self.looser_idx = np.arange(self.num_duels, 2 * self.num_duels)
        self.hessian_indicator = np.r_[
            np.c_[np.eye(self.num_duels), -1 * np.eye(self.num_duels)],
            np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)],
        ]
        self.flatten_K_inv = None

        self.A = np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)]
        self.X = X
        self.flatten_X = np.r_[X[:, : self.input_dim], X[:, self.input_dim :]]
        self.initial_points_sampler = qmc.Sobol(d=self.input_dim, seed=0)

    def inference(self):
        self.flatten_K = self.kernel.K(self.flatten_X) + self.noise_std**2 * np.eye(
            2 * self.num_duels
        )
        self.K_v = self.A @ self.flatten_K @ self.A.T
        self.K_v_inv = np.linalg.inv(self.K_v)

        self.mu_TN, self.sigma_TN, self.mu_tilde, self.sigma_tilde = ep_orthants_tmvn(
            upper=np.zeros(self.num_duels),
            mu=np.zeros(self.num_duels),
            sigma=self.K_v,
            L=self.num_duels,
        )
        self.sigma_plus_sigma_tilde = self.K_v + np.diag(self.sigma_tilde)
        self.sigma_plus_sigma_tilde_inv = np.linalg.inv(self.sigma_plus_sigma_tilde)
        self.sigma_plus_sigma_tilde_inv_mu_tilde = (
            self.sigma_plus_sigma_tilde_inv.T @ self.mu_tilde
        )

    def predict(self, X, full_cov=False):
        X = np.atleast_2d(X)
        test_point_size = np.shape(X)[0]
        transform_matrix = np.r_[
            np.c_[np.eye(test_point_size), np.zeros((test_point_size, self.num_duels))],
            np.c_[np.zeros((2 * self.num_duels, test_point_size)), self.A.T],
        ]

        cov_X_flattenX = self.kernel.K(X, self.flatten_X)
        K_X_flattenX = np.c_[
            np.r_[
                self.kernel.K(X) + jitter * np.eye(test_point_size), cov_X_flattenX.T
            ],
            np.r_[cov_X_flattenX, self.flatten_K],
        ]

        K_X_v = (
            transform_matrix.T[test_point_size:, :]
            @ K_X_flattenX
            @ transform_matrix[:, :test_point_size]
        )

        tmp = K_X_v.T @ self.sigma_plus_sigma_tilde_inv
        mean = K_X_v.T @ self.sigma_plus_sigma_tilde_inv_mu_tilde
        if full_cov:
            cov = self.kernel.K(X) - tmp @ K_X_v
            return mean, cov
        else:
            var = self.kernel.variance.values - np.einsum("ij,ji->i", tmp, K_X_v)
            return mean, var

    def minus_log_likelihood_ep(self, kernel_params):
        kernel = GPy.kern.RBF(
            input_dim=self.input_dim, lengthscale=kernel_params, ARD=True
        )

        ### inference ###########################################
        flatten_K = kernel.K(self.flatten_X) + self.noise_std**2 * np.eye(
            2 * self.num_duels
        )
        K_v = self.A @ flatten_K @ self.A.T
        # K_v_inv = np.linalg.inv(K_v)

        _, sigma_TN, mu_tilde, sigma_tilde = ep_orthants_tmvn(
            upper=np.zeros(self.num_duels),
            mu=np.zeros(self.num_duels),
            sigma=K_v,
            L=self.num_duels,
        )
        # sigma_plus_sigma_tilde = K_v + np.diag(sigma_tilde)
        # sigma_plus_sigma_tilde_inv = np.linalg.inv(sigma_plus_sigma_tilde)
        # sigma_plus_sigma_tilde_inv_mu_tilde = sigma_plus_sigma_tilde_inv.T @ mu_tilde
        ##########################################################33
        tmp_vec = mu_tilde / sigma_tilde

        _, original_logdet = np.linalg.slogdet(K_v)
        _, ep_logdet = np.linalg.slogdet(sigma_TN)
        log_likelihood = 0.5 * (
            -original_logdet
            + ep_logdet
            + np.c_[tmp_vec].T @ sigma_TN @ np.c_[tmp_vec]
            - np.c_[tmp_vec].T @ np.c_[mu_tilde]
        )

        return -log_likelihood

    def sample(self, sample_size=1):
        self.RFF = RFF_RBF(
            input_dim=self.input_dim, lengthscales=self.kernel.lengthscale.values
        )
        self.coefficient = np.random.randn(self.RFF.basis_dim, sample_size)

        v_sample = (
            np.linalg.cholesky(self.sigma_TN)
            @ np.random.randn(self.num_duels, sample_size)
            + self.mu_TN
        )
        flattenX_transform = self.RFF.transform(self.flatten_X)
        f_prior_flattenX = flattenX_transform @ self.coefficient
        f_prior_v = self.A @ f_prior_flattenX

        self.K_inv_f_sample = self.K_v_inv @ (v_sample - f_prior_v)
        pass

    def evaluate_sample(self, X):
        X = np.atleast_2d(X)
        test_point_size = np.shape(X)[0]
        transform_matrix = np.r_[
            np.c_[np.eye(test_point_size), np.zeros((test_point_size, self.num_duels))],
            np.c_[np.zeros((2 * self.num_duels, test_point_size)), self.A.T],
        ]

        cov_X_flattenX = self.kernel.K(X, self.flatten_X)
        K_X_flattenX = np.c_[
            np.r_[
                self.kernel.K(X) + jitter * np.eye(test_point_size), cov_X_flattenX.T
            ],
            np.r_[cov_X_flattenX, self.flatten_K],
        ]

        K_X_v = (
            transform_matrix.T[test_point_size:, :]
            @ K_X_flattenX
            @ transform_matrix[:, :test_point_size]
        )

        X_transform = self.RFF.transform(X)
        f_X_samples_prior = X_transform @ self.coefficient
        f_X_samples = f_X_samples_prior + K_X_v @ self.K_v_inv_v_sample
        return f_X_samples

    def add_data(self, X_win, X_loo):
        X_win = np.atleast_2d(X_win)
        X_loo = np.atleast_2d(X_loo)
        assert np.shape(X_win) == np.shape(
            X_loo
        ), "Shapes of winner and looser in added data do not match"
        self.X = np.r_[self.X, np.c_[X_win, X_loo]]
        self.num_duels = self.num_duels + np.shape(X_win)[0]
        self.flatten_X = np.r_[self.X[:, : self.input_dim], self.X[:, self.input_dim :]]
        self.winner_idx = np.arange(self.num_duels)
        self.looser_idx = np.arange(self.num_duels, 2 * self.num_duels)
        self.hessian_indicator = np.r_[
            np.c_[np.eye(self.num_duels), -1 * np.eye(self.num_duels)],
            np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)],
        ]
        self.A = np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)]

        assert (
            np.shape(np.unique(self.flatten_X, axis=1))[0] == 2 * self.num_duels
        ), "Input has same duel points, so the current implementation for gradient of objective cannot be considered correctly"


class PreferentialGP_Gibbs(PreferentialGP_Laplace):
    """
    X_sort in \RR^{#duels \times 2 input_dim}: left side x is winner, right side x is looser

    If sampling = GIbbs, then we use Gibbs sampling
        YIFANG LI AND SUJIT K. GHOSH Efficient Sampling Methods for Truncated Multivariate Normal and Student-t Distributions Subject to Linear Inequality Constraints, Journal of Statistical Theory and Practice, 9:712–732, 2015
        Christian P Robert. Simulation of truncated normal variables. Statistics and computing, 5(2):121–125, 1995.
        Sébastien Da Veiga and Amandine Marrel. Gaussian process modeling with inequality constraints, 2012.
        Sébastien Da Veiga and Amandine Marrel. Gaussian process regression with linear inequality constraints. Reliability Engineering & System Safety, 195:106732, 2020.
    """

    def __init__(
        self,
        X,
        kernel,
        kernel_bounds,
        noise_std=1e-2,
        sample_size=1000,
        burn_in=1000,
        thinning=1,
        sampling="Gibbs",
    ):
        self.input_dim = np.shape(X)[1] // 2
        self.num_duels = np.shape(X)[0]
        self.noise_std = noise_std
        self.kernel = kernel
        self.kernel_bounds = kernel_bounds
        self.sampling_method = sampling

        self.sample_size = sample_size
        self.burn_in = burn_in
        self.thinning = thinning

        # for LP inference
        self.winner_idx = np.arange(self.num_duels)
        self.looser_idx = np.arange(self.num_duels, 2 * self.num_duels)
        self.hessian_indicator = np.r_[
            np.c_[np.eye(self.num_duels), -1 * np.eye(self.num_duels)],
            np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)],
        ]
        self.flatten_K_inv = None

        self.A = np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)]
        self.X = X
        self.flatten_X = np.r_[X[:, : self.input_dim], X[:, self.input_dim :]]

        self.initial_sample = None
        self.v_sample = None
        self.initial_points_sampler = qmc.Sobol(d=self.input_dim, seed=0)

    def inference(self, sample_size=None):
        self.flatten_K = self.kernel.K(self.flatten_X) + self.noise_std**2 * np.eye(
            2 * self.num_duels
        )
        self.K_v = self.A @ self.flatten_K @ self.A.T
        self.K_v_inv = np.linalg.inv(self.K_v)

        if sample_size is None:
            sample_size = self.sample_size

        # sampling from truncated multivariate normal
        if self.sampling_method == "Gibbs":
            self.v_sample = orthants_MVN_Gibbs_sampling(
                dim=self.num_duels,
                cov_inv=self.K_v_inv,
                burn_in=self.burn_in,
                thinning=self.thinning,
                sample_size=sample_size,
                initial_sample=self.initial_sample,
            )

        self.v_mean = np.mean(self.v_sample, axis=1)

    # sampling sample path from prior (Wilson et al., 2020, ICML)
    def sample(self, sample_size=None):
        if sample_size is None:
            sample_size = self.sample_size
        self.RFF = RFF_RBF(
            input_dim=self.input_dim, lengthscales=self.kernel.lengthscale.values
        )
        self.coefficient = np.random.randn(self.RFF.basis_dim, sample_size)

        flattenX_transform = self.RFF.transform(self.flatten_X)
        f_prior_flattenX = flattenX_transform @ self.coefficient
        f_prior_v = self.A @ f_prior_flattenX
        self.K_v_inv_v_sample = self.K_v_inv @ (self.v_sample - f_prior_v)

    def evaluate_sample(self, X):
        X = np.atleast_2d(X)
        K_X_v = self._covariance_X_v(X)

        X_transform = self.RFF.transform(X)
        f_X_prior = X_transform @ self.coefficient

        return f_X_prior + K_X_v.T @ self.K_v_inv_v_sample

    def _covariance_X_v(self, X):
        X = np.atleast_2d(X)
        test_point_size = np.shape(X)[0]

        transform_matrix = np.r_[
            np.c_[np.eye(test_point_size), np.zeros((test_point_size, self.num_duels))],
            np.c_[np.zeros((2 * self.num_duels, test_point_size)), self.A.T],
        ]
        cov_X_flattenX = self.kernel.K(X, self.flatten_X)
        K_X_flattenX = np.c_[
            np.r_[
                self.kernel.K(X) + jitter * np.eye(test_point_size), cov_X_flattenX.T
            ],
            np.r_[cov_X_flattenX, self.flatten_K],
        ]

        return (
            transform_matrix.T[test_point_size:, :]
            @ K_X_flattenX
            @ transform_matrix[:, :test_point_size]
        )

    def one_sample_conditioned_predict(self, X, full_cov=False):
        K_X_v = self._covariance_X_v(X)

        tmp = K_X_v.T @ self.K_v_inv
        mean = tmp @ np.c_[self.v_sample[:, 0]]
        if full_cov:
            cov = self.kernel.K(X) - tmp @ K_X_v
            return mean, cov
        else:
            var = self.kernel.variance.values - np.einsum("ij,ji->i", tmp, K_X_v)
            return mean, var

    # return Pr(x1 > x2)
    def win_prob(self, x1, x2):
        X = np.r_[np.atleast_2d(x1), np.atleast_2d(x2)]
        K_X_v = self._covariance_X_v(X)

        tmp = K_X_v.T @ self.K_v_inv
        mean = tmp @ np.c_[self.v_sample]
        cov = self.kernel.K(X) - tmp @ K_X_v

        duel_mean = mean[1, :] - mean[0, :]
        duel_std = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
        return np.mean(normcdf(duel_mean / duel_std))

    def pdf(self, x1, x2, f):
        X = np.r_[np.atleast_2d(x1), np.atleast_2d(x2)]
        K_X_v = self._covariance_X_v(X)

        tmp = K_X_v.T @ self.K_v_inv
        mean = (tmp @ np.c_[self.v_sample]).T
        cov = self.kernel.K(X) - tmp @ K_X_v
        return np.mean(multivariate_normal.pdf(f - mean, cov=cov))

    def v_conditional_predict(self, X, full_cov=False):
        X = np.r_[np.atleast_2d(X)]
        K_X_v = self._covariance_X_v(X)

        tmp = K_X_v.T @ self.K_v_inv
        mean = tmp @ np.c_[self.v_sample]
        if full_cov:
            cov = self.kernel.K(X, X) - tmp @ K_X_v
            return mean, cov
        else:
            var = self.kernel.variance - np.einsum("ij,ji->i", tmp, K_X_v)
            return mean, np.c_[var]

    def mean(self, X):
        K_X_v = self._covariance_X_v(X)
        return K_X_v.T @ self.K_v_inv @ np.c_[self.v_mean]

    def quantile(self, X, prob=0.5):
        X = np.atleast_2d(X)
        test_point_size = np.shape(X)[0]

        K_X_v = self._covariance_X_v(X)
        tmp = K_X_v.T @ self.K_v_inv
        std = np.sqrt(self.kernel.variance.values - np.einsum("ij,ji->i", tmp, K_X_v))
        mean = (tmp @ self.v_sample).T

        cons = normal_cdf_inverse(prob)
        interval = cons * std
        quantile_lower = np.min(mean, axis=0) + interval
        quantile_upper = np.max(mean, axis=0) + interval

        evaluate_f = np.ones(test_point_size)
        center_cdf = np.ones(test_point_size)
        unconverged_idx = np.arange(test_point_size)

        while np.any(unconverged_idx):
            evaluate_f[unconverged_idx] = (
                quantile_upper[unconverged_idx] - quantile_lower[unconverged_idx]
            ) / 2.0 + quantile_lower[unconverged_idx]

            center_cdf[unconverged_idx] = np.mean(
                normcdf(
                    (evaluate_f[unconverged_idx] - mean[:, unconverged_idx])
                    / std[unconverged_idx]
                ),
                axis=0,
            )

            tmp_idx = center_cdf[unconverged_idx] >= prob
            cdf_large_idx = unconverged_idx[tmp_idx]
            cdf_small_idx = unconverged_idx[np.logical_not(tmp_idx)]

            quantile_lower[cdf_small_idx] = evaluate_f[cdf_small_idx]
            quantile_upper[cdf_large_idx] = evaluate_f[cdf_large_idx]

            unconverged_idx = unconverged_idx[
                np.abs(center_cdf[unconverged_idx] - prob) > 1e-4
            ]
        return evaluate_f

    def add_data(self, X_win, X_loo):
        X_win = np.atleast_2d(X_win)
        X_loo = np.atleast_2d(X_loo)
        assert np.shape(X_win) == np.shape(
            X_loo
        ), "Shape of winner and looser in added data does not match"
        self.X = np.r_[self.X, np.c_[X_win, X_loo]]
        self.num_duels = self.num_duels + np.shape(X_win)[0]
        self.flatten_X = np.r_[self.X[:, : self.input_dim], self.X[:, self.input_dim :]]
        self.A = np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)]

        self.initial_sample = None
        self.v_sample = None

        # for LP inference
        self.winner_idx = np.arange(self.num_duels)
        self.looser_idx = np.arange(self.num_duels, 2 * self.num_duels)
        self.hessian_indicator = np.r_[
            np.c_[np.eye(self.num_duels), -1 * np.eye(self.num_duels)],
            np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)],
        ]


############################################################################################
# util functions and classes
############################################################################################


def orthants_MVN_Gibbs_sampling(
    dim, cov_inv, burn_in=500, thinning=1, sample_size=1000, initial_sample=None
):
    if initial_sample is None:
        sample_chain = np.zeros((dim, 1))
    else:
        assert initial_sample.shape == (
            dim,
            1,
        ), "Shape of initial sample of Gibbs sampling is not (dim, 1)"
        sample_chain = initial_sample

    conditional_std = 1 / np.sqrt(np.diag(cov_inv))
    scaled_cov_inv = cov_inv / np.c_[np.diag(cov_inv)]
    sample_list = []
    for i in range((burn_in + thinning * (sample_size - 1)) * dim):
        j = i % dim
        conditional_mean = sample_chain[j] - scaled_cov_inv[j] @ sample_chain
        sample_chain[j] = (
            -1
            * one_side_trunc_norm_sampling(
                lower=conditional_mean[0] / conditional_std[j]
            )
            * conditional_std[j]
            + conditional_mean[0]
        )

        if ((i + 1) - burn_in * dim) % (
            dim * thinning
        ) == 0 and i + 1 - burn_in * dim >= 0:
            sample_list.append(sample_chain.copy())

    samples = np.hstack(sample_list)
    return samples


a_zero = 0.2570


def trunc_norm_sampling(lower=None, upper=None, mean=0, std=1):
    """
    See Sec.2.1 in YIFANG LI AND SUJIT K. GHOSH Efficient Sampling Methods for Truncated Multivariate Normal and Student-t Distributions Subject to Linear Inequality Constraints, Journal of Statistical Theory and Practice, 9:712–732, 2015
            Christian P Robert. Simulation of truncated normal variables. Statistics and computing, 5(2):121–125, 1995.
    """
    if lower is None and upper is None:
        return np.random.randn(1) * std + mean
    elif lower is None:
        upper = (upper - mean) / std
        return -1 * one_side_trunc_norm_sampling(lower=-upper) * std + mean
    elif upper is None:
        lower = (lower - mean) / std
        return one_side_trunc_norm_sampling(lower=lower) * std + mean
    elif lower <= 0 and 0 < upper:
        lower = (lower - mean) / std
        upper = (upper - mean) / std
        return (
            two_sided_trunc_norm_sampling_zero_containing(lower=lower, upper=upper)
            * std
            + mean
        )
    elif 0 <= lower:
        lower = (lower - mean) / std
        upper = (upper - mean) / std
        return (
            two_sided_trunc_norm_sampling_positive_lower(lower=lower, upper=upper) * std
            + mean
        )
    elif upper <= 0:
        lower = (lower - mean) / std
        upper = (upper - mean) / std
        return (
            -1
            * two_sided_trunc_norm_sampling_positive_lower(lower=-upper, upper=-lower)
            * std
            + mean
        )


def one_side_trunc_norm_sampling(lower=None):
    if lower > a_zero:
        alpha = (lower + np.sqrt(lower**2 + 4)) / 2.0
        while True:
            z = np.random.exponential(alpha) + lower
            rho_z = np.exp(-((z - alpha) ** 2) / 2.0)
            u = np.random.rand(1)
            if u <= rho_z:
                return z
    elif lower >= 0:
        while True:
            z = np.abs(np.random.randn(1))
            if lower <= z:
                return z
    else:
        while True:
            z = np.random.randn(1)
            if lower <= z:
                return z


def two_sided_trunc_norm_sampling_zero_containing(lower, upper):
    if upper <= lower * np.sqrt(2 * np.pi):
        M = 1.0 / np.sqrt(
            2 * np.pi
        )  # / (normcdf(upper) - normcdf(lower)) * (upper - lower)
        while True:
            z = np.random.rand(1) * (upper - lower) + lower
            u = np.random.rand(1)
            if u <= normpdf(z) / M:
                return z
    else:
        while True:
            z = np.random.randn(1)
            if lower <= z and z <= upper:
                return z


def two_sided_trunc_norm_sampling_positive_lower(lower, upper):
    if lower < a_zero:
        b_1_a = lower + np.sqrt(np.pi / 2.0) * np.exp(lower**2 / 2.0)
        if upper <= b_1_a:
            M = normpdf(
                lower
            )  # / (normcdf(upper) - normcdf(lower)) # * (upper - lower)
            while True:
                z = np.random.rand(1) * (upper - lower) + lower
                u = np.random.rand(1)
                if u <= normpdf(z) / M:
                    return z
        else:
            while True:
                z = np.abs(np.random.randn(1))
                if lower <= z and z <= upper:
                    return z
    else:
        tmp = np.sqrt(lower**2 + 4)
        b_2_a = lower + 2 / (lower + tmp) * np.exp(
            (lower**2 - lower * tmp) / 4.0 + 0.5
        )
        if upper <= b_2_a:
            M = normpdf(
                lower
            )  # / (normcdf(upper) - normcdf(lower)) # * (upper - lower)
            while True:
                z = np.random.rand(1) * (upper - lower) + lower
                u = np.random.rand(1)
                if u <= normpdf(z) / M:
                    return z
        else:
            alpha = (lower + np.sqrt(lower**2 + 4)) / 2.0
            while True:
                z = np.random.exponential(alpha) + lower
                if z <= upper:
                    rho_z = np.exp(-((z - alpha) ** 2) / 2.0)
                    u = np.random.rand(1)
                    if u <= rho_z:
                        return z


def ep_orthants_tmvn(upper, mu, sigma, L):
    """
    return means and covariance matrices of truncated multi-variate normal distribution truncated with truncation < upper.

    Parameters
    ----------
    mu : numpy array
        mean of original L-dimentional multi-variate nomal (L)
    sigma : numpy array
        covariance matrix of L-dimentional original multi-variate normal (L \times L)
    upper : numpy array
        upper position of M cells in region truncated by pareto frontier (L)

    Returns
    -------
    mu_TN : numpy array
        means of truncated multi-variate normal approximated by EP (L)
    sigma_TN : numpy array
        covariance matrix of truncated multi-variate normal approximated by EP (L \times L)
    """
    mu_tilde = np.zeros(L)
    sigma_tilde = np.inf * np.ones(L)
    mu_TN = mu
    sigma_TN = sigma
    mu_TN_before = mu
    sigma_TN_before = sigma

    sigma_inv = np.linalg.inv(sigma)
    sigma_inv_mu = sigma_inv.dot(mu)

    for i in range(1000):
        for j in range(L):
            sigma_bar = 1.0 / (1.0 / sigma_TN[j, j] - 1.0 / sigma_tilde[j])
            mu_bar = sigma_bar * (
                mu_TN[j] / sigma_TN[j, j] - mu_tilde[j] / sigma_tilde[j]
            )

            beta = (upper[j] - mu_bar) / np.sqrt(sigma_bar)
            Z = normcdf(beta)
            beta_pdf = normpdf(beta)
            diff_pdf = -beta_pdf
            diff_pdf_product = -beta * beta_pdf

            gamma = diff_pdf_product / Z - (diff_pdf / Z) ** 2
            if gamma == 0:
                # If gamma = 0, we can interprete that there is no condition
                sigma_tilde[j] = np.inf
                mu_tilde[j] = 0
            else:
                sigma_tilde[j] = -(1.0 / gamma + 1) * sigma_bar
                mu_tilde[j] = mu_bar - 1.0 / gamma * (diff_pdf / Z) * np.sqrt(sigma_bar)

            sigma_tilde_inv = np.diag(1.0 / sigma_tilde)
            sigma_TN = np.linalg.inv(sigma_tilde_inv + sigma_inv)
            mu_TN = sigma_TN.dot(sigma_tilde_inv.dot(mu_tilde) + sigma_inv_mu)

        change = np.max(
            [
                np.max(np.abs(sigma_TN - sigma_TN_before)),
                np.max(np.abs(mu_TN - mu_TN_before)),
            ]
        )
        sigma_TN_before = sigma_TN
        mu_TN_before = mu_TN

        if np.isnan(change):
            print("iteration :", i)
            print("mu", mu)
            print("sigma", sigma)
            print("upper", upper)
            print("mu_TN", mu_TN)
            print("sigma_TN", sigma_TN)
            print(gamma)
            exit()

        if change < 1e-8:
            # print('iteration :', i)
            break

    return mu_TN, sigma_TN, mu_tilde, sigma_tilde


class RFF_RBF:
    """
    rbf(gaussian) kernel of GPy k(x, y) = variance * exp(- 0.5 * ||x - y||_2^2 / lengthscale**2)
    """

    def __init__(self, lengthscales, input_dim, variance=1, basis_dim=1000):
        self.basis_dim = basis_dim
        self.std = np.sqrt(variance)
        self.random_weights = (1 / np.atleast_2d(lengthscales)) * np.random.normal(
            size=(basis_dim, input_dim)
        )
        self.random_offset = np.random.uniform(0, 2 * np.pi, size=basis_dim)

    def transform(self, X):
        X = np.atleast_2d(X)
        X_transform = X.dot(self.random_weights.T) + self.random_offset
        X_transform = self.std * np.sqrt(2 / self.basis_dim) * np.cos(X_transform)
        return X_transform

    """
    Only for one dimensional X
    """

    def transform_grad(self, X):
        X = np.atleast_2d(X)
        X_transform_grad = X.dot(self.random_weights.T) + self.random_offset
        X_transform_grad = (
            -self.std
            * np.sqrt(2 / self.basis_dim)
            * np.sin(X_transform_grad)
            * self.random_weights.T
        )
        return X_transform_grad


root_two = np.sqrt(2)


def normal_cdf_inverse(z):
    return special.erfinv(2 * z - 1) * root_two


def normcdf(x):
    return 0.5 * (1 + special.erf(x / root_two))


def normpdf(x):
    pdf = np.zeros(np.shape(x))
    small_x_idx = np.abs(x) < 50
    pdf[small_x_idx] = np.exp(-x[small_x_idx] ** 2 / 2) / (np.sqrt(2 * np.pi))
    return pdf


def RBF_ARD_kernel_gradient(K, X, lengthscale):
    N = np.shape(K)[0]
    d = np.shape(X)[1]
    X1 = X.reshape(N, 1, d)
    X2 = X.reshape(1, N, d)
    # N times N times d
    one_dimensional_squared_scaled_distance = (X1 - X2) ** 2 / lengthscale.reshape(
        1, 1, d
    ) ** 3
    return K.reshape(N, N, 1) * one_dimensional_squared_scaled_distance


if __name__ == "__main__":
    pass
