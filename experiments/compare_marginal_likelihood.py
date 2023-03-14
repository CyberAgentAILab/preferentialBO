# -*- coding: utf-8 -*-
import sys
import os

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")
import time

import GPy
from scipy.stats import multivariate_normal
from scipy.stats import norm
import scipy.linalg as sp_linalg
from scipy.stats import qmc
from scipy import special

from preferentialBO.test_functions import test_functions
from preferentialBO.src.preferential_gp_regression import *

plt.rcParams["figure.figsize"] = (8, 10)
plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["image.cmap"] = "plasma"
plt.rcParams["axes.labelpad"] = 0.2
plt.rcParams["axes.axisbelow"] = True

plt.rcParams["legend.fontsize"] = 20
plt.rcParams["legend.borderpad"] = 0.2  # border whitespace
plt.rcParams[
    "legend.labelspacing"
] = 0.2  # the vertical space between the legend entries
plt.rcParams["legend.handlelength"] = 1.5  # the length of the legend lines
plt.rcParams["legend.handleheight"] = 0.3  # the height of the legend handle
plt.rcParams[
    "legend.handletextpad"
] = 0.3  # the space between the legend line and legend text
plt.rcParams[
    "legend.borderaxespad"
] = 0.3  # the border between the axes and legend edge
plt.rcParams["legend.columnspacing"] = 1.0  # column separation

plt.rcParams["figure.constrained_layout.use"] = True


def main(func_name):
    # Problem dimension
    dim = 50
    seed = 0
    noise_std = 1e-2

    test_func = eval("test_functions." + func_name)()
    x_bounds = test_func.bounds
    input_dim = test_func.d

    interval_size = x_bounds[1] - x_bounds[0]
    kernel_bounds = np.array([interval_size * 1e-1, interval_size / 2.0])

    results_path = "MLE_plots/" + func_name + "/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # seed for reproducibility
    np.random.seed(seed)

    X_init = (
        np.random.rand(2 * dim, input_dim) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    )
    y_init = test_func.values(X_init)

    X_init = X_init.reshape(dim, 2, input_dim)
    y_init = y_init.reshape(dim, 2, 1)
    sort_idx = np.argsort(y_init, axis=1)

    X_init_sort = np.take_along_axis(X_init, sort_idx, axis=1)[:, ::-1, :].reshape(
        dim, 2 * input_dim
    )

    grid_num = 21
    ell1 = np.linspace(kernel_bounds[0, 0], kernel_bounds[1, 0], grid_num)
    ell2 = np.linspace(kernel_bounds[0, 1], kernel_bounds[1, 1], grid_num)
    Ell1, Ell2 = np.meshgrid(ell1, ell2)
    Ell = np.c_[np.c_[np.ravel(Ell1)], np.c_[np.ravel(Ell2)]]

    scipy_time_list = []
    scipy_logcdf_list = []

    Laplace_time_list = []
    Laplace_logcdf_list = []

    EP_time_list = []
    EP_logcdf_list = []
    upper = np.zeros(dim)
    for i in range(np.shape(Ell)[0]):
        kernel = GPy.kern.RBF(
            input_dim=input_dim, lengthscale=Ell[i], variance=1, ARD=True
        )

        # -----------------------------------------------------------------
        PGP_LA = PreferentialGP_Laplace(
            X_init_sort, kernel, kernel_bounds=kernel_bounds, noise_std=noise_std
        )
        start = time.time()
        minus_log_likelihood_Laplace = PGP_LA.minus_log_likelihood(
            np.atleast_2d(Ell[i])
        )
        Laplace_time_list.append(time.time() - start)
        Laplace_logcdf_list.append(-minus_log_likelihood_Laplace)

        # -----------------------------------------------------------------
        PGP_EP = PreferentialGP_EP(
            X_init_sort, kernel, kernel_bounds=kernel_bounds, noise_std=noise_std
        )
        start = time.time()
        minus_log_likelihood_EP = PGP_EP.minus_log_likelihood_ep(Ell[i])
        EP_time_list.append(time.time() - start)
        EP_logcdf_list.append(-minus_log_likelihood_EP)

        # -----------------------------------------------------------------
        A = np.c_[-1 * np.eye(dim), np.eye(dim)]
        flatten_K = PGP_LA.kernel.K(PGP_LA.flatten_X) + PGP_LA.noise_std**2 * np.eye(
            2 * PGP_LA.num_duels
        )
        cov = A @ flatten_K @ A.T

        start = time.time()
        likelihood = multivariate_normal.cdf(
            upper, mean=np.zeros(dim), cov=cov, abseps=1e-8, releps=1e-8
        )
        log_likelihood_scipy = np.log(likelihood)
        scipy_time_list.append(time.time() - start)
        scipy_logcdf_list.append(log_likelihood_scipy)

    print("average time scipy = {}".format(np.mean(scipy_time_list)))
    print("average time Laplace = {}".format(np.mean(Laplace_time_list)))
    print("average time EP = {}".format(np.mean(EP_time_list)))

    # vmin = np.min([np.min(scipy_logcdf_list), np.min(Laplace_logcdf_list), np.min(EP_logcdf_list)])
    # vmax = np.max([np.max(scipy_logcdf_list), np.max(Laplace_logcdf_list), np.max(EP_logcdf_list)])
    vmin = None
    vmax = None

    figure = plt.figure(figsize=(20, 5))

    ax2 = plt.subplot(1, 3, 1)
    scipy_logcdf = np.array(scipy_logcdf_list).reshape(np.shape(Ell1))
    scipy_logcdf[np.isnan(scipy_logcdf)] = -np.inf
    heatmap = ax2.pcolor(Ell1, Ell2, scipy_logcdf, vmin=vmin, vmax=vmax)

    # max_idx = np.unravel_index(np.argmax(scipy_logcdf), scipy_logcdf.shape)
    # ax2.scatter(ell1[max_idx[1]], ell2[max_idx[0]], marker="x", color="red")
    figure.colorbar(heatmap, format="%1.0f")
    ax2.set_title("Scipy: average time = {:.2e}".format(np.mean(scipy_time_list)))

    ax3 = plt.subplot(1, 3, 2)
    Laplace_logcdf = np.array(Laplace_logcdf_list).reshape(np.shape(Ell1))
    Laplace_logcdf[np.isnan(Laplace_logcdf)] = -np.inf
    heatmap = ax3.pcolor(Ell1, Ell2, Laplace_logcdf, vmin=vmin, vmax=vmax)

    # max_idx = np.unravel_index(np.argmax(Laplace_logcdf), Laplace_logcdf.shape)
    # ax3.scatter(ell1[max_idx[1]], ell2[max_idx[0]], marker="x", color="red")

    figure.colorbar(heatmap, format="%1.0f")
    ax3.set_title("LA: average time = {:.2e}".format(np.mean(Laplace_time_list)))

    ax4 = plt.subplot(1, 3, 3)
    EP_logcdf = np.array(EP_logcdf_list).reshape(np.shape(Ell1))
    EP_logcdf[np.isnan(EP_logcdf)] = -np.inf
    heatmap = ax4.pcolor(Ell1, Ell2, EP_logcdf, vmin=vmin, vmax=vmax)

    # max_idx = np.unravel_index(np.argmax(EP_logcdf), EP_logcdf.shape)
    # ax4.scatter(ell1[max_idx[1]], ell2[max_idx[0]], marker="x", color="red")

    figure.colorbar(heatmap, format="%1.0f")
    ax4.set_title("EP: average time = {:.2e}".format(np.mean(EP_time_list)))

    plt.savefig(results_path + "log_cdf_test-" + str(dim) + ".pdf")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    args = sys.argv
    func_name = args[1]
    main(func_name)
