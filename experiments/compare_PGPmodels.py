# -*- coding: utf-8 -*-
import sys
import time
import os

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")

import pickle
import GPy
from scipy.stats import norm

from preferentialBO.test_functions import test_functions
from preferentialBO.src.preferential_gp_regression import *

plt.rcParams["figure.figsize"] = (8, 10)
plt.rcParams["font.size"] = 28
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


def calculate_SE(func_name):
    # Problem dimension
    dim = 50
    seed = 0
    noise_std = 1e-2

    test_func = eval("test_functions." + func_name)()
    x_bounds = test_func.bounds
    input_dim = test_func.d

    interval_size = x_bounds[1] - x_bounds[0]
    kernel_bounds = np.array([interval_size * 1e-1, interval_size / 2.0])

    results_path = "PGPmodel_comparison/" + func_name + "/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    mean_list = []
    mode_list = []
    prob_list = []

    mean_EP_list = []
    mode_EP_list = []
    prob_EP_list = []

    mean_LA_list = []
    mode_LA_list = []
    prob_LA_list = []

    for seed in range(10):
        # seed for reproducibility
        np.random.seed(seed)

        X_init = (
            np.random.rand(2 * dim, input_dim) * (x_bounds[1] - x_bounds[0])
            + x_bounds[0]
        )
        y_init = test_func.values(X_init) + noise_std * np.random.randn(2 * dim, 1)

        X_init = X_init.reshape(dim, 2, input_dim)
        y_init = y_init.reshape(dim, 2, 1)
        sort_idx = np.argsort(y_init, axis=1)
        X_init_sort = np.take_along_axis(X_init, sort_idx, axis=1)[:, ::-1, :].reshape(
            dim, 2 * input_dim
        )

        kernel = GPy.kern.RBF(
            input_dim=input_dim,
            lengthscale=0.2 * np.ones(input_dim),
            variance=1,
            ARD=True,
        )

        tmp_X = (
            np.random.rand(2 * dim, input_dim) * (x_bounds[1] - x_bounds[0])
            + x_bounds[0]
        )
        tmp_X = tmp_X.reshape(dim, 2 * input_dim)
        test_X_prob = np.r_[X_init.reshape(dim, 2 * input_dim), tmp_X]
        test_X = np.r_[test_X_prob[:, :input_dim], test_X_prob[:, input_dim:]]

        # -----------------------------------------------------------------
        PGP_Gibbs_truth = PreferentialGP_Gibbs(
            X_init_sort,
            kernel,
            kernel_bounds=kernel_bounds,
            burn_in=1000,
            thinning=10,
            sample_size=10000,
        )
        PGP_Gibbs_truth.model_selection()
        kernel = PGP_Gibbs_truth.kernel
        PGP_Gibbs_truth.inference()

        mean = PGP_Gibbs_truth.mean(test_X).ravel()
        mode = PGP_Gibbs_truth.quantile(test_X, prob=0.5).ravel()
        prob = np.array(
            [
                PGP_Gibbs_truth.win_prob(
                    test_X_prob[i, :input_dim], test_X_prob[i, input_dim:]
                )
                for i in range(np.shape(test_X_prob)[0])
            ]
        ).ravel()

        mean_list.append(mean)
        mode_list.append(mode)
        prob_list.append(prob)

        PGP_LA = PreferentialGP_Laplace(
            X_init_sort, kernel, kernel_bounds=kernel_bounds
        )
        PGP_LA.inference()

        def LA_prob(x1, x2):
            X = np.r_[np.atleast_2d(x1), np.atleast_2d(x2)]
            mean, cov = PGP_LA.predict(X, full_cov=True)
            mean_duel = mean[0] - mean[1]
            std_duel = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[1, 0])
            return norm.cdf(-mean_duel / std_duel)

        mean_LA, _ = PGP_LA.predict(test_X)
        mean_LA = mean_LA.ravel()
        mode_LA = np.copy(mean_LA)
        prob_LA = np.array(
            [
                LA_prob(test_X_prob[i, :input_dim], test_X_prob[i, input_dim:])
                for i in range(np.shape(test_X_prob)[0])
            ]
        ).ravel()

        mean_LA_list.append(mean_LA)
        mode_LA_list.append(mode_LA)
        prob_LA_list.append(prob_LA)

        PGP_EP = PreferentialGP_EP(X_init_sort, kernel, kernel_bounds=kernel_bounds)
        PGP_EP.inference()

        def EP_prob(x1, x2):
            X = np.r_[np.atleast_2d(x1), np.atleast_2d(x2)]
            mean, cov = PGP_EP.predict(X, full_cov=True)
            mean_duel = mean[0] - mean[1]
            std_duel = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[1, 0])
            return norm.cdf(-mean_duel / std_duel)

        mean_EP, _ = PGP_EP.predict(test_X)
        mean_EP = mean_EP.ravel()
        mode_EP = np.copy(mean_EP)
        prob_EP = np.array(
            [
                EP_prob(test_X_prob[i, :input_dim], test_X_prob[i, input_dim:])
                for i in range(np.shape(test_X_prob)[0])
            ]
        ).ravel()

        mean_EP_list.append(mean_EP)
        mode_EP_list.append(mode_EP)
        prob_EP_list.append(prob_EP)

    with open(results_path + "mean.pickle", "wb") as f:
        pickle.dump(mean_list, f)
    with open(results_path + "mode.pickle", "wb") as f:
        pickle.dump(mode_list, f)
    with open(results_path + "prob.pickle", "wb") as f:
        pickle.dump(prob_list, f)

    with open(results_path + "mean_LA.pickle", "wb") as f:
        pickle.dump(mean_LA_list, f)
    with open(results_path + "mode_LA.pickle", "wb") as f:
        pickle.dump(mode_LA_list, f)
    with open(results_path + "prob_LA.pickle", "wb") as f:
        pickle.dump(prob_LA_list, f)

    with open(results_path + "mean_EP.pickle", "wb") as f:
        pickle.dump(mean_EP_list, f)
    with open(results_path + "mode_EP.pickle", "wb") as f:
        pickle.dump(mode_EP_list, f)
    with open(results_path + "prob_EP.pickle", "wb") as f:
        pickle.dump(prob_EP_list, f)


def plot_RMSE(func_name):
    results_path = "PGPmodel_comparison/" + func_name + "/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(results_path + "mean.pickle", "rb") as f:
        mean_list = pickle.load(f)
    with open(results_path + "mode.pickle", "rb") as f:
        mode_list = pickle.load(f)
    with open(results_path + "prob.pickle", "rb") as f:
        prob_list = pickle.load(f)
    prob = np.array(prob_list)

    with open(results_path + "mean_LA.pickle", "rb") as f:
        mean_LA_list = pickle.load(f)
    with open(results_path + "mode_LA.pickle", "rb") as f:
        mode_LA_list = pickle.load(f)
    with open(results_path + "prob_LA.pickle", "rb") as f:
        prob_LA_list = pickle.load(f)
    prob_LA = np.array(prob_LA_list)

    with open(results_path + "mean_EP.pickle", "rb") as f:
        mean_EP_list = pickle.load(f)
    with open(results_path + "mode_EP.pickle", "rb") as f:
        mode_EP_list = pickle.load(f)
    with open(results_path + "prob_EP.pickle", "rb") as f:
        prob_EP_list = pickle.load(f)
    prob_EP = np.array(prob_EP_list)

    #############################################################################################
    # Error of mean
    #############################################################################################
    plot_max = np.max([np.max(mean_LA_list), np.max(mean_list), np.max(mean_EP_list)])
    plot_min = np.min([np.min(mean_LA_list), np.min(mean_list), np.min(mean_EP_list)])
    tmp_x_y = np.linspace(plot_min, plot_max, 11)

    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(1, 1, 1)
    ax1.scatter(np.array(mean_list).ravel(), np.array(mean_LA_list).ravel())
    ax1.plot(tmp_x_y, tmp_x_y, color="black", zorder=10)
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_ylim(plot_min, plot_max)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_ylabel("LA")
    ax1.set_xlabel("Ground truth")
    ax1.set_title("Mean")

    # ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(results_path + func_name + "_LA_mean.pdf")
    # plt.show()
    plt.close()

    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(1, 1, 1)
    ax1.scatter(np.array(mean_list).ravel(), np.array(mean_EP_list).ravel())
    ax1.plot(tmp_x_y, tmp_x_y, color="black", zorder=10)
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_ylim(plot_min, plot_max)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_ylabel("EP")
    ax1.set_xlabel("Ground truth")
    ax1.set_title("Mean")

    # ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(results_path + func_name + "_EP_mean.pdf")
    # plt.show()
    plt.close()

    #############################################################################################
    # Error of mode
    #############################################################################################
    plot_max = np.max([np.max(mode_LA_list), np.max(mode_list), np.max(mode_EP_list)])
    plot_min = np.min([np.min(mode_LA_list), np.min(mode_list), np.min(mode_EP_list)])
    tmp_x_y = np.linspace(plot_min, plot_max, 11)

    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(1, 1, 1)
    ax1.scatter(np.array(mode_list).ravel(), np.array(mode_LA_list).ravel())
    ax1.plot(tmp_x_y, tmp_x_y, color="black", zorder=10)
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_ylim(plot_min, plot_max)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_ylabel("LA")
    ax1.set_xlabel("Ground truth")
    ax1.set_title("Mode")

    # ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(results_path + func_name + "_LA_mode.pdf")
    # plt.show()
    plt.close()

    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(1, 1, 1)
    ax1.scatter(np.array(mode_list).ravel(), np.array(mode_EP_list).ravel())
    ax1.plot(tmp_x_y, tmp_x_y, color="black", zorder=10)
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_ylim(plot_min, plot_max)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_ylabel("EP")
    ax1.set_xlabel("Ground truth")
    ax1.set_title("Mode")

    # ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(results_path + func_name + "_EP_mode.pdf")
    # plt.show()
    plt.close()

    #############################################################################################
    # Error of prob
    #############################################################################################
    plot_max = 1
    plot_min = 0.0
    tmp_x_y = np.linspace(plot_min, plot_max, 11)

    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(1, 1, 1)
    ax1.scatter(prob, prob_LA)
    ax1.plot(tmp_x_y, tmp_x_y, color="black", zorder=10)
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_ylim(plot_min, plot_max)
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax1.set_xticks([0.5, 1])
    # ax1.set_xticklabels(["0.5", "1"])
    # ax1.set_yticks([0.5, 1])
    # ax1.set_yticklabels(["0.5", "1"])

    ax1.set_ylabel("LA")
    ax1.set_xlabel("Ground truth")
    ax1.set_title("Duel prob.")

    # ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(results_path + func_name + "_LA_prob.pdf")
    # plt.show()
    plt.close()

    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(1, 1, 1)
    ax1.scatter(prob, prob_EP)
    ax1.plot(tmp_x_y, tmp_x_y, color="black", zorder=10)
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_ylim(plot_min, plot_max)
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax1.set_xticks([0.5, 1])
    # ax1.set_xticklabels(["0.5", "1"])
    # ax1.set_yticks([0.5, 1])
    # ax1.set_yticklabels(["0.5", "1"])

    ax1.set_ylabel("EP")
    ax1.set_xlabel("Ground truth")
    ax1.set_title("Duel prob.")

    # ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(results_path + func_name + "_EP_prob.pdf")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    args = sys.argv
    func_name = args[1]
    calculate_SE(func_name)
    plot_RMSE(func_name)
