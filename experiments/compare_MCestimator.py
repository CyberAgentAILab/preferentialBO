# -*- coding: utf-8 -*-
import sys
import time
import os

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")
from matplotlib.ticker import ScalarFormatter
from matplotlib import ticker

import pickle
import GPy

from preferentialBO.test_functions import test_functions
from preferentialBO.src.preferential_gp_regression import *

plt.rcParams["figure.figsize"] = (8, 10)
plt.rcParams["font.size"] = 24
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["image.cmap"] = "plasma"
plt.rcParams["axes.labelpad"] = 0.2
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["lines.markersize"] = 10

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

    results_path = "MC_estimator_comparison/" + func_name + "/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    sample_size_list = [100, 1000, 10000]
    mean_errors_list = [[], [], []]
    mode_errors_list = [[], [], []]
    prob_errors_list = [[], [], []]

    mean_FS_errors_list = [[], [], []]
    mode_FS_errors_list = [[], [], []]
    prob_FS_errors_list = [[], [], []]

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
        test_X_prob = np.r_[X_init_sort, tmp_X]
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

        mean_acc = PGP_Gibbs_truth.mean(test_X).ravel()
        mode_acc = PGP_Gibbs_truth.quantile(test_X, prob=0.5).ravel()
        prob_acc = np.array(
            [
                PGP_Gibbs_truth.win_prob(
                    test_X_prob[i, :input_dim], test_X_prob[i, input_dim:]
                )
                for i in range(np.shape(test_X_prob)[0])
            ]
        ).ravel()

        for i, sample_size in enumerate(sample_size_list):
            PGP_Gibbs = PreferentialGP_Gibbs(
                X_init_sort,
                kernel,
                kernel_bounds=kernel_bounds,
                burn_in=1000,
                thinning=1,
                sample_size=sample_size,
            )
            PGP_Gibbs.inference()

            # proposed
            mean = PGP_Gibbs.mean(test_X).ravel()
            mode = PGP_Gibbs.quantile(test_X, prob=0.5).ravel()
            prob = np.array(
                [
                    PGP_Gibbs.win_prob(
                        test_X_prob[i, :input_dim], test_X_prob[i, input_dim:]
                    )
                    for i in range(np.shape(test_X_prob)[0])
                ]
            ).ravel()

            mean_errors_list[i].append(((mean - mean_acc) ** 2).ravel())
            mode_errors_list[i].append(((mode - mode_acc) ** 2).ravel())
            prob_errors_list[i].append(((prob - prob_acc) ** 2).ravel())

            # sample path-based MC estimation (Fully Sampled)
            PGP_Gibbs.sample()
            sample_path = PGP_Gibbs.evaluate_sample(test_X)

            mean_FS = np.mean(sample_path, axis=1).ravel()
            sample_wise_sort_sample_path = np.sort(sample_path, axis=1)
            mode_FS = sample_wise_sort_sample_path[
                :, int(0.5 * np.shape(sample_path)[1])
            ].ravel()
            prob_FS = np.mean(
                np.heaviside(
                    -sample_path[: 2 * dim, :] + sample_path[2 * dim :, :], 0.5
                ),
                axis=1,
            ).ravel()

            mean_FS_errors_list[i].append(((mean_FS - mean_acc) ** 2).ravel())
            mode_FS_errors_list[i].append(((mode_FS - mode_acc) ** 2).ravel())
            prob_FS_errors_list[i].append(((prob_FS - prob_acc) ** 2).ravel())

    with open(results_path + "mean_error.pickle", "wb") as f:
        pickle.dump(mean_errors_list, f)
    with open(results_path + "mode_error.pickle", "wb") as f:
        pickle.dump(mode_errors_list, f)
    with open(results_path + "prob_error.pickle", "wb") as f:
        pickle.dump(prob_errors_list, f)

    with open(results_path + "mean_FS_error.pickle", "wb") as f:
        pickle.dump(mean_FS_errors_list, f)
    with open(results_path + "mode_FS_error.pickle", "wb") as f:
        pickle.dump(mode_FS_errors_list, f)
    with open(results_path + "prob_FS_error.pickle", "wb") as f:
        pickle.dump(prob_FS_errors_list, f)


def plot_RMSE(func_name):
    results_path = "MC_estimator_comparison/" + func_name + "/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    sample_size_list = [100, 1000, 10000]
    with open(results_path + "mean_error.pickle", "rb") as f:
        mean_errors_list = pickle.load(f)
    with open(results_path + "mode_error.pickle", "rb") as f:
        mode_errors_list = pickle.load(f)
    with open(results_path + "prob_error.pickle", "rb") as f:
        prob_errors_list = pickle.load(f)

    with open(results_path + "mean_FS_error.pickle", "rb") as f:
        mean_FS_errors_list = pickle.load(f)
    with open(results_path + "mode_FS_error.pickle", "rb") as f:
        mode_FS_errors_list = pickle.load(f)
    with open(results_path + "prob_FS_error.pickle", "rb") as f:
        prob_FS_errors_list = pickle.load(f)

    # Error of mean
    fig = plt.figure(figsize=(5.3, 7.0))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(
        sample_size_list,
        np.array([np.sqrt(np.mean(SE)) for SE in mean_errors_list]),
        marker="o",
        label="proposed",
        clip_on=False,
    )
    ax1.plot(
        sample_size_list,
        np.array([np.sqrt(np.mean(SE)) for SE in mean_FS_errors_list]),
        marker="s",
        label="Full MC",
        clip_on=False,
    )
    ax1.set_xlim(100, 10000)
    ax1.set_xscale("log")
    ax1.set_xlabel("# MC sample")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Mean")
    ax1.legend(loc="upper right")
    # ax1.set_yscale("log")

    # ax1.set_yticks([0.01, 0.1])
    # ax1.set_yticklabels(["0.5", "1"])

    ax1.yaxis.set_major_formatter(ScalarFormatterForceFormat(useMathText=True))
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(results_path + func_name + "_mean_error.pdf")
    # plt.show()
    plt.close()

    # Error of mode
    fig = plt.figure(figsize=(5.0, 7.0))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(
        sample_size_list,
        [np.sqrt(np.mean(SE)) for SE in mode_errors_list],
        marker="o",
        label="proposed",
        clip_on=False,
    )
    ax1.plot(
        sample_size_list,
        np.array([np.sqrt(np.mean(SE)) for SE in mode_FS_errors_list]),
        marker="s",
        label="Full MC",
        clip_on=False,
    )
    ax1.set_xlim(100, 10000)
    ax1.set_xscale("log")
    ax1.set_xlabel("# MC sample")
    # ax1.set_ylabel("RMSE")
    ax1.set_title("Mode")
    ax1.legend(loc="upper right")

    # ax1.set_yscale("log")
    # ax1.set_yticks([0.01, 0.1])
    # ax1.set_yticklabels(["0.5", "1"])

    ax1.yaxis.set_major_formatter(ScalarFormatterForceFormat(useMathText=True))
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(results_path + func_name + "_mode_error.pdf")
    # plt.show()
    plt.close()

    # Error of prob
    fig = plt.figure(figsize=(5.0, 7.0))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(
        sample_size_list,
        [np.sqrt(np.mean(SE)) for SE in prob_errors_list],
        marker="o",
        label="proposed",
        clip_on=False,
    )
    ax1.plot(
        sample_size_list,
        np.array([np.sqrt(np.mean(SE)) for SE in prob_FS_errors_list]),
        marker="s",
        label="Full MC",
        clip_on=False,
    )
    ax1.set_xlim(100, 10000)
    ax1.set_xscale("log")
    ax1.set_xlabel("# MC sample")
    # ax1.set_ylabel("RMSE")
    ax1.set_title("         Duel prob.")
    ax1.legend(loc="upper right")

    # ax1.set_yscale("log")
    # ax1.set_yticks([0.01, 0.1])
    # ax1.set_yticklabels(["0.5", "1"])
    ax1.yaxis.set_major_formatter(ScalarFormatterForceFormat(useMathText=True))
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(results_path + func_name + "_prob_error.pdf")
    # plt.show()
    plt.close()


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here


if __name__ == "__main__":
    args = sys.argv
    func_name = args[1]
    calculate_SE(func_name)
    plot_RMSE(func_name)
