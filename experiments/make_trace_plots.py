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

sys.path.append("./Benavoli_code")
import commoncode as Benavoli_code

from preferentialBO.test_functions import test_functions
from preferentialBO.src.preferential_gp_regression import *

plt.rcParams["figure.figsize"] = (8, 10)
plt.rcParams["font.size"] = 25
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["image.cmap"] = "plasma"
plt.rcParams["axes.labelpad"] = 0.2
plt.rcParams["axes.axisbelow"] = True
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


def main(func_name):
    # Problem dimension
    dim = 50
    seed = 0

    Gibbs_times = []
    ESS_times = []

    noise_std = 1e-2

    test_func = eval("test_functions." + func_name)()
    x_bounds = test_func.bounds
    input_dim = test_func.d

    interval_size = x_bounds[1] - x_bounds[0]
    kernel_bounds = np.array([interval_size * 1e-1, interval_size / 2.0])

    results_path = "trace_plots/" + func_name + "/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

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

        burn_in = 1
        thinning = 1
        sample_size = 1000
        # -----------------------------------------------------------------
        PGP_Gibbs = PreferentialGP_Gibbs(
            X_init_sort,
            kernel,
            kernel_bounds=kernel_bounds,
            burn_in=burn_in,
            thinning=thinning,
            sample_size=sample_size,
        )
        PGP_Gibbs.model_selection()
        kernel = PGP_Gibbs.kernel
        start = time.time()
        PGP_Gibbs.inference()
        Gibbs_times.append(time.time() - start)
        # print(time.time() - start)
        # print(PGP_Gibbs.v_sample)

        # -----------------------------------------------------------------

        start = time.time()
        # PGP_ESS.inference()
        ESS_v_sample = -1 * Benavoli_code.sample_truncated(
            trunc=np.c_[np.zeros(PGP_Gibbs.num_duels)],
            mean=np.atleast_2d(np.zeros(PGP_Gibbs.num_duels)),
            C=PGP_Gibbs.K_v,
            nsamples=sample_size,
            tune=PGP_Gibbs.burn_in,
        ).T.reshape(PGP_Gibbs.num_duels, sample_size)
        ESS_times.append(time.time() - start)
        # print(ESS_v_sample)

        if seed == 0:
            for idx in range(dim):
                fig = plt.figure(figsize=(15, 7))
                ax1 = plt.subplot(2, 2, 1)
                ax1.plot(np.arange(sample_size), ESS_v_sample[idx, :])
                ax1.set_xlim(0, sample_size)
                ax1.set_title("Trace of LinESS")

                ax2 = plt.subplot(2, 2, 2)
                ax2.plot(np.arange(sample_size), PGP_Gibbs.v_sample[idx, :])
                ax2.set_xlim(0, sample_size)
                ax2.set_title("Trace of Gibbs sampling")

                ax3 = plt.subplot(2, 2, 3)
                ax3.acorr(
                    ESS_v_sample[idx, :] - np.mean(ESS_v_sample[idx, :]),
                    maxlags=None,
                )
                ax3.set_ylabel("Autocorrelation")
                ax3.set_xlim(0, sample_size)

                ax4 = plt.subplot(2, 2, 4)
                ax4.acorr(
                    PGP_Gibbs.v_sample[idx, :] - np.mean(PGP_Gibbs.v_sample[idx, :]),
                    maxlags=None,
                )
                ax4.set_ylabel("Autocorrelation")
                ax4.set_xlim(0, sample_size)

                ax1.set_xlabel("Iterations of MCMC")
                ax2.set_xlabel("Iterations of MCMC")
                ax3.set_xlabel("Iterations of MCMC")
                ax4.set_xlabel("Iterations of MCMC")

                plt.tight_layout()
                plt.savefig(results_path + func_name + "_trace_{:0=2}.pdf".format(idx))
                # plt.show()
                plt.close()
            exit()

    print(ESS_times)
    print(Gibbs_times)
    with open(results_path + "computational_time.txt", "w") as f:
        f.write(
            "${:.2f} \\pm {:.2f}$ & ${:.2f} \\pm {:.2f}$".format(
                np.mean(ESS_times),
                np.std(ESS_times),
                np.mean(Gibbs_times),
                np.std(Gibbs_times),
            )
        )

    # ---------------------------------------------------------------


if __name__ == "__main__":
    args = sys.argv
    func_name = args[1]
    main(func_name)
