# -*- coding: utf-8 -*-
import os
import sys
import pickle

import numpy as np
import numpy.matlib
import matplotlib
from cycler import cycler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from preferentialBO.test_functions import test_functions

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 16
plt.rcParams["figure.figsize"] = (7, 11)
plt.rcParams["errorbar.capsize"] = 2.0
plt.rcParams["lines.linewidth"] = 3.5
plt.rcParams["lines.markeredgewidth"] = 1.5
plt.rcParams["lines.markersize"] = 10.0
plt.rcParams["xtick.alignment"] = "right"
plt.rcParams["legend.borderaxespad"] = 0.15
plt.rcParams["legend.borderpad"] = 0.2
plt.rcParams["legend.columnspacing"] = 0.5
plt.rcParams["legend.handletextpad"] = 0.5
plt.rcParams["legend.handlelength"] = 3.0
plt.rcParams["legend.handleheight"] = 0.5


def normal_plot(q=1, Parallel=False):
    plt.rcParams["figure.figsize"] = (5.0, 4.5)
    plt.rcParams["font.size"] = 16
    plt.rcParams["legend.fontsize"] = 16

    STR_NUM_WORKER = "Q=" + str(q)
    seeds_num = 10
    seeds = np.arange(seeds_num)

    func_names = [
        "Branin",
        "Ackley",
        "Bukin",
        "Cross_in_tray",
        "Eggholder",
        "Holder_table",
        "Langerman",
        "Levy",
        "Levy13",
        "HartMann3",
        "HartMann4",
        "HartMann6",
    ]
    y_max = [4.0, 20, 25, 0.2, 600, 10, None, 0.15, 13, 0.3, 0.6, 1.5]
    BO_methods = [
        "HB-EI",
        "HB-UCB",
        "LA-EI",
        "EP-EI",
        "EP-MUC",
        "DuelTS",
        "DuelUCB",
        "EIIG",
    ]

    cmap = plt.get_cmap("tab10")
    color_idx = np.arange(len(BO_methods)).tolist()
    color_idx[1] = 3
    color_idx[3] = 1

    # fig = plt.figure(figsize=(10, len(func_names)*2.5))
    for i, func_name in enumerate(func_names):
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)

        print(func_name + "--------------------------------------")
        test_func = eval("test_functions." + func_name)()
        input_dim = test_func.d

        GLOBAL_MAX = None
        if func_name == "Forrester":
            GLOBAL_MAX = 6.0207400553619665
        if func_name == "Material":
            GLOBAL_MAX = -0.008710978
        if func_name == "Branin":
            GLOBAL_MAX = -0.397887
        if func_name == "Beale":
            GLOBAL_MAX = 0
        if func_name == "Borehole":
            GLOBAL_MAX = -7.82086062
        if func_name == "Styblinski_tang":
            GLOBAL_MAX = 39.16599 * input_dim
        if func_name == "CurrinExp":
            GLOBAL_MAX = 13.798719
        if func_name == "HartMann6":
            GLOBAL_MAX = 3.32237
        if func_name == "HartMann3":
            GLOBAL_MAX = 3.86278
        if func_name == "HartMann4":
            GLOBAL_MAX = 3.1344918837890545
        if func_name == "Colville":
            GLOBAL_MAX = 0
        if func_name == "Powell":
            GLOBAL_MAX = 0
        if func_name == "Shekel":
            GLOBAL_MAX = 10.5364
        if func_name in [
            "Ackley",
            "Bukin",
            "Cross_in_tray",
            "Eggholder",
            "Holder_table",
            "Langerman",
            "Levy",
            "Levy13",
            "Rastrigin",
            "Shubert",
            "Schwefel",
        ]:
            GLOBAL_MAX = test_func.GLOBAL_MAXIMUM

        result_path = func_name + "_results/"
        for j, method in enumerate(BO_methods):
            errorevery = (2 * j, 16)
            plot = True
            Regret_all = list()
            for seed in seeds:
                temp_path = result_path + method + "/seed=" + str(seed) + "/"
                if (
                    os.path.exists(temp_path + "Regret.pickle")
                    and os.path.getsize(temp_path + "Regret.pickle") > 0
                ):
                    with open(temp_path + "Regret.pickle", "rb") as f:
                        Regret = pickle.load(f).ravel()
                    Regret_all.append(Regret)
                else:
                    plot = False

            if plot:
                min_len = np.min([np.size(reg) for reg in Regret_all])
                Regret_all = [reg[:min_len] for reg in Regret_all]
                Regret_all = np.vstack(Regret_all)

                Regret_ave = np.mean(Regret_all, axis=0)
                Regret_se = np.sqrt(
                    np.sum((Regret_all - Regret_ave) ** 2, axis=0) / (seeds_num - 1)
                ) / np.sqrt(seeds_num)
                if GLOBAL_MAX is None:
                    Regret_ave = np.abs(Regret_ave)
                else:
                    Regret_ave = GLOBAL_MAX - Regret_ave

                linestyle = None
                marker = None
                color = cmap(color_idx[j])
                label = method

                if "LA" in method or "EP" in method:
                    linestyle = "dashed"  # Gaussian approximation
                elif "HB" in method:
                    linestyle = "solid"  # Gibbs sampling
                else:
                    linestyle = "dashed"  # MCMC

                ax.errorbar(
                    np.arange(np.size(Regret_ave)),
                    Regret_ave,
                    yerr=Regret_se,
                    errorevery=errorevery,
                    capsize=4,
                    elinewidth=2,
                    label=label,
                    marker=marker,
                    markevery=5,
                    linestyle=linestyle,
                    color=color,
                    markerfacecolor="None",
                )

        # if 'HartMann' in func_name:
        func_name = func_name.replace("HartMann", "Hartmann")

        if Parallel:
            ax.set_title(
                func_name + "(d=" + str(input_dim) + ", " + STR_NUM_WORKER + ")"
            )
        else:
            ax.set_title(func_name + "(d=" + str(input_dim) + ")", loc="right")

        if y_max[i] is not None:
            ax.set_ylim(0, y_max[i])

        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])

        # ax.set_xlabel('Iteration')
        # ax.set_ylabel('Regret')
        ax.grid(which="major")
        ax.grid(which="minor")

        ax.yaxis.set_major_formatter(ScalarFormatterForceFormat(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        plt.tight_layout()
        if Parallel:
            fig.savefig(
                "plots/Results_Parallel_"
                + str(func_name)
                + "_"
                + STR_NUM_WORKER
                + ".pdf"
            )
        else:
            fig.savefig("plots/Results_" + str(func_name) + ".pdf")
        handles, labels = ax.get_legend_handles_labels()
        plt.close()

    legend_fig = plt.figure(figsize=(18, 2))
    legend_fig_ax = legend_fig.add_subplot(1, 1, 1)

    legend_fig_ax.legend(handles, labels, ncol=len(BO_methods), loc="upper left")
    legend_fig_ax.axis("off")
    if Parallel:
        legend_fig.savefig("plots/Results_" + STR_NUM_WORKER + "_legend.pdf")
    else:
        legend_fig.savefig("plots/Results_legend.pdf")

    plt.close()


def wctime_plot(q=1, Parallel=False):
    plt.rcParams["figure.figsize"] = (5.0, 4.5)
    plt.rcParams["font.size"] = 16
    plt.rcParams["legend.fontsize"] = 16

    STR_NUM_WORKER = "Q=" + str(q)
    seeds_num = 10
    seeds = np.arange(seeds_num)

    func_names = [
        "Branin",
        "Ackley",
        "Bukin",
        "Cross_in_tray",
        "Eggholder",
        "Holder_table",
        "Langerman",
        "Levy",
        "Levy13",
        "HartMann3",
        "HartMann4",
        "HartMann6",
    ]
    y_max = [4.0, 20, 25, 0.2, 600, 10, None, 0.15, 13, 0.3, 0.6, 1.5]
    BO_methods = [
        "HB-EI",
        "HB-UCB",
        "LA-EI",
        "EP-EI",
        "EP-MUC",
        "DuelTS",
        "DuelUCB",
        "EIIG",
    ]

    cmap = plt.get_cmap("tab10")
    color_idx = np.arange(len(BO_methods)).tolist()
    color_idx[1] = 3
    color_idx[3] = 1

    # fig = plt.figure(figsize=(10, len(func_names)*2.5))
    for i, func_name in enumerate(func_names):
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)

        print(func_name + "--------------------------------------")
        test_func = eval("test_functions." + func_name)()
        input_dim = test_func.d

        GLOBAL_MAX = None
        if func_name == "Forrester":
            GLOBAL_MAX = 6.0207400553619665
        if func_name == "Material":
            GLOBAL_MAX = -0.008710978
        if func_name == "Branin":
            GLOBAL_MAX = -0.397887
        if func_name == "Beale":
            GLOBAL_MAX = 0
        if func_name == "Borehole":
            GLOBAL_MAX = -7.82086062
        if func_name == "Styblinski_tang":
            GLOBAL_MAX = 39.16599 * input_dim
        if func_name == "CurrinExp":
            GLOBAL_MAX = 13.798719
        if func_name == "HartMann6":
            GLOBAL_MAX = 3.32237
        if func_name == "HartMann3":
            GLOBAL_MAX = 3.86278
        if func_name == "HartMann4":
            GLOBAL_MAX = 3.1344918837890545
        if func_name == "Colville":
            GLOBAL_MAX = 0
        if func_name == "Powell":
            GLOBAL_MAX = 0
        if func_name == "Shekel":
            GLOBAL_MAX = 10.5364
        if func_name in [
            "Ackley",
            "Bukin",
            "Cross_in_tray",
            "Eggholder",
            "Holder_table",
            "Langerman",
            "Levy",
            "Levy13",
            "Rastrigin",
            "Shubert",
            "Schwefel",
        ]:
            GLOBAL_MAX = test_func.GLOBAL_MAXIMUM

        result_path = func_name + "_results/"

        x_max = 2000
        errorevery = 100
        result_path = func_name + "_results/"
        for j, method in enumerate(BO_methods):
            errorevery = (j * 20, 160)
            plot = True
            Regret_all = list()
            computational_time_all = list()
            for seed in seeds:
                temp_path = result_path + method + "/seed=" + str(seed) + "/"
                if (
                    os.path.exists(temp_path + "Regret.pickle")
                    and os.path.getsize(temp_path + "Regret.pickle") > 0
                ):
                    with open(temp_path + "Regret.pickle", "rb") as f:
                        Regret = pickle.load(f).ravel()
                    with open(temp_path + "computation_time.pickle", "rb") as f:
                        computational_time = pickle.load(f).ravel()
                    Regret_all.append(Regret)
                    computational_time_all.append(np.cumsum(computational_time))
                else:
                    plot = False

            if plot:
                computational_time_unique = np.sort(
                    np.unique(np.hstack(computational_time_all))
                )

                x_max = np.min([x_max, np.max(computational_time_unique)])

                Regret_all_wctime = [
                    [
                        Regret_all[idx][computational_time_all[idx] <= tmp_time][-1]
                        for tmp_time in computational_time_unique
                    ]
                    for idx in seeds
                ]

                Regret_ave = np.mean(Regret_all_wctime, axis=0)
                Regret_se = np.sqrt(
                    np.sum((Regret_all_wctime - Regret_ave) ** 2, axis=0)
                    / (seeds_num - 1)
                ) / np.sqrt(seeds_num)
                if GLOBAL_MAX is None:
                    Regret_ave = np.abs(Regret_ave)
                else:
                    Regret_ave = GLOBAL_MAX - Regret_ave

                linestyle = None
                marker = None
                label = method
                color = cmap(color_idx[j])
                label = method

                if "LA" in method or "EP" in method:
                    linestyle = "dashed"  # Gaussian approximation
                elif "HB" in method:
                    linestyle = "solid"  # Gibbs sampling
                else:
                    linestyle = "dashed"  # MCMC

                ax.errorbar(
                    computational_time_unique,
                    Regret_ave,
                    yerr=Regret_se,
                    errorevery=errorevery,
                    capsize=4,
                    elinewidth=2,
                    label=label,
                    marker=marker,
                    markevery=5,
                    linestyle=linestyle,
                    color=color,
                    markerfacecolor="None",
                )

        # if 'HartMann' in func_name:
        func_name = func_name.replace("HartMann", "Hartmann")

        if Parallel:
            ax.set_title(
                func_name + "(d=" + str(input_dim) + ", " + STR_NUM_WORKER + ")"
            )
        else:
            ax.set_title(func_name + "(d=" + str(input_dim) + ")", loc="right")

        if y_max[i] is not None:
            ax.set_ylim(0, y_max[i])
        ax.set_xlim(0, x_max)
        # ax.set_xlabel('Computational time (sec)')
        # ax.set_ylabel('Regret')
        ax.grid(which="major")
        ax.grid(which="minor")

        ax.yaxis.set_major_formatter(ScalarFormatterForceFormat(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        plt.tight_layout()
        if Parallel:
            fig.savefig(
                "plots/Results_wctime_Parallel_"
                + str(func_name)
                + "_"
                + STR_NUM_WORKER
                + ".pdf"
            )
        else:
            fig.savefig("plots/Results_wctime_" + str(func_name) + ".pdf")
        plt.close()

    plt.close()


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here


if __name__ == "__main__":
    normal_plot(Parallel=False)
    wctime_plot(Parallel=False)
