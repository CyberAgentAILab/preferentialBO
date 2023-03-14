#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  18 14:24:17 2020

@author: benavoli
"""
import numpy as np
import sys

sys.path.append("../SkewGP")
import mvnxpb as mvnxpb
from scipy.stats import mvn


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


# compute Gaussian CDF
def gaussianCDF(Σ, left, right, method="other"):
    """
    Compute Gaussian CDF of N(0,Σ) between left and right

    Σ is a covariance matrix

    left is a vector that indicates the lower integration boundaries

    right is a vector that indicates the upper integration boundaries

    method is a string, either 'matlab' (use the qsilatmvnv matlab routine), 'other' (use mvnxpb)
    """
    C = Σ  #
    L = left  #
    R = right  #
    # print(L,R,C)
    # aaa
    if method == "other":
        res = mvnxpb.mvnxpb(C, L, R)
    else:
        # print("a")
        res = mvn.mvnun(L[:, 0], R[:, 0], np.zeros(L.shape[0], float), C)[0]
        # print(res)
    return np.atleast_1d(res)[0]


# sample truncated
def sample_truncated(trunc, mean, C, nsamples, sign=1, tune=1000, progress=True):
    """
    Sample from a truncated Gaussian vector N(mean,C) below trunc.

    trunc is a vector indicating the truncation

    mean is a vector indicating the mean of the Gaussian distribution

    C is a matrix indicating the covariance of the gaussianCDF

    nsamples is an integer that indicates the number of samples

    sign is the sign of the truncation (in order to do below/above)

    tune is an integer indicating the burn-in samples for the methods

    chains is the number of MCMC chains sampled for EllipticalSlice

    method is a string indicating the method used for sampling, "LinEss" or "EllipticalSlice"
    """
    import scipy as sp

    try:
        eigen_max = sp.linalg.eigh(
            C, eigvals_only=True, eigvals=(C.shape[0] - 1, C.shape[0] - 1)
        )
    except:
        eigen_max = np.max(np.trace(C))

    def normalize(a, b):
        if (a > b) and not (a < 0 and b < 0):
            a -= 2 * np.pi
        elif (a > b) and (a < 0 and b < 0):
            # a1= a+2*np.pi
            b = 2 * np.pi + b
        return a, b

    def getOverlap(a, b):
        return np.hstack([max(a[0], b[0]), min(a[1], b[1])])

    def intersection(a, b, c, d):
        A, B = normalize(a, b)
        C, D = normalize(c, d)
        if B < C or D < A:
            raise ("empty intersection")
        I_1 = getOverlap([A, B], [C, D])
        return I_1

    b = trunc / np.sqrt(eigen_max)
    x0 = b + 0.1 * np.random.rand(len(b), 1)
    Q = []
    L = np.linalg.cholesky(C / eigen_max)
    mc = 0
    while mc < nsamples + tune:
        nu0 = np.random.randn(b.shape[0], 1)
        nu = L @ nu0
        r_sq = (x0) ** 2 + (nu) ** 2

        thetas_1 = 2 * np.arctan2(nu - np.sqrt(r_sq - b**2), x0 + b)  # +np.pi
        thetas_2 = 2 * np.arctan2(nu + np.sqrt(r_sq - b**2), x0 + b)  # +np.pi
        I1 = [-2 * np.pi, 2 * np.pi]
        vmin = -2 * np.pi
        vmax = 2 * np.pi
        empty = True

        for i in range(thetas_1.shape[0]):
            if np.isnan(thetas_1[i, 0] + thetas_2[i, 0]) == 0:
                empty = False
                eps = np.mod(abs(thetas_2[i, 0] - thetas_1[i, 0]), 2 * np.pi) * 0.001
                # print([thetas_1[i],thetas_2[i]])
                if x0[i] * np.cos(thetas_1[i] + eps) + nu[i] * np.sin(
                    thetas_1[i] + eps
                ) >= trunc[i] / np.sqrt(eigen_max):
                    vmin = thetas_1[i]
                    vmax = thetas_2[i]
                elif x0[i] * np.cos(thetas_1[i] - eps) + nu[i] * np.sin(
                    thetas_1[i] - eps
                ) >= trunc[i] / np.sqrt(eigen_max):
                    vmin = thetas_2[i]
                    vmax = thetas_1[i]
                I1 = intersection(np.copy(I1[0]), np.copy(I1[1]), vmin, vmax)
        if empty == False:
            theta = I1[0] + np.random.rand(1) * np.mod((I1[1] - I1[0]), 2 * np.pi)
            # print(theta)
            x_sample = x0 * np.cos(theta) + nu * np.sin(theta)

            if np.min(-trunc / np.sqrt(eigen_max) + x_sample) < 0:
                raise ("error: the constraint is violated")
            Q.append(x_sample[:, 0] * np.sqrt(eigen_max) + mean)
            x0 = x_sample
            mc = mc + 1
            if progress == True:
                printProgressBar(
                    mc,
                    nsamples + tune,
                    prefix="Progress:",
                    suffix="Complete",
                    length=50,
                )
        else:
            theta = np.random.rand(1) * 2 * np.pi
            x_sample = x0 * np.cos(theta) + nu * np.sin(theta)
            if np.min(-trunc / np.sqrt(eigen_max) + x_sample) < 0:
                raise ("error: the constraint is violated")
            Q.append(x_sample[:, 0] * np.sqrt(eigen_max) + mean)
            x0 = x_sample
            mc = mc + 1
            if progress == True:
                printProgressBar(
                    mc,
                    nsamples + tune,
                    prefix="Progress:",
                    suffix="Complete",
                    length=50,
                )
    res = np.array(Q)[tune:, :]
    return res


# Transformations
class logexp:
    def __init__(self):
        self.name = "logexp"

    def transform(self, x):
        return np.log(x)

    def inverse_transform(self, x):
        return np.exp(x)


class identity:
    def __init__(self):
        self.name = "identity"

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


# Params dictionary
class DictVectorizer:
    def __init__(self):
        self.Name = []
        self.Size = []
        self.Bounds = []
        self.Transforms = []

    def fit_transform(self, x):
        self.Name = []
        self.Size = []
        Vec = np.array([])
        TBounds = np.empty((0, 2), float)
        self.Transforms = []
        for f, v in x.items():
            self.Name.append(f)
            self.Size.append(v["value"].shape)
            assert v["value"].size == v["range"].shape[0]
            value = np.clip(
                v["value"].flatten(),
                np.vstack(v["range"])[:, 0],
                np.vstack(v["range"])[:, 1],
            )  # clip values outside bounds
            transformed = v["transform"].transform(
                value
            )  # apply transformation to parameters
            Vec = np.hstack([Vec, transformed])  # clip values outside bounds
            TBounds = np.vstack([TBounds, v["transform"].transform(v["range"])])
            self.Transforms.append(v["transform"])
        return Vec, TBounds

    def inverse_transform(self, Vec, Bounds):
        pp = {}
        prev = 0
        for i in range(len(self.Size)):
            if len(self.Size[i]) == 1:
                value = Vec[prev : prev + self.Size[i][0]].reshape(self.Size[i])
                pp[self.Name[i]] = {
                    "value": self.Transforms[i].inverse_transform(value),
                    "range": self.Transforms[i].inverse_transform(
                        Bounds[prev : prev + self.Size[i][0]]
                    ),
                    "transform": self.Transforms[i],
                }
                prev = prev + self.Size[i][0]
            else:
                value = Vec[prev : prev + self.Size[i][0] * self.Size[i][1]].reshape(
                    self.Size[i]
                )
                pp[self.Name[i]] = {
                    "value": self.Transforms[i].inverse_transform(value),
                    "range": self.Transforms[i].inverse_transform(
                        Bounds[prev : prev + self.Size[i][0] * self.Size[i][1]]
                    ),
                    "transform": self.Transforms[i],
                }
                prev = prev + self.Size[i][0] * self.Size[i][1]
        return pp
