from abc import ABCMeta, abstractmethod

import numpy as np

# import numpy.matlib
# import matplotlib
# # matplotlib.use("Agg")
# import matplotlib.pyplot as plt

"""
https://www.sfu.ca/~ssurjano/index.html
Test functions in above site.
All functions should be maximized.
Parameters:
d : input dimension
M : number of fidelity
"""


class test_func:
    __metaclass__ = ABCMeta

    def mf_values(self, input_list):
        """
        return each fidelity output list

        Parameters:
        -----------
            input_list : list of numpy array
            list size is the number of fidelity M
            each numpy array size is (N_m \times d), N_m is the number of data of each fidelity.

        Returns:
        --------
            output_list : list of numpy array
            each numpy array size is (N_m, 1)
        """
        func_values_list = []
        for m in range(len(input_list)):
            if np.size(input_list[m]) != 0:
                func_values_list.append(self.values(input_list[m], fidelity=m))
            else:
                func_values_list.append(np.array([]))
        return func_values_list

    def mf_costs(self, input_list):
        """
        return each fidelity cost list depending on x

        Parameters:
        -----------
            input_list : list of numpy array
            list size is the number of fidelity M
            each numpy array size is (N_m \times d), N_m is the number of data of each fidelity.

        Returns:
        --------
            output_list : list of numpy array
            each numpy array size is (N_m, 1)
        """
        func_costs_list = []
        for m in range(len(input_list)):
            if np.size(input_list[m]) != 0:
                func_costs_list.append(self.costs(input_list[m], fidelity=m))
            else:
                func_costs_list.append(np.array([]))
        return func_costs_list

    @abstractmethod
    def values(self, input, fidelity=None):
        pass

    @abstractmethod
    def costs(self, input, fidelity=None):
        pass


def standard_length_scale(bounds):
    return (bounds[1] - bounds[0]) / 2.0


class Beale(test_func):
    """
    Beale Function: d = 2, M = 2
    Three constants is changed to make low fidelity function.
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-4.5, -4.5], [4.5, 4.5]])
        self.d = 2
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 0

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, cons_list=[1.2, 2.5, 2.5])
        elif fidelity == 1:
            return self._common_processing(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _common_processing(self, input, cons_list=None):
        if cons_list is None:
            cons_list = [1.5, 2.25, 2.625]
        first_term = (cons_list[0] - input[:, 0] + input[:, 0] * input[:, 1]) ** 2
        second_term = (cons_list[1] - input[:, 0] + input[:, 0] * input[:, 1] ** 2) ** 2
        third_term = (cons_list[2] - input[:, 0] + input[:, 0] * input[:, 1] ** 3) ** 2
        return -np.c_[(first_term + second_term + third_term)]


class HartMann3(test_func):
    """
    HartMann 3-dimensional function: d = 3, M = 3
    The alpha is minused constant (fidelity=0->0.2, fidelity=1->0.1) to make low fidelity functions.
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0], [1, 1, 1]])
        self.d = 3
        self.M = 3
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 3.86278

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, alpha_error=0.2)
        elif fidelity == 1:
            return self._common_processing(input, alpha_error=0.1)
        elif fidelity == 2:
            return self._common_processing(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _common_processing(self, input, alpha_error=0):
        alpha = np.array([1.0, 1.2, 3.0, 3.2]) - alpha_error
        A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = (
            np.array(
                [
                    [3689, 1170, 2673],
                    [4699, 4387, 7470],
                    [1091, 8732, 5547],
                    [381, 5743, 8828],
                ]
            )
            * 1e-4
        )

        values = 0
        for i in range(4):
            inner = 0
            for j in range(3):
                inner -= A[i, j] * (input[:, j] - P[i, j]) ** 2
            values += alpha[i] * np.power(np.e, inner)
        return np.c_[values]


class HartMann4(test_func):
    """
    HartMann 4-dimensional function: d = 4, M = 3
    The alpha is minused constant (fidelity=0->0.2, fidelity=1->0.1) to make low fidelity functions.
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        self.d = 4
        self.M = 3
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 3.135474

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, alpha_error=0.2)
        elif fidelity == 1:
            return self._common_processing(input, alpha_error=0.1)
        elif fidelity == 2:
            return self._common_processing(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _common_processing(self, input, alpha_error=0):
        alpha = np.array([1.0, 1.2, 3.0, 3.2]) - alpha_error
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = (
            np.array(
                [
                    [1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381],
                ]
            )
            * 1e-4
        )

        values = 0
        for i in range(4):
            inner = 0
            for j in range(4):
                inner -= A[i, j] * (input[:, j] - P[i, j]) ** 2
            values += alpha[i] * np.power(np.e, inner)
        return np.c_[(values - 1.1) / 0.839]


class HartMann6(test_func):
    """
    HartMann 6-dimensional function: d = 6, M = 3
    The alpha is minused constant (fidelity=0->0.2, fidelity=1->0.1) to make low fidelity functions.
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
        self.d = 6
        self.M = 3
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 3.32237

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, alpha_error=0.2)
        elif fidelity == 1:
            return self._common_processing(input, alpha_error=0.1)
        elif fidelity == 2:
            return self._common_processing(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _common_processing(self, input, alpha_error=0):
        alpha = np.array([1.0, 1.2, 3.0, 3.2]) - alpha_error
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )

        P = (
            np.array(
                [
                    [1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381],
                ]
            )
            * 1e-4
        )

        values = 0
        for i in range(4):
            inner = 0
            for j in range(6):
                inner -= A[i, j] * (input[:, j] - P[i, j]) ** 2
            values += alpha[i] * np.power(np.e, inner)
        return np.c_[values]


class Borehole(test_func):
    """
    Borehole function: d = 8, M = 2
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array(
            [
                [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855],
                [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045],
            ]
        )
        self.d = 8
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = (
            -7.8198
        )  # On Efficient Global Optimization via Universal Kriging Surrogate Models

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        numerator = 2 * np.pi * input[:, 2] * (input[:, 3] - input[:, 5])
        log_ratio = np.log(input[:, 1] / input[:, 0])
        denominator = log_ratio * (
            1
            + (2 * input[:, 6] * input[:, 2])
            / (log_ratio * input[:, 0] ** 2 * input[:, 7])
            + input[:, 2] / input[:, 4]
        )
        values = numerator / denominator
        return -np.c_[values]

    def _low_fidelity_values(self, input):
        numerator = 5 * input[:, 2] * (input[:, 3] - input[:, 5])
        log_ratio = np.log(input[:, 1] / input[:, 0])
        denominator = log_ratio * (
            1.5
            + (2 * input[:, 6] * input[:, 2])
            / (log_ratio * input[:, 0] ** 2 * input[:, 7])
            + input[:, 2] / input[:, 4]
        )
        values = numerator / denominator
        return -np.c_[values]


class Branin(test_func):
    """
    Branin function: d = 2, M = 2
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-5, 0], [10, 15]])
        self.d = 2
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = -0.397887

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        first_term = a * (input[:, 1] - b * input[:, 0] ** 2 + c * input[:, 0] - r) ** 2
        second_term = s * (1 - t) * np.cos(input[:, 0])
        return -np.c_[(first_term + second_term + s)]

    def _low_fidelity_values(self, input):
        a = 1.1
        b = 5.0 / (4 * np.pi**2)
        c = 4 / np.pi
        r = 5
        s = 8
        t = 1 / (10 * np.pi)
        first_term = a * (input[:, 1] - b * input[:, 0] ** 2 + c * input[:, 0] - r) ** 2
        second_term = s * (1 - t) * np.cos(input[:, 0])
        return -np.c_[(first_term + second_term + s)]


class Colville(test_func):
    """
    Colville function: d = 4, M = 2
    low_fidelity function is
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-10, -10, -10, -10], [10, 10, 10, 10]])
        self.d = 4
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 0

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, cons_list=[90, 0.9, 0.9, 100, 9, 20])
        elif fidelity == 1:
            return self._common_processing(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _common_processing(self, input, cons_list=None):
        if cons_list is None:
            cons_list = [100, 1, 1, 90, 10.1, 19.8]
        term1 = cons_list[0] * (input[:, 0] ** 2 - input[:, 1]) ** 2
        term2 = cons_list[1] * (input[:, 0] - 1) ** 2
        term3 = cons_list[2] * (input[:, 2] - 1) ** 2
        term4 = cons_list[3] * (input[:, 2] ** 2 - input[:, 3]) ** 2
        term5 = cons_list[4] * ((input[:, 1] - 1) ** 2 + (input[:, 3] - 1) ** 2)
        term6 = cons_list[5] * (input[:, 1] - 1) * (input[:, 3] - 1)
        return -np.c_[(term1 + term2 + term3 + term4 + term5 + term6)]


class CurrinExp(test_func):
    """
    Currin exponential function : d=2, M=2
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0], [1, 1]])
        self.d = 2
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = None

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        FormerTerm = 1 - np.exp(-1 / (2.0 * input[:, 1]))
        LatterTerm = (
            2300 * input[:, 0] ** 3 + 1900 * input[:, 0] ** 2 + 2092 * input[:, 0] + 60
        ) / (100 * input[:, 0] ** 3 + 500 * input[:, 0] ** 2 + 4 * input[:, 0] + 20)
        values = FormerTerm * LatterTerm
        return np.c_[values]

    def _low_fidelity_values(self, input):
        input1 = np.copy(input) + 0.05

        input2 = np.copy(input)
        input2[:, 0] = input2[:, 0] + 0.05
        input2[:, 1] = input2[:, 1] - 0.05
        input2[:, 1][input2[:, 1] < 0] = 0

        input3 = np.copy(input)
        input3[:, 0] = input3[:, 0] - 0.05
        input3[:, 1] = input3[:, 1] + 0.05

        input4 = np.copy(input)
        input4[:, 0] = input4[:, 0] - 0.05
        input4[:, 1] = input4[:, 1] - 0.05
        input4[:, 1][input4[:, 1] < 0] = 0

        values = (
            self._high_fidelity_values(input1)
            + self._high_fidelity_values(input2)
            + self._high_fidelity_values(input3)
            + self._high_fidelity_values(input4)
        ) / 4.0
        return np.c_[values]


class Forrester(test_func):
    """
    Forrester function : d=1, M=2
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0], [1]])
        self.d = 1
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        return -np.c_[((6 * input - 2) ** 2 * np.sin(12 * input - 4))]

    def _low_fidelity_values(self, input):
        A = 0.5
        B = 10
        C = -5
        values = self._high_fidelity_values(input)
        return np.c_[A * values - B * (input - 0.5) + C]


class Styblinski_tang(test_func):
    """
    Styblinski-tang function : d=2, M=2
    I fix the dimension.
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-5, -5], [5, 5]])
        self.d = 2
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        d = 2  # np.size(input, 1)
        values = 0
        for i in range(d):
            values += input[:, i] ** 4 - 16 * input[:, i] ** 2 + 5 * input[:, i]
        return -np.c_[values / 2]

    def _low_fidelity_values(self, input):
        d = 2  # np.size(input, 1)
        values = 0
        for i in range(d):
            values += 0.9 * input[:, i] ** 4 - 15 * input[:, i] ** 2 + 6 * input[:, i]
        return -np.c_[values / 2]


class Park1(test_func):
    """
    Park function 1 : d=4, M=2
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        self.d = 4
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        FirstTerm = (
            input[:, 0]
            / 2.0
            * (
                np.sqrt(
                    1
                    + (input[:, 1] + input[:, 2] ** 2) * input[:, 3] / input[:, 0] ** 2
                )
                - 1
            )
        )
        SecondTerm = (input[:, 1] + 3 * input[:, 3]) * np.power(
            np.e, 1 + np.sin(input[:, 2])
        )
        values = FirstTerm + SecondTerm
        return -np.c_[values]

    def _low_fidelity_values(self, input):
        values = (1 + np.sin(input[:, 0]) / 10.0) * self._high_fidelity_values(
            input
        ).ravel() - (-2 * input[:, 0] + input[:, 1] ** 2 + input[:, 2] ** 2 + 0.5)
        return np.c_[values]


class Park2(test_func):
    """
    Park function 2 : d=4, M=2
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        self.d = 4
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        values = (
            2.0 / 3.0 * np.power(np.e, input[:, 0] + input[:, 1])
            - input[:, 3] * np.sin(input[:, 2])
            + input[:, 2]
        )
        return -np.c_[values]

    def _low_fidelity_values(self, input):
        values = 1.2 * self._high_fidelity_values(input) + 1
        return values


class Powell(test_func):
    """
    Powell function : d=4, M=2
    I fix the dimension.
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-4, -4, -4, -4], [5, 5, 5, 5]])
        self.d = 4
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        term1 = (input[:, 0] - 10 * input[:, 1]) ** 2
        term2 = 5 * (input[:, 2] - input[:, 3]) ** 2
        term3 = (input[:, 1] - 2 * input[:, 2]) ** 4
        term4 = 10 * (input[:, 0] ** 2 - input[:, 3]) ** 4
        return -np.c_[term1 + term2 + term3 + term4]

    def _low_fidelity_values(self, input):
        term1 = 0.9 * (input[:, 0] - 10 * input[:, 1]) ** 2
        term2 = 4 * (input[:, 2] - input[:, 3]) ** 2
        term3 = 0.9 * (input[:, 1] - 2 * input[:, 2]) ** 4
        term4 = 9 * (input[:, 0] ** 2 - input[:, 3]) ** 4
        return -np.c_[term1 + term2 + term3 + term4]


class Shekel(test_func):
    """
    Shekel function : d=4, M=2
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0], [10, 10, 10, 10]])
        self.d = 4
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, m=5)
        elif fidelity == 1:
            return self._common_processing(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _common_processing(self, input, m=10):
        beta = np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]) / 10.0
        C = np.array(
            [
                [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
            ]
        )
        values = 0
        for i in range(m):
            inner = 0
            for j in range(4):
                inner += (input[:, j] - C[j, i]) ** 2
            inner += beta[i]
            values += 1 / inner
        return np.c_[values]


##########################################################################################################################################################################################


class Ackley(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 4
        self.bounds = (
            32.768 * np.r_[-1 * np.c_[np.ones(self.d)].T, np.c_[np.ones(self.d)].T]
        )
        # self.bounds = np.array([[-32.768, -32.768, -32.768, -32.768], [32.768, 32.768, 32.768, 32.768]])
        self.M = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.GLOBAL_MAXIMUM = 0

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        a = 20
        b = 0.2
        c = 2 * np.pi

        input_squared_sum = np.sum(np.c_[input] ** 2, axis=1)
        input_cos_sum = np.sum(np.cos(np.c_[c * input]), axis=1)

        values = -1 * (
            -a * np.exp(-b * np.sqrt(input_squared_sum / self.d))
            - np.exp(input_cos_sum / self.d)
            + a
            + np.e
        )
        return np.c_[values]


class Bukin(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.bounds = np.array([[-15.0, -3], [-5, 3]])
        self.M = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.GLOBAL_MAXIMUM = 0

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        values = -1 * (
            100 * np.sqrt(np.abs(input[:, 1] - 0.01 * input[:, 0] ** 2))
            + 0.01 * np.abs(input[:, 0] + 10)
        )
        return np.c_[values]


class Cross_in_tray(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.bounds = np.array([[-10.0, -10], [10, 10]])
        self.M = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.GLOBAL_MAXIMUM = 2.06261

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        values = -1 * (
            -1e-4
            * np.power(
                np.abs(
                    np.sin(input[:, 0])
                    * np.sin(input[:, 1])
                    * np.exp(
                        np.abs(
                            100
                            - np.sqrt(np.sum(np.atleast_2d(input) ** 2, axis=1)) / np.pi
                        )
                    )
                )
                + 1,
                0.1,
            )
        )
        return np.c_[values]


class Eggholder(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.bounds = np.array([[-512.0, -512], [512, 512]])
        self.M = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.GLOBAL_MAXIMUM = 959.6407

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        values = -1 * (
            -(input[:, 1] + 47)
            * np.sin(np.sqrt(np.abs(input[:, 1] + input[:, 0] / 2.0 + 47)))
            - input[:, 0] * np.sin(np.sqrt(np.abs(input[:, 0] - (input[:, 1] + 47))))
        )
        return np.c_[values]


class Holder_table(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.bounds = np.array([[-10.0, -10.0], [10.0, 10.0]])
        self.M = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.GLOBAL_MAXIMUM = 19.208502567886747

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        values = -1 * (
            -np.abs(
                np.sin(input[:, 0])
                * np.cos(input[:, 1])
                * np.exp(np.abs(1 - np.sqrt(np.sum(input**2, axis=1)) / np.pi))
            )
        )
        return np.c_[values]


class Langerman(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.bounds = np.array([[0.0, 0.0], [10.0, 10.0]])
        self.M = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.GLOBAL_MAXIMUM = 4.155809291847781

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        m = 5
        c = np.array([[1.0, 2.0, 5.0, 2.0, 3.0]])
        A = np.array([[3.0, 5.0, 2.0, 1.0, 7.0], [5.0, 2.0, 1.0, 4.0, 9.0]]).T

        tmp_input = []
        for i in range(m):
            tmp_input.append(np.sum((np.atleast_2d(input) - A[i, :]) ** 2, axis=1))
        tmp_input = np.vstack(tmp_input).T

        values = -1 * np.sum(
            c * np.exp(-tmp_input / np.pi) * np.cos(np.pi * tmp_input), axis=1
        )
        return np.c_[values]


class Levy(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 4
        self.bounds = (
            10.0 * np.r_[-1 * np.c_[np.ones(self.d)].T, np.c_[np.ones(self.d)].T]
        )
        # self.bounds = np.array([[-10., -10., -10., -10., -10.], [10., 10., 10., 10., 10.]])
        self.M = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.GLOBAL_MAXIMUM = 0

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        w = 1 + (input - 1) / 4.0

        iterated_term = (w[:, :-1] - 1) ** 2 * (
            1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2
        )
        iterated_term = np.sum(iterated_term, axis=1)

        values = -1 * (
            np.sin(np.pi * w[:, 0]) ** 2
            + (w[:, -1] - 1) ** 2 * np.sin(2 * np.pi * w[:, -1]) ** 2
        )
        return np.c_[values]


class Levy13(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.bounds = np.array([[-10.0, -10.0], [10.0, 10.0]])
        self.M = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.GLOBAL_MAXIMUM = 0

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        values = -1 * (
            np.sin(3 * np.pi * input[:, 0]) ** 2
            + (input[:, 0] - 1) ** 2 * (1 + np.sin(3 * np.pi * input[:, 1]) ** 2)
            + (input[:, 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * input[:, 1]) ** 2)
        )
        return np.c_[values]


class Rastrigin(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 3
        self.bounds = (
            5.12 * np.r_[-1 * np.c_[np.ones(self.d)].T, np.c_[np.ones(self.d)].T]
        )
        # self.bounds = np.array([[-32.768, -32.768, -32.768, -32.768], [32.768, 32.768, 32.768, 32.768]])
        self.M = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.GLOBAL_MAXIMUM = 0

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        values = -1 * (
            10 * self.d + np.sum(input**2 - 10.0 * np.cos(2 * np.pi * input), axis=1)
        )
        return np.c_[values]


class Shubert(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.bounds = np.array([[-5.12, -5.12], [5.12, 5.12]])
        self.M = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.GLOBAL_MAXIMUM = 186.73090883102384

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        indice = np.arange(5) + 1
        first_term = np.sum(
            indice * np.cos((indice + 1) * np.c_[input[:, 0]] + indice), axis=1
        )
        second_term = np.sum(
            indice * np.cos((indice + 1) * np.c_[input[:, 1]] + indice), axis=1
        )
        values = -1 * (first_term * second_term)
        return np.c_[values]


class Schwefel(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 3
        self.bounds = (
            500.0 * np.r_[-1 * np.c_[np.ones(self.d)].T, np.c_[np.ones(self.d)].T]
        )
        # self.bounds = np.array([[-32.768, -32.768, -32.768, -32.768], [32.768, 32.768, 32.768, 32.768]])
        self.M = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.GLOBAL_MAXIMUM = 0

    def values(self, input, fidelity=None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input):
        values = -1 * (
            418.9829 * self.d - np.sum(input * np.sin(np.sqrt(np.abs(input))), axis=1)
        )
        return np.c_[values]
