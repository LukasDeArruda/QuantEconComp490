import numpy as np
from scipy.signal import dlsim

import quantecon
from quantecon.util import check_random_state
import numpy as np
from numba import jit

@jit(nopython=True)
def simulate_linear_model(A, x0, v, ts_length):
    r"""
    This is a separate function for simulating a vector linear system of
    the form

    .. math::

        x_{t+1} = A x_t + v_t

    given :math:`x_0` = x0

    Here :math:`x_t` and :math:`v_t` are both n x 1 and :math:`A` is n x n.

    The purpose of separating this functionality out is to target it for
    optimization by Numba.  For the same reason, matrix multiplication is
    broken down into for loops.

    Parameters
    ----------
    A : array_like or scalar(float)
        Should be n x n
    x0 : array_like
        Should be n x 1.  Initial condition
    v : np.ndarray
        Should be n x ts_length-1.  Its t-th column is used as the time t
        shock :math:`v_t`
    ts_length : int
        The length of the time series

    Returns
    -------
    x : np.ndarray
        Time series with ts_length columns, the t-th column being :math:`x_t`
    """
    A = np.asarray(A)
    n = A.shape[0]
    x = np.empty((n, ts_length))
    x[:, 0] = x0
    for t in range(ts_length-1):
        x[:, t+1] = A @ (x[:, t]) + v[:, t]
        for i in range(n):
            x[i, t+1] = v[i, t]                   # Shock
            for j in range(n):
                x[i, t+1] += A[i, j] * x[j, t]   # Dot Product
    return x

class TimeSeriesModel:
    def __init__(self, params):
        """
        Initialize the TimeSeriesModel class with common parameters.
        """
        self.params = params


    def simulate(self, ts_length=90, random_state=None):
        """
        Compute a simulated sample path for the time series model.

        Parameters
        ----------
        ts_length : int, optional (default=90)
            Number of periods to simulate for.

        random_state : int or np.random.RandomState/Generator, optional
            Random seed (integer) or np.random.RandomState or Generator
            instance to set the initial state of the random number
            generator for reproducibility. If None, a randomly
            initialized RandomState is used.

        Returns
        -------
        vals : array_like, shape (ts_length,)
            A simulation of the time series model.

        """
        if type(self) == quantecon._arma.ARMA:
            print(type(self))
            random_state = check_random_state(random_state)

            sys = self.ma_poly, self.ar_poly, 1
            u = random_state.standard_normal((ts_length, 1)) * self.sigma
            vals = dlsim(sys, u)[1]
            vals = vals.flatten()
            x = 0
            y = 0
        elif type(self) == quantecon._lss.LinearStateSpace:
            random_state = check_random_state(random_state)
            vals = []
            x0 = random_state.multivariate_normal(self.mu_0.flatten(),
                                                  self.Sigma_0)
            w = random_state.standard_normal((self.m, ts_length - 1))
            v = self.C @ w  # Multiply each w_t by C to get v_t = C w_t
            # == simulate time series == #
            x = simulate_linear_model(self.A, x0, v, ts_length)

            if self.H is not None:
                v = random_state.standard_normal((self.l, ts_length))
                y = self.G @ x + self.H @ v
            else:
                y = self.G @ x

        return vals, x, y
