"""
Implements the Kalman filter for a linear Gaussian state space model.

References
----------

https://lectures.quantecon.org/py/kalman.html

"""
from textwrap import dedent

import numpy
import numpy as np
from scipy.linalg import inv

from ._consolidated_model import Consolidated
from ._lss import LinearStateSpace
from ._matrix_eqn import solve_discrete_riccati

class Kalman(Consolidated):
    r"""
    Implements the Kalman filter for the Gaussian state space model

    .. math::

        x_{t+1} = A x_t + C w_{t+1} \\
        y_t = G x_t + H v_t

    Here :math:`x_t` is the hidden state and :math:`y_t` is the measurement.
    The shocks :math:`w_t` and :math:`v_t` are iid standard normals. Below
    we use the notation

    .. math::

        Q := CC'
        R := HH'


    Parameters
    ----------
    ss : instance of LinearStateSpace
        An instance of the quantecon.lss.LinearStateSpace class
    x_hat : scalar(float) or array_like(float), optional(default=None)
        An n x 1 array representing the mean x_hat of the
        prior/predictive density.  Set to zero if not supplied.
    Sigma : scalar(float) or array_like(float), optional(default=None)
        An n x n array representing the covariance matrix Sigma of
        the prior/predictive density.  Must be positive definite.
        Set to the identity if not supplied.

    Attributes
    ----------
    Sigma, x_hat : as above
    Sigma_infinity : array_like or scalar(float)
        The infinite limit of Sigma_t
    K_infinity : array_like or scalar(float)
        The stationary Kalman gain.


    References
    ----------

    https://lectures.quantecon.org/py/kalman.html

    """

    def __init__(self, A=None, C=None, G=None, H=None, mu_0=None, Sigma_0=None, x_hat=None, Sigma=None):
        print(H)
        if type(A) is LinearStateSpace:
            self.ss = A
            self.A = self.ss.A
            self.C = self.ss.C
            self.G = self.ss.G
            self.H = self.ss.H
        elif type(A) in [np.ndarray, list, np.array]:
            self.A = A
            self.C = C
            self.G = G
            self.H = H
            self.ss = LinearStateSpace(A, C, G, H, mu_0, Sigma_0)
            # print(A, C, G, H)
            # print("ss is", ss)
        else:
            print("Invalid argument specified")
            return
        self.set_state(x_hat, Sigma)
        self._K_infinity = None
        self._Sigma_infinity = None


    def set_state(self, x_hat, Sigma):
        if Sigma is None:
            self.Sigma = np.identity(self.ss.n)
        else:
            self.Sigma = np.atleast_2d(Sigma)
        if x_hat is None:
            self.x_hat = np.zeros((self.ss.n, 1))
        else:
            self.x_hat = np.atleast_2d(x_hat)
            self.x_hat.shape = self.ss.n, 1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        m = """\
        Kalman filter:
          - dimension of state space          : {n}
          - dimension of observation equation : {k}
        """
        return dedent(m.format(n=self.ss.n, k=self.ss.k))

    @property
    def Sigma_infinity(self):
        if self._Sigma_infinity is None:
            self.stationary_values()
        return self._Sigma_infinity

    @property
    def K_infinity(self):
        if self._K_infinity is None:
            self.stationary_values()
        return self._K_infinity

    def whitener_lss(self):
        r"""
        This function takes the linear state space system
        that is an input to the Kalman class and it converts
        that system to the time-invariant whitener represenation
        given by

        .. math::

            \tilde{x}_{t+1}^* = \tilde{A} \tilde{x} + \tilde{C} v
            a = \tilde{G} \tilde{x}

        where

        .. math::

            \tilde{x}_t = [x+{t}, \hat{x}_{t}, v_{t}]

        and

        .. math::

            \tilde{A} =
            \begin{bmatrix}
            A  & 0    & 0  \\
            KG & A-KG & KH \\
            0  & 0    & 0 \\
            \end{bmatrix}

        .. math::

            \tilde{C} =
            \begin{bmatrix}
            C & 0 \\
            0 & 0 \\
            0 & I \\
            \end{bmatrix}

        .. math::

            \tilde{G} =
            \begin{bmatrix}
            G & -G & H \\
            \end{bmatrix}

        with :math:`A, C, G, H` coming from the linear state space system
        that defines the Kalman instance

        Returns
        -------
        whitened_lss : LinearStateSpace
            This is the linear state space system that represents
            the whitened system
        """
        K = self.K_infinity

        # Get the matrix sizes
        n, k, m, l = self.ss.n, self.ss.k, self.ss.m, self.ss.l
        A, C, G, H = self.A, self.C, self.G, self.H

        Atil = np.vstack([np.hstack([A, np.zeros((n, n)), np.zeros((n, l))]),
                          np.hstack([K @ G,
                                     A-(K @ G),
                                     K @ H]),
                          np.zeros((l, 2*n + l))])

        Ctil = np.vstack([np.hstack([C, np.zeros((n, l))]),
                          np.zeros((n, m+l)),
                          np.hstack([np.zeros((l, m)), np.eye(l)])])

        Gtil = np.hstack([G, -G, H])

        whitened_lss = LinearStateSpace(Atil, Ctil, Gtil)
        self.whitened_lss = whitened_lss

        return whitened_lss

    def prior_to_filtered(self, y):
        r"""
        Updates the moments (x_hat, Sigma) of the time t prior to the
        time t filtering distribution, using current measurement :math:`y_t`.

        The updates are according to

        .. math::

            \hat{x}^F = \hat{x} + \Sigma G' (G \Sigma G' + R)^{-1}
                (y - G \hat{x})
            \Sigma^F = \Sigma - \Sigma G' (G \Sigma G' + R)^{-1} G
                \Sigma

        Parameters
        ----------
        y : scalar or array_like(float)
            The current measurement

        """
        # === simplify notation === #
        G, H = self.G, self.H
        R = (H @ H.T)

        # === and then update === #
        y = np.atleast_2d(y)
        y.shape = self.ss.k, 1
        E = (self.Sigma @ G.T)
        F = ((G @ self.Sigma) @ G.T) + R
        M = (E @ inv(F))
        self.x_hat = self.x_hat + (M @ (y - (G @ self.x_hat)))
        self.Sigma = self.Sigma - (M @ (G @  self.Sigma))

    def filtered_to_forecast(self):
        """
        Updates the moments of the time t filtering distribution to the
        moments of the predictive distribution, which becomes the time
        t+1 prior

        """
        # === simplify notation === #
        A, C = self.A, self.C
        Q = (C @ C.T)

        # === and then update === #
        self.x_hat = (A @ self.x_hat)
        self.Sigma = (A @ (self.Sigma @ A.T)) + Q

    def update(self, y):
        """
        Updates x_hat and Sigma given k x 1 ndarray y.  The full
        update, from one period to the next

        Parameters
        ----------
        y : np.ndarray
            A k x 1 ndarray y representing the current measurement

        """
        self.prior_to_filtered(y)
        self.filtered_to_forecast()


    def stationary_coefficients(self, j, coeff_type='ma'):
        """
        Wold representation moving average or VAR coefficients for the
        steady state Kalman filter.

        Parameters
        ----------
        j : int
            The lag length
        coeff_type : string, either 'ma' or 'var' (default='ma')
            The type of coefficent sequence to compute.  Either 'ma' for
            moving average or 'var' for VAR.
        """
        # == simplify notation == #
        A, G = self.A, self.G
        K_infinity = self.K_infinity
        # == compute and return coefficients == #
        coeffs = []
        i = 1
        if coeff_type == 'ma':
            coeffs.append(np.identity(self.ss.k))
            P_mat = A
            P = np.identity(self.ss.n)  # Create a copy
        elif coeff_type == 'var':
            coeffs.append(G @ K_infinity)
            P_mat = A - K_infinity @ G
            P = np.copy(P_mat)  # Create a copy
        else:
            raise ValueError("Unknown coefficient type")
        while i <= j:
            coeffs.append(G @ P @ K_infinity)
            P = P @ P_mat
            i += 1
        return coeffs

    def stationary_innovation_covar(self):
        # == simplify notation == #
        H, G = self.H, self.G
        R = H @ H.T
        Sigma_infinity = self.Sigma_infinity

        return (G @ Sigma_infinity @ G.T) + R
