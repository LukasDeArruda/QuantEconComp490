"""
Tests for the kalman.py

"""
import numpy as np
from numpy.testing import assert_allclose
from quantecon import LinearStateSpace
from quantecon import Kalman


class TestKalman:

    def setup_method(self):
        # Initial Values
        self.A = np.array([[.95, 0], [0., .95]])
        self.C = np.eye(2) * np.sqrt(0.5)
        self.G = np.eye(2) * .5
        self.H = np.eye(2) * np.sqrt(0.2)

        self.Q = self.C @ self.C.T
        self.R = self.H @ self.H.T

        ss = LinearStateSpace(self.A, self.C, self.G, self.H)

        #self.kf = Kalman(ss)
        self.kf = Kalman(A=self.A, C=self.C, G=self.G, H=self.H)

        self.methods = ['doubling', 'qz']


    def teardown_method(self):
        del self.kf


    def test_stationarity(self):
        A, Q, G, R = self.A, self.Q, self.G, self.R
        kf = self.kf

        for method in self.methods:
            sig_inf, kal_gain = kf.stationary_values(method=method)

            mat_inv = np.linalg.inv(G@(sig_inf)@(G.T) + R)

            # Compute the kalmain gain and sigma infinity according to the
            # recursive equations and compare
            kal_recursion = A @ sig_inf @ G.T @ mat_inv
            sig_recursion = (A@(sig_inf)@(A.T) -
                                kal_recursion@(G)@(sig_inf)@(A.T) + Q)

            assert_allclose(kal_gain, kal_recursion, rtol=1e-4, atol=1e-1)
            assert_allclose(sig_inf, sig_recursion, rtol=1e-4, atol=1e-1)


    def test_update_using_stationary(self):
        kf = self.kf

        for method in self.methods:
            sig_inf, kal_gain = kf.stationary_values(method=method)

            kf.set_state(np.zeros((2, 1)), sig_inf)

            kf.update(np.zeros((2, 1)))

            assert_allclose(kf.Sigma, sig_inf, rtol=1e-4, atol=1e-1)
            assert_allclose(kf.x_hat.squeeze(), np.zeros(2),
                            rtol=1e-4, atol=1e-2)


    def test_update_nonstationary(self):
        A, Q, G, R = self.A, self.Q, self.G, self.R
        kf = self.kf

        curr_x, curr_sigma = np.ones((2, 1)), np.eye(2) * .75
        y_observed = np.ones((2, 1)) * .75

        kf.set_state(curr_x, curr_sigma)
        kf.update(y_observed)

        mat_inv = np.linalg.inv(G@(curr_sigma)@(G.T) + R)
        curr_k = A @ curr_sigma @ G.T @ mat_inv
        new_sigma = (A@(curr_sigma)@(A.T) -
                    curr_k@(G)@(curr_sigma)@(A.T) + Q)

        new_xhat = A@(curr_x) + curr_k@(y_observed - G@(curr_x))

        assert_allclose(kf.Sigma, new_sigma, rtol=1e-4, atol=1e-2)
        assert_allclose(kf.x_hat, new_xhat, rtol=1e-4, atol=1e-2)
