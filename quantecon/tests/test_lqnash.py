"""
Tests for lqnash.py

"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from quantecon import nnash
from quantecon import LQ
from quantecon.game_theory import random
from quantecon.util.random import check_random_state


class TestLQNash:
    def test_noninteractive(self):
        "Test case for when agents don't interact with each other"
        # Copied these values from test_lqcontrol
        a = np.array([[.95, 0.], [0, .95]])
        b1 = np.array([.95, 0.])
        b2 = np.array([0., .95])
        r1 = np.array([[-.25, 0.], [0., 0.]])
        r2 = np.array([[0., 0.], [0., -.25]])
        q1 = np.array([[-.15]])
        q2 = np.array([[-.15]])
        f1, f2, p1, p2 = nnash(a, b1, b2, r1, r2, q1, q2, 0, 0, 0, 0, 0, 0,
                               tol=1e-8, max_iter=10000)

        alq = a[:1, :1]
        blq = b1[:1].reshape((1, 1))
        rlq = r1[:1, :1]
        qlq = q1

        lq_obj = LQ(qlq, rlq, alq, blq, beta=1.)
        p, f, d = lq_obj.stationary_values()

        assert_allclose(f1, f2[:, ::-1])
        assert_allclose(f1[0, 0], f[0])
        assert_allclose(p1[0, 0], p2[1, 1])
        assert_allclose(p1[0, 0], p[0, 0])

    def test_nnash(self):
        "Use judd test case for nnash. Follows judd.m"
        # Define Parameters
        delta = 0.02
        d = np.array([[-1, 0.5], [0.5, -1]])
        B = np.array([25, 25])
        c1 = np.array([1, -2, 1])
        c2 = np.array([1, -2, 1])
        e1 = np.array([10, 10, 3])
        e2 = np.array([10, 10, 3])
        delta_1 = 1 - delta

        ## Define matrices
        a = np.array([[delta_1, 0, -delta_1*B[0]],
                     [0, delta_1, -delta_1*B[1]],
                     [0, 0, 1]])

        b1 = delta_1 * np.array([[1, -d[0, 0]],
                                [0, -d[1, 0]],
                                [0, 0]])
        b2 = delta_1 * np.array([[0, -d[0, 1]],
                                [1, -d[1, 1]],
                                [0, 0]])

        r1 = -np.array([[0.5*c1[2], 0, 0.5*c1[1]],
                       [0, 0, 0],
                       [0.5*c1[1], 0, c1[0]]])
        r2 = -np.array([[0, 0, 0],
                       [0, 0.5*c2[2], 0.5*c2[1]],
                       [0, 0.5*c2[1], c2[0]]])

        q1 = np.array([[-0.5*e1[2], 0], [0, d[0, 0]]])
        q2 = np.array([[-0.5*e2[2], 0], [0, d[1, 1]]])

        s1 = np.zeros((2, 2))
        s2 = np.copy(s1)

        w1 = np.array([[0, 0],
                      [0, 0],
                      [-0.5*e1[1], B[0]/2.]])
        w2 = np.array([[0, 0],
                      [0, 0],
                      [-0.5*e2[1], B[1]/2.]])

        m1 = np.array([[0, 0], [0, d[0, 1] / 2.]])
        m2 = np.copy(m1)

        # build model and solve it
        f1, f2, p1, p2 = nnash(a, b1, b2, r1, r2, q1, q2, s1, s2, w1, w2, m1,
                               m2)

        aaa = a - b1 @ f1 - b2 @ f2
        aa = aaa[:2, :2]
        tf = np.eye(2)-aa
        tfi = np.linalg.inv(tf)
        xbar = tfi@(aaa[:2, 2])

        # Define answers from matlab. TODO: this is ghetto
        f1_ml = np.array([
               [0.243666582208565,   0.027236062661951, -6.827882928738190],
               [0.392370733875639,   0.139696450885998, -37.734107291009138]
           ])

        f2_ml = np.array([
               [0.027236062661951,   0.243666582208565,  -6.827882928738186],
               [0.139696450885998,   0.392370733875639, -37.734107291009131]
           ])

        xbar_ml = np.array([1.246871007582702, 1.246871007582685])

        assert_allclose(f1, f1_ml)
        assert_allclose(f2, f2_ml)
        assert_allclose(xbar, xbar_ml)

    def test_check_random_state(self):
        int_seed = check_random_state(5)
        int_seed2 = check_random_state(3)
        int_seed3 = check_random_state(1)
        int_seed4 = check_random_state(2)

        none_seed = check_random_state(None)
        assert none_seed == np.random.mtrand._rand
        assert isinstance(int_seed, np.random.RandomState)
        assert int_seed.random() == 0.22199317108973948
        rand_state_seed = check_random_state(int_seed)
        assert rand_state_seed == int_seed

        assert int_seed.randint(1, 5) == 4
        assert int_seed2.randint(1, 5) == 3
        assert int_seed3.randint(100, 204) == 137
        assert int_seed4.randint(30, 50) == 38

        # Test with a negative integer seed (Which raises ValueError)
        with pytest.raises(ValueError) as parameter_error:
            neg_seed = check_random_state(-1)
            isinstance(neg_seed, np.random.RandomState)
        assert parameter_error.type is ValueError

        # Test with a float seed
        with pytest.raises(ValueError) as parameter_error:
            float_seed = check_random_state(0.5)
            isinstance(float_seed, np.random.RandomState)
        assert parameter_error.type is ValueError

        with pytest.raises(ValueError) as error:
            check_random_state("Bad data test")
        assert error.type is ValueError
