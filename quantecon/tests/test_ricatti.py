"""
Tests for solve_discrete_riccati in matrix_eqn.py file

"""
import numpy as np
from numpy.testing import assert_allclose, assert_raises
from quantecon import solve_discrete_riccati
import pytest


def dare_golden_num_float(method):
    val = solve_discrete_riccati(1.0, 1.0, 1.0, 1.0, method=method)
    gold_ratio = (1 + np.sqrt(5)) / 2.
    assert_allclose(val, gold_ratio)


def dare_golden_num_2d(method):
    A, B, R, Q = np.eye(2), np.eye(2), np.eye(2), np.eye(2)
    gold_diag = np.eye(2) * (1 + np.sqrt(5)) / 2.
    val = solve_discrete_riccati(A, B, R, Q, method=method)
    assert_allclose(val, gold_diag)


def dare_tjm_1(method):
    A = [[0.0, 0.1, 0.0],
         [0.0, 0.0, 0.1],
         [0.0, 0.0, 0.0]]
    B = [[1.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0]]
    Q = [[10**5, 0.0, 0.0],
         [0.0, 10**3, 0.0],
         [0.0, 0.0, -10.0]]
    R = [[0.0, 0.0],
         [0.0, 1.0]]
    X = solve_discrete_riccati(A, B, Q, R, method=method)
    Y = np.diag((1e5, 1e3, 0.0))
    assert_allclose(X, Y, atol=1e-07)


def dare_tjm_2(method):
    A = [[0, -1],
         [0, 2]]
    B = [[1, 0],
         [1, 1]]
    Q = [[1, 0],
         [0, 0]]
    R = [[4, 2],
         [2, 1]]
    X = solve_discrete_riccati(A, B, Q, R, method=method)
    Y = np.zeros((2, 2))
    Y[0, 0] = 1
    assert_allclose(X, Y, atol=1e-07)


def dare_tjm_3(method):
    r = 0.5
    I = np.identity(2)
    A = [[2 + r**2, 0],
         [0,        0]]
    A = np.array(A)
    B = I
    R = [[1, r],
         [r, r*r]]
    Q = I - (A.T @ A) + (A.T @ np.linalg.solve(R + I, A))
    X = solve_discrete_riccati(A, B, Q, R, method=method)
    Y = np.identity(2)
    assert_allclose(X, Y, atol=1e-07)


_test_funcs = [
    dare_golden_num_float, dare_golden_num_2d,
    dare_tjm_1, dare_tjm_2, dare_tjm_3
]


_test_methods = [
    'doubling', 'qz'
]


@pytest.mark.parametrize("test_func", _test_funcs)
@pytest.mark.parametrize("test_method", _test_methods)
def test_solve_discrete_riccati(test_func, test_method):
    test_func(test_method)


def test_solve_discrete_riccati_invalid_method():
    method = 'invalid_method'
    assert_raises(ValueError, _test_funcs[0], method)
