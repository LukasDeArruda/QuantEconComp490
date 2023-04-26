import quantecon._kalman
import quantecon._lqcontrol
from ._matrix_eqn import solve_discrete_riccati
from scipy.linalg import inv
import numpy as np
from scipy.linalg import solve

class Consolidated:
    def __init__(self):
        # Initialize the ConsolidatedClass instance
        print('Consolidated Class')

    def stationary_values(self, method='doubling', max_iter=1000):
        # Consolidated implementation of stationary_values method
        # You can reuse the common logic from Kalman and LQ classes here
        # === simplify notation === #
        if type(self) == quantecon._kalman.Kalman:
            A, C, G, H = self.A, self.C, self.G, self.H
            Q, R = C @ C.T, H @ H.T

            # === solve Riccati equation, obtain Kalman gain === #
            Sigma_infinity = solve_discrete_riccati(A.T, G.T, Q, R, method=method)
            temp1 = A @ Sigma_infinity @ G.T
            temp2 = inv(G @ Sigma_infinity @ G.T + R)
            K_infinity = temp1 @ temp2

            # == record as attributes and return == #
            self._Sigma_infinity, self._K_infinity = Sigma_infinity, K_infinity
            return Sigma_infinity, K_infinity
        elif type(self) == quantecon._lqcontrol.LQ:
            Q, R, A, B, N, C = self.Q, self.R, self.A, self.B, self.N, self.C

            # === solve Riccati equation, obtain P === #
            A0, B0 = np.sqrt(self.beta) * A, np.sqrt(self.beta) * B
            P = solve_discrete_riccati(A0, B0, R, Q, N, method=method)

            # == Compute F == #
            S1 = Q + self.beta * np.dot(B.T, np.dot(P, B))
            S2 = self.beta * np.dot(B.T, np.dot(P, A)) + N
            F = solve(S1, S2)

            # == Compute d == #
            if self.beta == 1:
                d = 0
            else:
                d = self.beta * np.trace(np.dot(P, np.dot(C, C.T))) / (1 - self.beta)

            # == Bind states and return values == #
            self.P, self.F, self.d = P, F, d

            return P, F, d

