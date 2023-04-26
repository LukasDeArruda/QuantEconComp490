import quantecon
from scipy.linalg import inv
import numpy as np
from scipy.linalg import solve
from ._matrix_eqn import solve_discrete_riccati, solve_discrete_riccati_system

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

        elif type(self) == quantecon._lqcontrol.LQMarkov:
            beta, Π = self.beta, self.Π
            m, n, k = self.m, self.n, self.k
            As, Bs, Cs = self.As, self.Bs, self.Cs
            Qs, Rs, Ns = self.Qs, self.Rs, self.Ns

            # == Solve for P(s) by iterating discrete riccati system== #
            Ps = solve_discrete_riccati_system(Π, As, Bs, Cs, Qs, Rs, Ns, beta,
                                               max_iter=max_iter)

            # == calculate F and d == #
            Fs = np.array([np.empty((k, n)) for i in range(m)])
            X = np.empty((m, m))
            sum1, sum2 = np.empty((k, k)), np.empty((k, n))
            for i in range(m):
                # CCi = C_i C_i'
                CCi = Cs[i] @ Cs[i].T
                sum1[:, :] = 0.
                sum2[:, :] = 0.
                for j in range(m):
                    # for F
                    sum1 += beta * Π[i, j] * Bs[i].T @ Ps[j] @ Bs[i]
                    sum2 += beta * Π[i, j] * Bs[i].T @ Ps[j] @ As[i]

                    # for d
                    X[j, i] = np.trace(Ps[j] @ CCi)

                Fs[i][:, :] = solve(Qs[i] + sum1, sum2 + Ns[i])

            ds = solve(np.eye(m) - beta * Π,
                       np.diag(beta * Π @ X).reshape((m, 1))).flatten()

            self.Ps, self.ds, self.Fs = Ps, ds, Fs

            return Ps, ds, Fs
