#SVT.py
#Created by ImKe on 2020/2/28
#Copyright © 2020 ImKe. All rights reserved.

import numpy as np
from numpy import linalg as la
from sparsesvd import sparsesvd
from scipy.sparse.linalg import norm
import scipy.sparse as ss
import scipy.io
import random

def SVT(M1, iter_num):
    n1, n2 = M1.shape
    total_num = len(M1.nonzero()[0])
    proportion = 1.0
    idx = random.sample(range(total_num), int(total_num * proportion))
    Omega = (M1.nonzero()[0][idx], M1.nonzero()[1][idx])
    p = 0.5
    tau = 20000
    delta = 2
    maxiter = iter_num
    tol = 0.001
    incre = 5

    # SVT
    r = 0
    b = M1[Omega]
    P_Omega_M = ss.csr_matrix((np.ravel(b), Omega), shape=(n1, n2))
    normProjM = norm(P_Omega_M)
    k0 = np.ceil(tau / (delta * normProjM))
    Y = k0 * delta * P_Omega_M
    iternum = 0
    rmse = []

    for k in range(maxiter):
        # print (str(k+1) + ' iterative.')
        s = r + 1
        while True:
            u1, s1, v1 = sparsesvd(ss.csc_matrix(Y), s)
            if s1[s - 1] <= tau: break
            s = min(s + incre, n1, n2)
            if s == min(n1, n2): break

        r = np.sum(s1 > tau)
        U = u1.T[:, :r]
        V = v1[:r, :]
        S = s1[:r] - tau
        x = (U * S).dot(V)
        x_omega = ss.csr_matrix((x[Omega], Omega), shape=(n1, n2))

        if norm(x_omega - P_Omega_M) / norm(P_Omega_M) < tol:
            break

        diff = P_Omega_M - x_omega
        Y += delta * diff
        rmse_current = float(la.norm(M1[M1.nonzero()] - x[M1.nonzero()]) / np.sqrt(len(x[M1.nonzero()])))
        print('Iter %d , RMSE %.3f' % (iternum, rmse_current))
        rmse.append(rmse_current)
        iternum += 1

    return rmse