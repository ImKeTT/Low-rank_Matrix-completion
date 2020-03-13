#OtraceEEC_python.py
#Created by ImKe on 2020/3/5
#Copyright © 2020 ImKe. All rights reserved.

import numpy as np
import time
import math
import random
from numpy import linalg as la
from scipy.sparse.linalg import norm
import scipy.sparse as ss
import scipy.io
import random
import itertools
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

def OtraceEEC(M, H, p, iter_num, plot = True):
    #M m*n input data matrix, Y.*A is the observed elements in the data matrix 
    #H m*n mask matrix, A(i,j)=1 if Y(i,j) is observed, otherwise A(i,j)=0.(denoted as H in the paper)
    #p the p value of the Schatten p-Norm
    #st: regularization to avoid singularity of matrix
    #X: recovered data matrix
    #obj: objective values during iterations
    #plot: draw the evaluation plot based on obj

    m,n = M.shape
    M_1 = M#to calculate rmse 
    M = M*H
    X = M
    temp = np.dot(M.T, M)
    st = 0.002*max(abs(np.diag(temp)))
    #different from the one in the paper
    D =fractional_matrix_power((np.dot(X.T, X) + np.dot(st, np.eye(n))),(1-(p/2)))
    print(D.shape)
    rmse = []
    obj = []
    for iter in range(iter_num):
        Lambda = np.zeros(shape = (m,n))
        for i in range(m):
            Hi = H[i].T
            idx = np.argwhere(Hi == 1)
            idx = idx[0][0]
            DH1 = D[idx][idx]
            Lambda[i][idx] = np.dot(((M[i][idx]).T),1/(DH1))
        X = np.dot((Lambda*H), D)
        diff = M_1 - X
        D = fractional_matrix_power((np.dot(X.T, X) + np.dot(st, np.eye(n))),(1-(p/2)))
        obj.append(np.trace((np.real(fractional_matrix_power((np.dot(X.T, X)+np.dot(st,np.eye(n))),(p/2))))))
        print("iternum:%d  RMSE:%f" % (iter,(np.linalg.norm(diff) / np.sqrt(n*m))))
        print_("Minimazation object:%f"% np.trace((np.real(fractional_matrix_power((np.dot(X.T, X)+np.dot(st,np.eye(n))),(p/2))))))
        rmse.append(np.linalg.norm(diff) / np.sqrt(n*m))
    if plot:
        plt.figure(1)
        plt.plot(range(iter_num), obj, 'b-.')
        plt.grid()
        plt.title("Evaluation")
        plt.ylabel("Value of objective function")
        plt.xlabel("iteration")
        plt.show()
    return st, X, rmse, obj 