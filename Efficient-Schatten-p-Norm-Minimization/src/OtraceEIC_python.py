#OtraceEIC_python.py
#Created by ImKe on 2020/3/13
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

def OtraceEIC(M, H, p, r, iter_num, plot = True):
    #M m*n input data matrix, Y.*A is the observed elements in the data matrix 
    #H m*n mask matrix, A(i,j)=1 if Y(i,j) is observed, otherwise A(i,j)=0.(denoted as H in the paper)
    #p the p value of the Schatten p-Norm
    #r:parameter (\lambda in the paper)
    #st: regularization to avoid singularity of matrix
    #iter_num:iteration number
    #X: recovered data matrix
    #obj: objective values during iterations
    #plot: draw the evaluation plot
    
    m,n = M.shape
    M_1 = M
    M = M*H
    X = M
    temp = np.dot(M.T, M)
    st = 0.002*max(abs(np.diag(temp)))
    
    #svd to calculate the value of objective function
    #U,S,V = la.svd(X.T.dot(X))
    #s = np.zeros(shape = (n,1))
    #t = S
    #for i in range(len(t)):
    #    s[i][0] = t[i]
    #D = p / 2 * U.dot(np.diag(np.power(s + st, p / 2 - 1).T[0])).dot(V.T)
    D = p/2*fractional_matrix_power((np.dot(X.T, X) + np.dot(st, np.eye(n))),(p/2-1))
    rmse = []
    obj = []
    for iter in range(iter_num):
        for i in range(m):
            Hi = H[i]
            Mi = M[i]
            MH = Hi*Mi
            Hid = np.diag(Hi)
            X[i] = np.dot(la.inv(Hid + r*D), MH.T).T
        #U,S,V = la.svd(X.T.dot(X))
        #s = np.zeros(shape = (n,1))
        #t = S
        #for i in range(len(t)):
        #    s[i][0] = t[i]
        #D = p / 2 * U.dot(np.diag(np.power(s + st, p / 2 - 1).T[0])).dot(V.T)
        D = p/2*fractional_matrix_power((np.dot(X.T, X) + np.dot(st, np.eye(n))),(p/2-1))
        diff = M_1 - X
        print("iternum:%d  RMSE:%f" % (iter, (np.linalg.norm(diff) / np.sqrt(n * m))))
        rmse.append(np.linalg.norm(diff) / np.sqrt(n * m))
        obj.append(np.trace((np.real(fractional_matrix_power((np.dot(X.T, X)+np.dot(st,np.eye(n))),(p/2))))))
        print("Minimazation object:%f" %np.trace((np.real(fractional_matrix_power((np.dot(X.T, X)+np.dot(st,np.eye(n))),(p/2))))))
    if plot:
        plt.figure(1)
        plt.plot(range(iter_num), obj, 'b-.')
        plt.grid()
        plt.title("Evaluation")
        plt.ylabel("Value of objective function")
        plt.xlabel("iteration")
        plt.show()
    return st,X,rmse,obj