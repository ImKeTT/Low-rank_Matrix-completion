#main.py
#Created by ImKe on 2020/3/5
#Copyright © 2020 ImKe. All rights reserved.

from __future__ import division
import random
import numpy as np
import time
import math
import itertools
import scipy.sparse as ss
from dataloader import dataloader
from gen_matrix import gen_matrix
from OtraceEEC_python import *
from OtraceEIC_python import *

#parameters
NO = 1#which algorithm 1 or 2
load_data = False#load movilens data or genarate a random matrix
plot = True#draw the plot based on obj
iter_num = 1000
p = 0.1#the p value of the Schatten p-Norm
r = 3# parameter in OtraceEIC
m = 150
n = 300
k = 10#rank of generated matrix

if __name__ == '__main__':
    if load_data:
        M = dataloader()
        proportion = 1.0
        #M = M[:100]
        m,n = M.shape
        total_num = len(M.nonzero()[0])
        idx = random.sample(range(total_num),int(total_num*proportion))
        Omega = (M.nonzero()[0][idx],M.nonzero()[1][idx])
        H = np.ones(shape = (m,n))
        data_H = H[Omega]
        H_1 = ss.csr_matrix((data_H, Omega),shape = (m,n)).A#to ndarray
        M = M.A#to_ndarray
        if (NO == 1):
            st, X,rmse, obj = OtraceEEC(M, H_1, p, iter_num, plot)
        elif (NO == 2):
            st, X, rmse, obj = OtraceEIC(M, H_1, p, r, iter_num, plot)
    else:
        M, H_1 = gen_matrix(m,n,k)
        if (NO == 1):
            st, X, rmse, obj = OtraceEEC(M, H_1, p, iter_num, plot)
        elif (NO == 2):
            st, X, rmse, obj = OtraceEIC(M, H_1, p, r, iter_num, plot)