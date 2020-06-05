#gen_matrix.py
#Created by ImKe on 2020/3/1
#Copyright © 2020 ImKe. All rights reserved.

import numpy as np
import random
import scipy.sparse as ss

#generate a random matrix with shape n1*n2 and rank r to evaluate the algorithm 
def gen_matrix(n1, n2, r):
    np.random.seed(999)
    H = np.ones((n1,n2))
    M = np.random.random((n1,r)).dot(np.random.random((r,n2)))
    df = r*(n1+n2-r);
    m = min(1*df,round(.99*n1*n2)); 
    ind = random.sample(range(n1*n2),m)
    #set sample space Omega
    Omega = np.unravel_index(ind, (n1,n2))

    data_H = H[Omega]
    #to ndarray type
    H_1 = ss.csr_matrix((data_H,Omega),shape = (n1,n2)).A
    
    data_M = M[Omega]
    #M_1 = M*H_1(Hadamard product)
    M_1 = ss.csr_matrix((data_M, Omega),shape = (n1,n2)).A
    return M, H_1