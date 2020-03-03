#dataloader.py
#Created by ImKe on 2020/2/28
#Copyright © 2020 ImKe. All rights reserved.

import os
import numpy as np
import scipy.sparse as ss

path_prefix = '/Users/imke/Downloads/Librec/dataset/'
def dataloader(dataset = 'ratings'):
    fname = path_prefix + dataset+'.dat'
    max_uid = 0
    max_vid = 0
    users = []
    movies = []
    ratings = []
    first_line_flag = True
    with open(fname) as f:
        for line in f:
            tks = line.strip().split('::')
            # tks = m
            if first_line_flag:
                max_uid = int(tks[0])
                max_vid = int(tks[1])
                first_line_flag = False
                continue
            max_uid = max(max_uid, int(tks[0]))
            max_vid = max(max_vid, int(tks[1]))
            users.append((int(tks[0]) - 1))
            movies.append(int(tks[1]) - 1)
            ratings.append(int(tks[2]))
    M_original = ss.csr_matrix((ratings, (users, movies)), shape=(max_uid, max_vid))
    return M_original