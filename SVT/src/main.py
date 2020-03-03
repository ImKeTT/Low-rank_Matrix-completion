#main.py
#Created by ImKe on 2020/2/28
#Copyright © 2020 ImKe. All rights reserved.
import time
import matplotlib.pyplot as plt
from SVT import *
from dataloader import *

#parameters
num_of_entries = 1000
iter_list = [50,100, 200, 300, 500, 700, 900]
plot = True
if __name__ == '__main__':
    start = time.clock()
    M_original = dataloader('ratings')
    M1 = M_original[:num_of_entries]
    rmse1 = []
    s = 1
    for i in iter_list:
        rmse1.append(SVT(M1, i)[-1])
        print("The %d iteration is completed"%s)
        s += 1
    if plot:
        plt.plot(iter_list, rmse1, '*-')
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        plt.show()
    print("Time:%f s" %(time.clock() - start))