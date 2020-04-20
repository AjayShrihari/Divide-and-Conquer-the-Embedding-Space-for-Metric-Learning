# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:07:57 2020

@author: aniket
"""
import pandas as pd
import numpy as np
file = 'slurm-127563.out'
job_num = file.split('-')[1].rstrip('.out')
initial_epochs = 20
num_learners = 8
k_val = [1,10,100,1000]
L_full = []
L_k = []
test = []
cnt_k = 0
with open(file) as f:
    for line in f:
        elem = line.split(' ')
        if(elem[0] == 'Epoch'):
            
            if(elem[1] == '(Train)'):
                if(cnt_k == 0):
                    L_temp = {}
                if(elem[4] == '-1:'):
                    L_full.append({'Loss':float(elem[7].strip('[]\n'))})
                else:
                    L_temp['Learner_' + str(cnt_k)] = float(elem[7].strip('[]\n'))
                    cnt_k = (cnt_k + 1)%num_learners
                    if(cnt_k == 0):
                        L_k.append(L_temp)
            
            elif(elem[1] == '(Test)'):
                test_epoch = {}
                test_epoch['NMI'] = float(elem[4].strip('[]'))
                for k,i in zip(k_val,np.arange(8,15,2)):
                    test_epoch['R_' + str(k)] = float(elem[i].strip(',]\n'))
                test.append(test_epoch)
                    
L_full = pd.DataFrame(L_full)
L_full.to_csv(job_num + '_L_full.csv')
L_k = pd.DataFrame(L_k)
L_k.to_csv(job_num + '_L_k.csv')
test = pd.DataFrame(test)
test.to_csv(job_num + '_test.csv')            
            
