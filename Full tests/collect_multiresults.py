# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:14:18 2022

@author: sega01
"""
import numpy as np
import pandas as pd
import pickle
#import pickle5 as pickle
import copy
from matplotlib import cm as CM
from matplotlib.lines import Line2D
from elasticsearch import Elasticsearch
import pickle
from cvxopt import solvers, matrix, spdiag, log, exp, div
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pickle
import datetime
import math
from sklearn.linear_model import LinearRegression
import traceback
import copy
import os
import json
from matplotlib.colors import LogNorm
import multiprocessing

"""normed_train_data_o=pd.read_csv(file_norm_train,index_col=0).sample(150000)
#filter to only use contour conditions:
cont_cond=['ControlRegCompAC.VarFrequencyHzMSK','WaterFlowMeter','W_OutTempUser'] #,'W_InTempUser','ExtTemp'
if cont_cond!=[]:
    training_set=copy.deepcopy(normed_train_data_o[cont_cond])
else:
    training_set=copy.deepcopy(normed_train_data_o)
std_train_pool=training_set.describe().transpose().loc[:,'std'].values"""
run_num=4
print_file='sampling_multitest{}_220222.txt'.format(run_num)#'sampling_test_CC_vhard_270122.txt', 'sampling_test_v1_260122.txt','sampling_test_CC_260122.txt'
sample_file='selection_multitest{}_220222.csv'.format(run_num)#'selection_Training_Data_CC_vhard_270122.csv','selection_Training_Data_260122.csv','selection_Training_Data_CC_260122.csv'
ratio_file_root='ratios_multitest{}_rm'.format(run_num)
ratio_min_set=[0.1,0.2,0.3,0.4,0.5]
ratio_std_set=[0.2,0.3,0.4]
selected_samples_file_root='subsampled_GaussIter{}_rm'.format(run_num) #'subsampled_GaussIter_initialTrDt.csv','subsampled_GaussIter_FullTrDt.csv'
n_0=100
P=1000
gamma=30
stop_needed=10
ratio_min=0.4 #epsilon
ratio_std=0.3 #omega
softener=0.75

searching=True
turn=0
#time_costs=[]
#best_record=[]
new_set_stats=[]
stop_augur=0
num_workers = multiprocessing.cpu_count()
#stds=std_train_pool/gamma
#vec_f_weight=np.vectorize(f_weight, excluded=['mus','stds','amplitude'])
ratios=[]
std_rat=[]

dic_r={0.1:[],0.2:[],0.3:[],0.4:[],0.5:[]}
dic_mean={0.1:[],0.2:[],0.3:[],0.4:[],0.5:[]}
dic_std={0.1:[],0.2:[],0.3:[],0.4:[],0.5:[]}
ratio_std_dic={0.2:[],0.3:[],0.4:[]}
for i in dic_r:
    dic_r[i]={0.2:[],0.3:[],0.4:[]}
    dic_mean[i]={0.2:[],0.3:[],0.4:[]}
    dic_std[i]={0.2:[],0.3:[],0.4:[]}
numbers=[1,2,3,4,5]
for run_num in numbers:
    print_file='sampling_multitest{}_220222.txt'.format(run_num)#'sampling_test_CC_vhard_270122.txt', 'sampling_test_v1_260122.txt','sampling_test_CC_260122.txt'
    sample_file='selection_multitest{}_220222.csv'.format(run_num)#'selection_Training_Data_CC_vhard_270122.csv','selection_Training_Data_260122.csv','selection_Training_Data_CC_260122.csv'
    ratio_file_root='ratios_multitest{}_rm'.format(run_num)
    selected_samples_file_root='subsampled_GaussIter{}_rm'.format(run_num) 
    for ratio_min in ratio_min_set:
        for ratio_std in ratio_std_set:
            ratio_file=ratio_file_root+str(int(10*ratio_min))+'_rs'+str(int(10*ratio_std))+'.csv'
            selected_samples_file=selected_samples_file_root+str(int(10*ratio_min))+'_rs'+str(int(10*ratio_std))+'.csv'
            print('   -->  rm {} | rs {}'.format(ratio_min,ratio_std))
            
            df=pd.read_csv(selected_samples_file,index_col=0)
            dic_r[ratio_min][ratio_std].append(df.shape[0])
            if ratio_std==ratio_std_set[-1]:
                print('     * The rs is {}, and now the dic r corresponding:'.format(ratio_std))
                print(dic_r[ratio_min][ratio_std])
            print('     ')
                

for ratio_min in ratio_min_set:
    for ratio_std in ratio_std_set:
        dic_mean[ratio_min][ratio_std]=np.mean(dic_r[ratio_min][ratio_std])
        dic_std[ratio_min][ratio_std]=np.std(dic_r[ratio_min][ratio_std])
result=pd.DataFrame(dic_mean)
std_res=pd.DataFrame(dic_std)