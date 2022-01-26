# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 09:18:32 2022

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
import multiprocessing 

file_normed='normed_full_telemetry.csv'
file_telemetry='full_telemetry.csv'
root_autosave='telemetry_'
sensors={"1":"ControlRegCompAC.VarFrequencyHzMSK","2":"EbmpapstFan_1_Mng.InfoSpeed_EBM_1.CurrSpeed","3":"Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val","7":"WaterFlowMeter","8":"SuctSH_Circ1","9":"DscgTempCirc1","12":"EvapTempCirc1","13":"CondTempCirc1","16":"W_OutTempUser","17":"W_OutTempEvap","18":"W_InTempUser","20":"FiltPress","23":"PumpPress","24":"ExtTemp"}
var_names=['ControlRegCompAC.VarFrequencyHzMSK','EbmpapstFan_1_Mng.InfoSpeed_EBM_1.CurrSpeed',
             'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val','WaterFlowMeter',
             'SuctSH_Circ1','DscgTempCirc1','EvapTempCirc1',
             'CondTempCirc1','W_OutTempUser','W_OutTempEvap',
             'W_InTempUser','FiltPress','PumpPress','ExtTemp']
dates_goal = [["2021-05-19T15:45:00.000Z","2021-05-25T02:10:00.000Z"],["2021-05-25T17:00:00.000Z","2021-05-30T07:10:00.000Z"],["2021-06-01T00:45:00.000Z","2021-06-07T12:10:00.000Z"],["2021-06-23T17:00:00.000Z","2021-06-30T23:00:00.000Z"],["2021-07-01T01:00:00.000Z","2021-07-10T01:00:00.000Z"],["2021-07-10T01:00:00.000Z","2021-07-21T10:00:00.000Z"]]#["2021-05-19T16:30:00.000Z","2021-05-26T08:30:00.000Z"]
names_analysis = ['models_error', 'low_bounds', 'high_bounds','activations', 'confidence', 'group_prob', 'timestamp']
files_txt=True
not_prepared=False
file_norm_train='Normed_data_train.csv'
if not_prepared:
    if files_txt:
        telemetry={'timestamp':[]}
        for var in var_names:
            telemetry[var]=[]
        for d in dates_goal:
            autosave=root_autosave+d[0][:10]+'.txt'
            try:
                with open(autosave, 'r') as handle:
                    tel=json.load(handle)
                    for k in (tel.keys()):
                        telemetry[k]=telemetry[k]+tel[k]
            except:
                print('Missing File: '+autosave)
             
    for t in telemetry:
        print(len(telemetry[t]))
    df=pd.DataFrame(telemetry)
    
    def norm(x,train_stats):
        y={}
        for name in x.columns:
            y[name]=[]
            for i in x.index:
                a=float((x.loc[i,name]-train_stats.loc[name,'mean'])/train_stats.loc[name,'std'])
                y[name].append(a)  #.apply(lambda num: (num-train_stats.loc[name,'mean'])/train_stats.loc[name,'std'])
        return pd.DataFrame(y)
    
    # selection of the training range
    file_norm_train='Normed_data_train.csv'
    training_dates=[["2021-05-19T12:45:00.000Z","2021-05-25T00:10:00.000Z"],["2021-05-27T17:45:00.000Z","2021-05-30T02:10:00.000Z"]]
    keep=df['timestamp']<= '2001-02-24T00:00:10.101Z' # RIP Claude Shannon
    for d in training_dates:
        lowerDate=d[0]
        upperDate=d[1]
        keep=keep|((df['timestamp']>= lowerDate)&(df['timestamp']<= upperDate))
    training_data=df.iloc[keep.values]
    train_stats=training_data.describe()
    train_stats = train_stats.transpose()
    normed_train_data_o = norm(training_data.drop('timestamp',axis=1),train_stats)
    normed_train_data_o.to_csv(file_norm_train)
# define functions needed 
# gaussian funct for multidim cases
def gaussian(x,sigmar,mu):
    return np.exp(-np.sum(np.power(x - mu, 2.) / (2 * np.power(sigmar, 2.))))

# the combination of many gaussians to make the weight function -> sample and stds are an array only, mus is the dataframe
def f_weight(sample,mus,stds,amplitude=1):
    total_w=0
    total_w=np.array([gaussian(sample,stds,mu=mus.iloc[i]) for i in range(len(mus))])
    """for i in range(len(mus)):
        total_w=total_w+gaussian(sample,stds,mu=mus.iloc[i])"""
    return np.sum(total_w)*amplitude

def f_weight_parallel(sample,mus,stds,weights,position): #,amplitude=1
    total_w=0
    for i in range(mus.shape[0]):
        total_w=total_w+gaussian(sample,stds,mu=mus.iloc[i].values)
    weights[position]=total_w
    print('Done Position #'+str(position))
    return True
#f_weight(training_data.iloc[20].values[1:], mes, std_train_pool)

# THE LOOP: Start with one random sample, then get random P sets to evaluate and select ONLY 1, eventually all are too similar
normed_train_data_o=pd.read_csv(file_norm_train,index_col=0)
#filter to only use contour conditions:
cont_cond=['ControlRegCompAC.VarFrequencyHzMSK','WaterFlowMeter','W_OutTempUser','W_InTempUser','ExtTemp']
if cont_cond!=[]:
    training_set=copy.deepcopy(normed_train_data_o[cont_cond])
else:
    training_set=copy.deepcopy(normed_train_data_o)
std_train_pool=training_set.describe().transpose().loc[:,'std'].values
print_file='sampling_test_CCharsh_260122.txt'# 'sampling_test_v1_260122.txt','sampling_test_CC_260122.txt'
sample_file='selection_Training_Data_CCharsh_260122.csv'#'selection_Training_Data_260122.csv','selection_Training_Data_CC_260122.csv'
n_0=20
P=200
gamma=5
stop_needed=3
ratio_req=0.333
try:
    selected=pd.read_csv(sample_file,index_col=0)
except:
    selected=training_set.sample(n=n_0)
training_set=training_set.drop(selected.index)
searching=True
turn=-1
#time_costs=[]
#best_record=[]
new_set_stats=[]
stop_augur=0
num_workers = multiprocessing.cpu_count()
stds=std_train_pool/gamma
#vec_f_weight=np.vectorize(f_weight, excluded=['mus','stds','amplitude'])
while searching:
    t_a=datetime.datetime.now()
    turn=turn+1
    new_batch=training_set.sample(n=P)
    best_w=100
    record_p=[]
    for i in range(P):
        wei=f_weight(new_batch.iloc[i].values, selected, stds)
        record_p.append(wei)
        if wei<best_w:
            best=i
            best_w=wei
    """record_p=np.array([f_weight(new_batch.iloc[i].values, selected, stds) for i in range(P)])
    best_w=record_p.min()        
    best=np.where(record_p==best_w)[0][0]"""
    #best_record.append(best_w)
    selected=selected.append(new_batch.iloc[best], ignore_index=True)
    training_set=training_set.drop(new_batch.iloc[best].name)
    if turn%20==0:
        t_b=datetime.datetime.now()
        dif=t_b-t_a
        #time_costs.append(dif)
        file_printed = open(print_file, "a")
        print('--> Turn '+str(turn),file=file_printed)
        print('   - Time: {}'.format(dif),file=file_printed)
        print('   - Ratio: {}'.format(best_w/np.mean(record_p)),file=file_printed)
        print('   - Best Cost: {}'.format(best_w),file=file_printed)
        print('   - STD P set: {}'.format(np.std(record_p)),file=file_printed)
        print('   - Mean P set: {}'.format(np.mean(record_p)),file=file_printed)
        print('   - Max P set: {}'.format(np.max(record_p)),file=file_printed)
        file_printed.close()
        selected.to_csv(sample_file)
    if best_w/np.mean(record_p)>ratio_req:
        #selected.to_csv('selection_Training_Data.csv')
        stop_augur=stop_augur+1
        file_printed = open(print_file, "a")
        print(' FOUND AUGUR TO STOP #'+str(stop_augur),file=file_printed)
        file_printed.close()
        if stop_augur>stop_needed:
            new_batch=training_set.sample(n=100*P)
            best_w=100
            record_p=np.array([f_weight(new_batch.iloc[i].values, selected, stds) for i in range(P)])
            best_w=record_p.min()        
            best=np.where(record_p==best_w)[0][0]
            #best_record.append(best_w)
            selected=selected.append(new_batch.iloc[best], ignore_index=True)
            training_set=training_set.drop(new_batch.iloc[best].name)
            t_b=datetime.datetime.now()
            dif=t_b-t_a
            #time_costs.append(dif)
            file_printed = open(print_file, "a")
            print(' [S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S]',file=file_printed)
            print(' [S]  - Time: {}'.format(dif),file=file_printed)
            print(' [S]  - Ratio: {}'.format(best_w/np.mean(record_p)),file=file_printed)
            print(' [S]  - Best Cost: {}'.format(best_w),file=file_printed)
            print(' [S]  - STD P set: {}'.format(np.std(record_p)),file=file_printed)
            print(' [S]  - Mean P set: {}'.format(np.mean(record_p)),file=file_printed)
            print(' [S]  - Max P set: {}'.format(np.max(record_p)),file=file_printed)
            print(' [S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S]',file=file_printed)
            file_printed.close()
            if best_w/np.mean(record_p)>ratio_req*0.5 or selected.shape[0]>training_set.shape[0]/10:
                file_printed = open(print_file, "a")
                print(' ----------- THE END -------------',file=file_printed)
                file_printed.close()
                searching=False
                selected.to_csv(sample_file)
        
        





