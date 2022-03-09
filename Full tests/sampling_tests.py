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

file_normed='normed_full_telemetry_filtered.csv'
file_telemetry='full_telemetry_filtered.csv'
root_autosave='telemetry_filtered_'
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
file_norm_train='Normed_data_initial_train_filtered.csv'#'Normed_data_initial_train_filtered.csv'#'Normed_data_train_filtered.csv'
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
    file_norm_train='Normed_data_initial_train_filtered.csv'#'Normed_data_train_filtered.csv'
    training_dates=[["2021-05-19T12:45:00.000Z","2021-05-25T00:10:00.000Z"]]#,["2021-05-27T17:45:00.000Z","2021-05-30T02:10:00.000Z"]]
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
# gaussian funct for multidim cases - can take a matrix of mus - on in each row
def gaussian(x,mu,sigmar):
    return np.exp(-np.sum(np.power(x - mu, 2.) / (2 * np.power(sigmar, 2.)),axis=1))

# the combination of many gaussians to make the weight function -> sample and stds are an array only, mus is the dataframe
def f_weight(sample,mus,stds,amplitude=1):
    total_w=0
    #♦total_w=np.array([gaussian(sample,stds,mu=mus.iloc[i]) for i in range(len(mus))])
    #times_loop=[]
    for i in range(len(mus)):
        #t_c=datetime.datetime.now()
        total_w=total_w+gaussian(sample,stds,mu=mus.iloc[i])
        #t_d=datetime.datetime.now()
        #times_loop.append(t_d-t_c)
    return total_w*amplitude#,np.mean(times_loop),np.max(times_loop)  #np.sum()

def f_weight_parallel(sample,mus,stds,weights,position): #,amplitude=1
    total_w=0
    for i in range(mus.shape[0]):
        total_w=total_w+gaussian(sample,stds,mu=mus.iloc[i].values)
    weights[position]=total_w
    print('Done Position #'+str(position))
    return True
#f_weight(training_data.iloc[20].values[1:], mes, std_train_pool)

# THE LOOP: Start with one random sample, then get random P sets to evaluate and select ONLY 1, eventually all are too similar
normed_train_data_o=pd.read_csv(file_norm_train,index_col=0).sample(150000)
#filter to only use contour conditions:
cont_cond=['ControlRegCompAC.VarFrequencyHzMSK','WaterFlowMeter','W_OutTempUser'] #,'W_InTempUser','ExtTemp'
if cont_cond!=[]:
    training_set=copy.deepcopy(normed_train_data_o[cont_cond])
else:
    training_set=copy.deepcopy(normed_train_data_o)
std_train_pool=training_set.describe().transpose().loc[:,'std'].values
run_num=5
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
stds=std_train_pool/gamma
#vec_f_weight=np.vectorize(f_weight, excluded=['mus','stds','amplitude'])
ratios=[]
std_rat=[]
for ratio_min in ratio_min_set:
    for ratio_std in ratio_std_set:
        ratio_file=ratio_file_root+str(int(10*ratio_min))+'_rs'+str(int(10*ratio_std))+'.csv'
        selected_samples_file=selected_samples_file_root+str(int(10*ratio_min))+'_rs'+str(int(10*ratio_std))+'.csv'
        file_printed = open(print_file, "a")
        print('**********************************************************',file=file_printed)
        print('******** Test with ratio mean {} and ratio std {} ********'.format(ratio_min,ratio_std),file=file_printed)
        file_printed.close()
        N=100
        turn=0
        new_set_stats=[]
        stop_augur=0
        searching=True
        index_record=[]
        training_set=copy.deepcopy(normed_train_data_o[cont_cond])
        selected=training_set.sample(n=n_0)
        index_record=list(selected.index)
        training_set=training_set.drop(selected.index)
        ratios=[]
        std_rat=[]
        min_tol=0.001 # if the ratio is smaller than min_tol we just keep going
        while searching:
            t_a=datetime.datetime.now()
            turn=turn+1
            new_batch=training_set.sample(n=P)
            best_w=1000000
            record_p=[]
            for i in range(P):
                weis=gaussian(new_batch.iloc[i].values, selected.values, stds)
                wei=np.sum(weis)
                record_p.append(wei)
                if wei<best_w:
                    best=i
                    best_w=wei
        
            #best_record.append(best_w)
            selected=selected.append(new_batch.iloc[best], ignore_index=True)
            index_record.append(new_batch.iloc[best].name)
            training_set=training_set.drop(new_batch.iloc[best].name)
            t_b=datetime.datetime.now()
            dif=t_b-t_a
            ratios.append(best_w/np.mean(record_p))
            std_rat.append(np.std(record_p)/np.mean(record_p))
            if turn%N==0:
                mov_avg=np.convolve(ratios, 2*np.ones(int(N/2))/N, mode='valid')
                stdr_avg=np.convolve(std_rat, 2*np.ones(int(N/2))/N, mode='valid')
                t_b=datetime.datetime.now()
                dif=t_b-t_a
                #time_costs.append(dif)
                file_printed = open(print_file, "a")
                print('--> Turn '+str(turn),file=file_printed)
                print('   - Time: {}'.format(dif),file=file_printed)
                print('   - Distance From Best (MAV): {}'.format(abs(np.array(ratios).argmax()-len(ratios))),file=file_printed)
                print('   - Ratio (STD): {}'.format(stdr_avg[-1]),file=file_printed)
                print('   - Ratio (MAV): {}'.format(mov_avg[-1]),file=file_printed)
                print('   - Ratio: {}'.format(best_w/np.mean(record_p)),file=file_printed)
                print('   - Best Cost: {}'.format(best_w),file=file_printed)
                print('   - STD P set: {}'.format(np.std(record_p)),file=file_printed)
                print('   - Mean P set: {}'.format(np.mean(record_p)),file=file_printed)
                file_printed.close()
                selected.to_csv(sample_file)
                normed_train_data_o.loc[index_record].to_csv(selected_samples_file)
                #searching=False
                if (mov_avg[-1]>ratio_min and stdr_avg[-1]<ratio_std) or abs(np.array(ratios).argmax()-len(ratios))>3*N:     
                    #selected.to_csv('selection_Training_Data.csv')
                    stop_augur=stop_augur+1
                    file_printed = open(print_file, "a")
                    print(' [!] FOUND AUGUR TO STOP #'+str(stop_augur),file=file_printed)
                    print('     --> Ratio from best Ratio (MAV): '+str((max(mov_avg)-mov_avg[-1])/max(mov_avg)),file=file_printed)
                    print('     --> STDs: '+str(stds[0]),file=file_printed)
                    file_printed.close()
                    #if stop_augur>stop_needed:
                    new_batch=training_set
                    best_w=100000
                    #record_p=np.array([f_weight(new_batch.iloc[i].values, selected, stds) for i in range(P)])
                    #best_w=record_p.min()        
                    #best=np.where(record_p==best_w)[0][0]
                    record_p=[]
                    for i in range(new_batch.shape[0]):
                        #t_c=datetime.datetime.now()
                        weis=gaussian(new_batch.iloc[i].values, selected.values, stds)
                        wei=np.sum(weis)
                        record_p.append(wei)
                        #if wei<best_w:
                            #best=i
                            #best_w=wei
                    smallest=np.array(record_p).argsort()[:5]
                    #best_record.append(best_w)
                    selected=selected.append(new_batch.iloc[smallest], ignore_index=True)
                    index_record=index_record+list(new_batch.iloc[smallest].index)
                    training_set=training_set.drop(new_batch.iloc[smallest].index)
                    t_b=datetime.datetime.now()
                    dif=t_b-t_a
                    new_batch=training_set
                    best_w=100000
                    #record_p=np.array([f_weight(new_batch.iloc[i].values, selected, stds) for i in range(P)])
                    #best_w=record_p.min()        
                    #best=np.where(record_p==best_w)[0][0]
                    for i in range(new_batch.shape[0]):
                        #t_c=datetime.datetime.now()
                        weis=gaussian(new_batch.iloc[i].values, selected.values, stds)
                        wei=np.sum(weis)
                        record_p.append(wei)
                        if wei<best_w:
                            best=i
                            best_w=wei
                    #time_costs.append(dif)
                    extended=best_w/np.mean(record_p)
                    ext_std=np.std(record_p)/np.mean(record_p)
                    file_printed = open(print_file, "a")
                    print(' [S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S]',file=file_printed)
                    print(' [S]  - Time: {}'.format(dif),file=file_printed)
                    #print(' [S]  - Ratio (MAV): {}'.format((max(mov_avg)-extended)/max(mov_avg)),file=file_printed)
                    print(' [S]  - Ratio min/mean: {}'.format(extended),file=file_printed)
                    print(' [S]  - Ratio std/mean: {}'.format(ext_std),file=file_printed)
                    print(' [S]  - Best Cost: {}'.format(best_w),file=file_printed)
                    print(' [S]  - STD P set: {}'.format(np.std(record_p)),file=file_printed)
                    print(' [S]  - Mean P set: {}'.format(np.mean(record_p)),file=file_printed)
                    print(' [S]  - Max P set: {}'.format(np.max(record_p)),file=file_printed)
                    print(' [S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S][S]',file=file_printed)
                    file_printed.close()
                    
                    normed_train_data_o.loc[index_record].to_csv(selected_samples_file)
                    # here the ratio req is relaxed ... the whole set can have a bigger deviation from the best example (Only with P samples)
                    if (extended>softener*ratio_min and softener*ext_std<ratio_std) or selected.shape[0]>training_set.shape[0]/10:
                        file_printed = open(print_file, "a")
                        
                        print(' ----------- THE END -------------',file=file_printed)
                        file_printed.close()
                        searching=False
                        selected.to_csv(sample_file)
                        df_r=pd.DataFrame({'ratios':ratios})
                        df_r.to_csv(ratio_file)
                    else:
                        # the bigger std the worse isolated samples will be -> we will start with the smaller value
                        #stds=std_train_pool/((1+stop_augur*0.5)*gamma)
                        stds=std_train_pool*(1+stop_augur*0.25)/(gamma)
                        file_printed = open(print_file, "a")
                        print(' [!] Go on with higher sensibility',file=file_printed)
                        print('     --> New STDs: '+str(stds[0]),file=file_printed)
                        file_printed.close()
                        #◘ratios=[]
                    






