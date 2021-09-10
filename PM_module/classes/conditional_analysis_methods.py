# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:56:09 2021

@author: sega01
"""
import numpy as np
import pandas as pd

def get_joint_tables(var_names,telemetry,bins,activations,confidences,labels,mso_set):
    joint_activ={}
    for var in var_names:
        to_pd={var:pd.cut(telemetry[var].values[:,0],bins[var],labels=labels)}
        for i in range(len(mso_set)):
            name='MSO_'+str(i)
            to_pd[name]=activations[i]
        joint_activ[var]=pd.DataFrame(to_pd)
      
    # get the conditional probabilities for activations only (each var and each MSO)
    N_appear=joint_activ[var_names[0]][var_names[0]].shape[0]
    joint_results_activ={}
    for var in var_names:
        joint_results_activ[var]={}
        N_appear=joint_activ[var].shape[0]
        for i in range(len(mso_set)):
            name='MSO_'+str(i)
            joint_results_activ[var][name]={}
            subset=joint_activ[var].loc[joint_activ[var][name]==1]
            joint_results_activ[var][name]['Total_MSO_activ']=subset.shape[0]
            a = subset[var].unique()
            for j in a:
                if j==0:
                    interval='-Inf to '+str(np.round(bins[var][j+1],decimals=2))
                elif j==len(labels):
                    interval=str(np.round(bins[var][j],decimals=2))+' to +Inf'
                else:
                    interval=str(np.round(bins[var][j],decimals=2))+' to '+str(np.round(bins[var][j+1],decimals=2))
                joint_results_activ[var][name][interval]={}
                joint_results_activ[var][name][interval]['Legend_index']=j
                joint_results_activ[var][name][interval]['Activations_%']=subset.loc[joint_activ[var][var]==j].shape[0]*100/joint_results_activ[var][name]['Total_MSO_activ']
                joint_results_activ[var][name][interval]['P_joint']=subset.loc[joint_activ[var][var]==j].shape[0]/N_appear
                joint_results_activ[var][name][interval]['P_var']=joint_activ[var].loc[joint_activ[var][var]==j].shape[0]/N_appear
                joint_results_activ[var][name][interval]['P_cond']=joint_results_activ[var][name][interval]['P_joint']/joint_results_activ[var][name][interval]['P_var']
    
    # get the conditional probabilities for confidences only (each var and each MSO)  
    #create the joint tables
    joint={}
    for var in var_names:
        to_pd={var:pd.cut(telemetry[var].values[:,0],bins[var],labels=labels)}
        for i in range(len(mso_set)):
            name='MSO_'+str(i)
            to_pd[name]=confidences[i]
        joint[var]=pd.DataFrame(to_pd)
        
    # get the conditional probabilities for activations only (each var and each MSO)
    N_appear=joint[var_names[0]][var_names[0]].shape[0]
    joint_results={}
    for var in var_names:
        joint_results[var]={}
        N_appear=joint[var].shape[0]
        for i in range(len(mso_set)):
            name='MSO_'+str(i)
            joint_results[var][name]={}
            a = joint[var][var].unique()
            for j in a:
                if j==0:
                    interval='-Inf to '+str(np.round(bins[var][j+1],decimals=2))
                elif j==len(labels):
                    interval=str(np.round(bins[var][j],decimals=2))+' to +Inf'
                else:
                    interval=str(np.round(bins[var][j],decimals=2))+' to '+str(np.round(bins[var][j+1],decimals=2))
                joint_results[var][name][interval]={}
                joint_results[var][name][interval]['Legend_index']=j
                
                subset=joint[var].loc[joint[var][var]==j]
                joint_results[var][name][interval]['C_mean']=subset[name].describe()['mean']
                joint_results[var][name][interval]['P_var']=joint[var].loc[joint[var][var]==j].shape[0]/N_appear
                joint_results[var][name][interval]['C_std']=subset[name].describe()['std']
                
    return joint_results_activ,joint_results
    
def get_cond_activ_mtrs(joint_results,mso_set,labels,var_names,bins):
    mtr_condactiv={}
    mtr_perc_activ={}
    for n in range(len(mso_set)):       
        name='MSO_'+str(n)
        matr=np.zeros([len(var_names),len(labels)])
        i=-1
        for var in var_names:
            i=i+1
            for j in labels:
                if j==0:
                    interval='-Inf to '+str(np.round(bins[var][j+1],decimals=2))
                elif j==len(labels):
                    interval=str(np.round(bins[var][j],decimals=2))+' to +Inf'
                else:
                    interval=str(np.round(bins[var][j],decimals=2))+' to '+str(np.round(bins[var][j+1],decimals=2))
                if interval in joint_results[var][name]:
                    matr[i,j]=np.round(joint_results[var][name][interval]['P_cond']*100,decimals=3)
                    mtr_perc_activ[name][i,j]=joint_results[var][name][interval]['Activations_%']
                else:
                    matr[i,j]=0
                    mtr_perc_activ[name][i,j]=0
        mtr_condactiv[name]=matr
    return mtr_condactiv,mtr_perc_activ

def get_mean_std_mtrs(joint_results,mso_set,labels,var_names,bins):
    mtr_mean_set={}
    mtr_std_set={}
    for n in range(len(mso_set)):       
        name='MSO_'+str(n)
        matr_mean=np.zeros([len(var_names),len(labels)])
        matr_std=np.zeros([len(var_names),len(labels)])
        i=-1
        for var in var_names:
            i=i+1
            for j in labels:
                if j==0:
                    interval='-Inf to '+str(np.round(bins[var][j+1],decimals=2))
                elif j==len(labels):
                    interval=str(np.round(bins[var][j],decimals=2))+' to +Inf'
                else:
                    interval=str(np.round(bins[var][j],decimals=2))+' to '+str(np.round(bins[var][j+1],decimals=2))
                if interval in joint_results[var][name]:
                    matr_mean[i,j]=np.round(joint_results[var][name][interval]['C_mean'],decimals=3)
                    matr_std[i,j]=np.round(joint_results[var][name][interval]['C_std'],decimals=3)
                else:
                    matr_mean[i,j]=0
                    matr_std[i,j]=0
        mtr_mean_set[name]=matr_mean
        mtr_std_set[name]=matr_std
    
    return mtr_mean_set,mtr_std_set

#get the best value for a moving average of the size of 20% of the cells >0 in the prob matr
def moving_average_variables(matr,matr_prbs,var_names,wind_ratio=0.20):
    key_intervals={}
    best_array=[]
    for i in range(matr.shape[0]):
        best=-1
        pos=0
        wind=int(np.where(matr_prbs[i]>0)[0].shape[0]*wind_ratio)
        if wind%2==0:
            wind=wind+1
        half=int((wind-1)/2)
        if half>=1:
            for j in range(half,matr.shape[1]-half):
                mean=sum(matr[i,j-half:j+half+1])/wind
                if mean>best:
                    best=mean
                    pos=j
            best_array.append(best)
            key_intervals[var_names[i]]={'score':best,'window':wind,'position':pos}
        else:
            key_intervals[var_names[i]]={'score':'too narrow to evaluate'}
    return key_intervals,best_array