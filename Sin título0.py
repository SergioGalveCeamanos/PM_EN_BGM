# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 07:34:29 2021

@author: sega01
"""
# we want to load a model and combine the training/kde/validation data with a new batch of data that was detected as deviant but healthy
import os, requests, uuid, json, pickle
from os import path
import numpy as np
import pandas as pd
from classes.fault_detector_class_ES import fault_detector
from classes.MSO_selector_GA import find_set_v2
from classes.test_cross_var_exam import launch_analysis
from classes.pm_manager import homo_sampling
import datetime 
import traceback
import copy
import multiprocessing
def load_model(filename,model_dir):
    fault_manager = fault_detector(filename=[],mso_txt=[],host=[],machine=[],matrix=[],sensors=[],faults=[],sensors_lookup=[],sensor_eqs=[],preferent=[],filt_value=9,filt_parameter=[],filt_delay_cap=[],main_ca=[],max_ca_jump=[],cont_cond=[])
    fault_manager.Load(model_dir,filename)
    return fault_manager

def file_location(device,version="",root_path='/models/model_'):
    device=int(device)
    filename = root_path+str(device)+version+'/FM_'+str(device)+'.pkl'
    model_dir = root_path+str(device)+version+'/'
    return filename, model_dir

device=74124
version='_test_VI_StabilityFilt_090621'#_test_VI_StabilityFilt_090621,_test_V_StabilityFilt_040621,_test_IV_StabilityFilt_040621,#_test_I_260521,_test_II_StabilityFilt_260521,_test_XII_NewSelec_120421'#"_test_XI_NewSelec_080421"#"_test_X_NewSelec_070421"
time_start="2021-05-25T12:30:00.000Z"
time_stop="2021-05-26T09:30:00.000Z"
if True:
    aggSeconds=1
    file, folder = file_location(device,version)
    fm=load_model(file, folder)
    names,times_b=fm.get_data_names(option='CarolusRex',times=[[time_start,time_stop]])
    names.append(fm.target_var)
    ######## NOT LIKE THIS --> MUST BE CHANGED #############
    if device==71471 or device==74124:
        aggSeconds=1

if True:
    base_time=30 #minutes per request
    if len(times_b)==1:
        start=times_b[0][0]
        end=times_b[0][1]
        new_set=[]
        go_on=True
        from_t=start
        until_t=''
        i=0
        while go_on:
            i=i+1
            next_t=datetime.datetime(year=int(start[:4]), month=int(start[5:7]), day=int(start[8:10]), hour=int(start[11:13]),  minute=int(start[14:16]), second=0, microsecond=1000)+datetime.timedelta(minutes=30*i)
            next_t=next_t.isoformat()
            until_t=next_t[:(len(next_t)-3)]+'Z'
            new_set.append([from_t,until_t])
            from_t=until_t
            if until_t>=end:
                go_on=False
        time_set=new_set
    else:
        time_set=times_b
        
if True:
    shared_list = []
    #jobs = []
    for time in time_set:
        #p = multiprocessing.Process(target=collect_timeband, args=(time,machine,names,aggSeconds,shared_list))
        #p.start()
        #jobs.append(p)
    #for proc in jobs:
        #proc.join()
        body={'device':device,'names':names,'times':[time],'aggSeconds':aggSeconds}
        r = requests.post('http://db_manager:5001/collect-data',json = body)
        dd=r.json()
        shared_list.append(pd.DataFrame(dd))
    first=True
    for df in shared_list:
        if first:
            data=df
            first=False
        else:
            data=data.append(df,ignore_index=True)  
            
sam=int(fm.training_data.shape[0]/10)
if True:
    keep_1=data['EvapTempCirc1']<100
    filtered=copy.deepcopy(data[keep_1])
    filtered.loc[:,'UnitStatus']=filtered.loc[:,'UnitStatus']*10      
    filt_data=fm.filter_samples(filtered)
    target=filt_data.pop(fm.target_var)
    filt_data=fm.filter_stability(filt_data,target)
    new_training_data=homo_sampling(filt_data,fm.cont_cond,s=sam)
    common = filt_data.merge(new_training_data, on=["timestamp"])
    rest=filt_data[~filt_data.timestamp.isin(common.timestamp)]
    new_test_data=rest.sample(n=int(sam/2))
    new_kde_data=homo_sampling(filt_data,fm.cont_cond,s=int(sam/2))

mso=86
t=1
kde_dt=new_kde_data.append(fm.kde_data,ignore_index=True)  
tr_dt=new_training_data.append(fm.training_data,ignore_index=True)  
val_dt=new_test_data.append(fm.test_data,ignore_index=True)  


fm.models[mso].train(tr_dt,val_dt,kde_dt,fm.models[mso].source,fm.models[mso].objective,fm.cont_cond)
version='_test_II_Redo_100621'
file, folder = file_location(device,version)
