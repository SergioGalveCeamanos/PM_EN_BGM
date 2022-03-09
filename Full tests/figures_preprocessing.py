# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:34:31 2022

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
traductor={'CondTempCirc1':'CondT', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val':'EEV','DscgTempCirc1':'DechT', 
                     'EbmpapstFan_1_Mng.ElectrInfo_EBM_1.CurrPower':'Ven','EvapTempCirc1':'EvapT', 'ExtTemp':'ExT', 
                     'FiltPress':'FP', 'InvInfoCirc1.Info_MotPwr':'Com','PumpPress':'PP', 'SubCoolCir1':'SbCoT', 
                     'SuctSH_Circ1':'SucTh', 'W_InTempUser':'Win','W_OutTempEvap':'Wev', 'W_OutTempUser':'Wout', 
                     'WaterFlowMeter':'Wfl','ControlRegCompAC.VarFrequencyHzMSK':'Com','EbmpapstFan_1_Mng.InfoSpeed_EBM_1.CurrSpeed':'Fan'}

files_txt=True
not_prepared=False
file_norm_train='Normed_data_train_filtered.csv'#♠'Normed_data_train_filtered.csv' , 'Normed_data_initial_train_filtered.csv'
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
    file_norm_train='Normed_data_initial_train_filtered.csv' #'Normed_data_train.csv','Normed_data_train_filtered.csv'
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
    


def plot_continuous_heatmap(df,title):
    sides=df.shape[1]
    fields=df.columns
    t=-1
    fig = plt.figure(figsize=(20.0, 15.0))
    mins=0
    maxs=[]
    for i in range(sides-1):
        for j in range(i+1,sides):
            x=df[fields[i]].values
            y=df[fields[j]].values
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=300)
            maxs.append(np.max(heatmap))
            #print(j)
    for i in range(sides-1):
        for j in range(i+1,sides):
            t=t+1
            splot = plt.subplot(2, 5, 1 + t)
            x=df[fields[i]].values
            y=df[fields[j]].values
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=300)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            #splot.clf()
            #plt.clf()
            im_scale=splot.imshow(heatmap.T, extent=extent,norm=LogNorm(vmin=0.1, vmax=max(maxs)),interpolation='gaussian', origin='lower',cmap='cool', aspect="auto") #, vmin=mins, vmax=
            splot.set_title(traductor[fields[i]]+' vs '+traductor[fields[j]])
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im_scale, cax=cbar_ax)
    fig.suptitle(title.format(int(np.round(df.shape[0]/1000))))
    #plt.tight_layout()
    plt.show()
    
# this class will return a subset of size s where the samples are evenly distributed to include maximum diversity 
def homo_sampling(data,cont_cond,print_file,s=50000,uncertainty=[]):
    stats=data.describe()
    if uncertainty==[]:
        for i in cont_cond:
            a=(data[i].max()-data[i].min())/(70*stats.loc['std'][i])
            uncertainty.append(a)
    lat_var=[]
    for i in data.columns:
        if i not in cont_cond:
            lat_var.append(i)
    done=False
    ws=copy.deepcopy(data)
    sets=[]
    first=True
    pd_sets=[]
    while not done:
        #ind=list(ws.index)
        it_done=False
        sets=[]
        while not it_done:
            i=np.random.randint(0,high=ws.shape[0])
            #samp=ind.pop(i)
            q=ws.iloc[i]
            filt_bool=np.array([True]*ws.shape[0])
            t_a=datetime.datetime.now()
            for cc in range(len(cont_cond)):
                l=q[cont_cond[cc]]-uncertainty[cc]
                h=q[cont_cond[cc]]+uncertainty[cc]
                filt_bool = filt_bool & np.array(ws[cont_cond[cc]]>l) & np.array(ws[cont_cond[cc]]<h)
            t_b=datetime.datetime.now()
            dif=t_b-t_a
            #fault_manager.Load(folder,file)
            #print('  [T] Time for filtering one sample out ---> '+str(dif))
            # with filt_bool we have extracted 
            filt_ws=ws[filt_bool]
            #ind=list(set(ind) - set(filt_ws.index))
            ws=ws.drop(filt_ws.index)
            sets.append(filt_ws)
            if ws.shape[0]==0:
                it_done=True
                file_printed = open(print_file, "a")
                print('[!] Sample set found from ws with size='+str(len(sets)),file=file_printed)
                file_printed.close()
        #once the list has been sorted in sets, we check if there are too many or too few sets        
        if len(pd_sets)>s or first:
            # we will do the previous process again using as ws the average of the samples 
            new_ws={'set_behind':[]}
            new_sets=[]
            for i in cont_cond:
                new_ws[i]=[]
            # after new_ws is initialize we fill it with the means of the new sets created
            for i in range(len(sets)):
                new_ws['set_behind'].append(i)
                st=sets[i]
                for i in cont_cond:
                    new_ws[i].append(st[i].mean())
            ws=pd.DataFrame(new_ws)
            file_printed = open(print_file, "a")
            print('[!] New working set:',file=file_printed)
            print(ws,file=file_printed)
            file_printed.close()
            # rearrange the pd_sets so that the new set_behind points out to the combined subsets from previous iteration
            if not first:
                new_sets=[]
                for i in range(len(sets)):
                    temp_set=[]
                    ff=True
                    for j in range(sets[i].shape[0]):
                        to_add=int(sets[i].iloc[j]['set_behind'])
                        if ff:
                            temp_set=pd_sets[to_add]
                            ff=False
                        else:
                            temp_set=temp_set.append(pd_sets[to_add],ignore_index=True) 
                    new_sets.append(temp_set)
            else:
                new_sets=sets
                first=False
            for x in range(len(uncertainty)):
                uncertainty[x]=uncertainty[x]*1.25
            pd_sets=copy.deepcopy(new_sets)
            file_printed = open(print_file, "a")
            print('[!] End new set arrangement --> pd_sets size: '+str(len(pd_sets)),file=file_printed)
            file_printed.close()
        # Once the number of sets is smaller than the required samples, we stop the agregation and take on sample from each set. To fill in the remaining samples we get new samples from each set according to their size                                
        if len(pd_sets)<s:
            # we will take one sample from each and the remaining to fill up to s will be extracted according to the cumm count of samples among sets
            file_printed = open(print_file, "a")
            print('[R] Inside low set cond final part of process!',file=file_printed)
            file_printed.close()
            initial=True
            missing=s-len(pd_sets)
            dic_sets={'index':[],'size':[]}
            tot=0
            for i in range(len(pd_sets)):
                dic_sets['index'].append(i)
                tot=pd_sets[i].shape[0]
                dic_sets['size'].append(tot)
            # we arrange it in a DF to sort and obtain the cummulative function
            df_sizes=pd.DataFrame(dic_sets)
            df_sizes=df_sizes.sort_values(by=['size'],ascending=False)
            df_sizes['size']=df_sizes['size'].cumsum(axis = 0)
            hits=np.round(np.linspace(0,df_sizes['size'].max(),num=missing))
            loc=0
            new_samples=[]
            # we go though the evenly spaced numbers and collect from which
            for i in hits:
                not_yet=True
                #print(' [o] New selected size and hit: '+str(df_sizes.iloc[loc]['size'])+' | '+str(i))
                while not_yet and loc<df_sizes.shape[0]:
                    if df_sizes.iloc[loc]['size']>i:
                        new_samples.append(df_sizes.iloc[loc]['index'])
                        not_yet=False
                    else:
                        loc=loc+1
            # now we count how many elements are there from each pd_sets
            start_count=True
            for i in new_samples:
                if start_count:
                    start_count=False
                    count_hits = {i:new_samples.count(i)}
                else:
                    count_hits[i] = new_samples.count(i)
            # just load in a new var one sample from each set plus the ones listed in count_hits
            for i in range(len(pd_sets)):
                if i in count_hits:
                    si=count_hits[i]+1
                    randoms=list(np.random.randint(low=0, high=pd_sets[i].shape[0], size=(si,)))
                    add=pd_sets[i].iloc[randoms]
                else:
                    randoms=list(np.random.randint(low=0, high=pd_sets[i].shape[0], size=(1,)))
                    add=pd_sets[i].iloc[randoms]  
                if initial:
                    final_samp=add
                    initial=False
                else:
                    final_samp=final_samp.append(add,ignore_index=True)
                    
            done=True
            file_printed = open(print_file, "a")
            print(' [R] --> Final subset obtained:',file=file_printed)
            print(final_samp,file=file_printed)
            file_printed.close()
        first=False
    # we return the    
    return final_samp

file_norm_train='Normed_data_train_filtered.csv'#'Normed_data_train_filtered.csv' , 'Normed_data_initial_train_filtered.csv'
normed_train_data_o=pd.read_csv(file_norm_train,index_col=0)
#filter to only use contour conditions:
cont_cond=['ControlRegCompAC.VarFrequencyHzMSK','WaterFlowMeter','W_OutTempUser','W_InTempUser','ExtTemp']
if cont_cond!=[]:
    training_set=copy.deepcopy(normed_train_data_o[cont_cond])
else:
    training_set=copy.deepcopy(normed_train_data_o)
std_train_pool=training_set.describe().transpose().loc[:,'std'].values
print_file='sampling_Report_Homog25k_160222.txt'#'sampling_test_CC_vhard_270122.txt', 'sampling_test_v1_260122.txt','sampling_test_CC_260122.txt'
sample_file='selection_Report_Homog_160222.csv'#'selection_Training_Data_CC_vhard_270122.csv','selection_Training_Data_260122.csv','selection_Training_Data_CC_260122.csv'
selected_samples_file='subsampled_Homog25k_FullTrDt.csv' 
sizes_to_test=[28000,30000,32000,34000,36000]#[6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000]
results={}
for si in sizes_to_test:
    results[si]=homo_sampling(normed_train_data_o,cont_cond,print_file,s=si,uncertainty=[])
#subsampled.to_csv(selected_samples_file)

"""file_names=['Normed_data_initial_train_filtered.csv','subsampled_Homog15k_initialTrDt.csv','subsampled_GaussIter_initialTrDt_2.csv','Normed_data_train_filtered.csv','subsampled_Homog25k_FullTrDt.csv','subsampled_GaussIter_FullTrDt.csv']
titles=['Controlled Test Training Data (CTTD) - {}k samples','Homogeneous Sampling CTTD - {}k samples','Gaussian Sampling CTTD - {}k samples','Augmented Training Data (ATD) - {}k samples','Homogeneous Sampling ATD - {}k samples','Gaussian Sampling ATD - {}k samples']

for i in range(len(file_names)):
    df=pd.read_csv(file_names[i],index_col=0)[cont_cond]#.sample(15000)
    print(df.shape[0])
    plot_continuous_heatmap(df,titles[i])"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
target='EvapTempCirc1' #'DscgTempCirc1'
repeats=10
mean_res={}

for si in sizes_to_test: 
    for i in range(repeats):
        X_data=copy.deepcopy(results[si])
        y_data=X_data.pop(target)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
        alpha= 0.1
        model_enet = ElasticNet(alpha=alpha, l1_ratio=0.3)
        model_enet.fit(X_train, y_train)
        pred = model_enet.predict(X_test)
        res=(y_test-pred).abs()
        if i==0:
            mean_res[si]=res.describe()
        else:
            mean_res[si]=res.describe()+mean_res[si]
        print(si)
        print("   - r^2 of EN on test data : %f" % r2_score(y_test, pred))
    mean_res[si]=mean_res[si]/repeats 


    
