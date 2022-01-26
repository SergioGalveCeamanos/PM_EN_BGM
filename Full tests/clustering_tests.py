# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:27:00 2021

@author: sega01
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:27:12 2021

@author: sega01
"""
# https://stackabuse.com/multiple-linear-regression-with-python/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from elasticsearch import Elasticsearch
import matplotlib.pyplot as plt
import copy
import seaborn as sns
from matplotlib.colors import LogNorm
import itertools
from scipy import linalg
import matplotlib as mpl
from collections import OrderedDict
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift, estimate_bandwidth,OPTICS, cluster_optics_dbscan
from scipy import stats

cmaps = OrderedDict()
#########################################################################
def get_index_analytics(date,ty):
    index=ty+date[5:7]+date[0:4]
    return index 

def get_analytics(client,time_start, time_stop, device,version, names_analysis):
         ty='pm_data_'
         ind=get_index_analytics(time_start,ty) 
         response = client.search(
            index=ind,
            body={
                  "query": {
                    "bool": {
                      # Also: filter, must_not, should
                      "must": [ 
                        {
                          "match": {
                            "device": device
                          }
                        },
                        {
                          "match": {
                            "trained_version": version
                          }
                        },
                        {
                        "range": {
                        # Timestap format= "2019-12-30T09:25:20.000Z"
                        "timestamp": { 
                        "gt": time_start, # Date Format in Fault Manager: '2019-05-02 08:00:10'
                        "lt": time_stop 
                                     }
                                 }
                        }
                      ],
                      "must_not": [],
                      "should": []
                    }
                  },
                  "from": 0,
                  "size": 100,
                  "sort": [{ "timestamp" : {"order" : "asc"}}],
                  "aggs": {}
                },
            scroll='5m'
         )
         # DATA ARRANGE: each mso will have a dictionary with as many temporal series as in self.names_analysis --> all msos in the list data
         data=[]
         first=True
         for hit in response['hits']['hits']:
             #print(hit)
             if first:
                 first=False
                 n_msos=len(hit['_source'][names_analysis[0]])
                 for i in range(n_msos):
                     new_mso={}
                     for name in names_analysis:
                         field=hit['_source'][name]
                         if name=='timestamp':
                             new_mso[name]=[field]
                         else:
                             new_mso[name]=[field[i]]
                     data.append(new_mso)
             else: 
                 for i in range(n_msos):
                     for name in names_analysis:
                         field=hit['_source'][name]
                         if name=='timestamp':
                             data[i][name].append(field)
                         else:
                             data[i][name].append(field[i])

            
         sc_id=response['_scroll_id']
         more=True
         while more:
             sc=client.scroll(scroll_id=sc_id,scroll='2m') # ,scroll='1m'
             #sc_id=response['_scroll_id']
             if len(sc['hits']['hits'])==0: #|| total>20
                 more=False
             else:
                 for hit in sc['hits']['hits']:
                     if len(hit['_source'][names_analysis[0]])==n_msos:
                         for i in range(n_msos):
                             for name in names_analysis:
                                 field=hit['_source'][name]
                                 if name=='timestamp':
                                     data[i][name].append(field)
                                 else:
                                     data[i][name].append(field[i])
                     else:
                         print('  [!] WARNING: The gathered analysis data might come from different models, two sizes of MSO_SET: '+str(len(hit['_source'][names_analysis[0]]))+', '+str(n_msos)+'  | timestamp: '+hit['_source']['timestamp'])
                    
         return data

def get_telemetry(client,time_start, time_stop, device, var):
         ty='telemetry_'
         ind=get_index_analytics(time_start,ty) 
         response = client.search(
            index=ind,
            body={
                  "query": {
                    "bool": {
                      # Also: filter, must_not, should
                      "must": [ 
                        {
                          "match": {
                            "deviceId": device
                          }
                        },
                        {
                          "match": {
                            "aggregationSeconds": 1
                          }
                        },
                        {
                          "match": {
                            # name of the variable from sensors
                            "param": var 
                          }
                        },
                        {
                        "range": {
                        # Timestap format= "2019-12-30T09:25:20.000Z"
                        "timestamp": { 
                        "gt": time_start, # Date Format in Fault Manager: '2019-05-02 08:00:10'
                        "lt": time_stop 
                                     }
                                 }
                        }
                      ],
                      "must_not": [],
                      "should": []
                    }
                  },
                  "from": 0,
                  "size": 100,
                  "sort": [{ "timestamp" : {"order" : "asc"}}],
                  "aggs": {}
                },
            scroll='5m'
         )
         # DATA ARRANGE: each mso will have a dictionary with as many temporal series as in self.names_analysis --> all msos in the list data
         data={}
         for hit in response['hits']['hits']:
             #print(hit)
             data[hit['_source']['timestamp']]=hit['_source']['avg']
            
         sc_id=response['_scroll_id']
         more=True
         while more:
             sc=client.scroll(scroll_id=sc_id,scroll='2m') # ,scroll='1m'
             #sc_id=response['_scroll_id']
             if len(sc['hits']['hits'])==0: #|| total>20
                 more=False
             else:
                 for hit in sc['hits']['hits']:
                     data[hit['_source']['timestamp']]=hit['_source']['avg']
                     
         return data
#########################################################################

names_analysis=['models_error', 'low_bounds', 'high_bounds', 'activations', 'confidence','timestamp']
bounds=[ 'low_bounds', 'high_bounds']
device=71471
version='_test_6B_250121'

variables=['CondTempCirc1', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val','DscgTempCirc1', 'EbmpapstFan_1_Mng.ElectrInfo_EBM_1.CurrPower',
       'EvapTempCirc1', 'ExtTemp', 'FiltPress', 'InvInfoCirc1.Info_MotPwr','PumpPress', 'SubCoolCir1', 'SuctSH_Circ1', 'W_InTempUser',
       'W_OutTempEvap', 'W_OutTempUser', 'WaterFlowMeter']
traductor={'CondTempCirc1':'CondT', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val':'EEV','DscgTempCirc1':'DechT', 
                     'EbmpapstFan_1_Mng.ElectrInfo_EBM_1.CurrPower':'Ven','EvapTempCirc1':'EvapT', 'ExtTemp':'ExT', 
                     'FiltPress':'FP', 'InvInfoCirc1.Info_MotPwr':'Com','PumpPress':'PP', 'SubCoolCir1':'SbCoT', 
                     'SuctSH_Circ1':'SucTh', 'W_InTempUser':'Win','W_OutTempEvap':'Wev', 'W_OutTempUser':'Wout', 
                     'WaterFlowMeter':'Wfl'}
residuals=[0, 4, 8, 72, 100, 124, 135]
dates_issue=[["2020-10-07T12:00:00.000Z","2020-10-07T12:30:00.000Z"], # Good Behaviour_ random values
       ["2020-10-07T15:00:00.000Z","2020-10-07T15:30:00.000Z"], # Good Behaviour
       ["2020-10-07T22:00:00.000Z","2020-10-07T22:30:00.000Z"], # Good Behaviour
       ["2020-10-08T08:00:00.000Z","2020-10-08T08:30:00.000Z"], # Good Behaviour
       ["2020-10-08T09:00:00.000Z","2020-10-08T09:30:00.000Z"], # Good Behaviour
       ["2020-11-04T12:00:00.000Z","2020-11-04T12:30:00.000Z"], # Good Behaviour
       ["2020-11-04T16:00:00.000Z","2020-11-04T16:30:00.000Z"], # Good Behaviour
       ["2020-11-05T04:00:00.000Z","2020-11-05T04:30:00.000Z"], # Good Behaviour
       ["2020-11-05T05:00:00.000Z","2020-11-05T05:30:00.000Z"], # Good Behaviour
       ["2020-11-05T06:00:00.000Z","2020-11-05T06:30:00.000Z"]] # Good Behaviour
mso_variables={0:["PumpPress"],
               4:["InvInfoCirc1.Info_MotPwr","Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val","SuctSH_Circ1","SubCoolCir1","EvapTempCirc1","CondTempCirc1","W_OutTempUser","W_OutTempEvap"],
               8:['W_OutTempEvap', 'W_OutTempUser',"CondTempCirc1", 'DscgTempCirc1', 'InvInfoCirc1.Info_MotPwr', 'SubCoolCir1', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val'],
               72:['InvInfoCirc1.Info_MotPwr', 'DscgTempCirc1', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val', "CondTempCirc1", 'SubCoolCir1'],
               100:['InvInfoCirc1.Info_MotPwr', 'SuctSH_Circ1', 'EvapTempCirc1', 'SubCoolCir1', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val', "W_OutTempUser", 'WaterFlowMeter'],
               124:['InvInfoCirc1.Info_MotPwr', 'DscgTempCirc1', 'EvapTempCirc1', 'CondTempCirc1', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val',"W_OutTempUser" , 'WaterFlowMeter'],
               135:['InvInfoCirc1.Info_MotPwr', 'SuctSH_Circ1', 'DscgTempCirc1', 'CondTempCirc1', 'SubCoolCir1', 'W_OutTempEvap', 'W_OutTempUser', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val']}    

mso_outs={0:"FiltPress", 4:"W_InTempUser", 8:'W_InTempUser', 72:'EvapTempCirc1', 100:'W_InTempUser', 124:'W_OutTempEvap', 135:"W_InTempUser"}
mso_mean={0:-0.01337335448923473, 4:-0.16219452274855842, 8:-0.2008424606948894, 72:-0.43560362993315765, 100:0.06996989003482901, 124:-0.06893093812480769, 135:-0.10565125616369779}
mso_std={0:0.06467767829197925, 4:0.33390002512051153, 8:1.987155736223016, 72:2.9179740835802117, 100:0.48389134225103464, 124:0.47619842372436144, 135:0.4534009536255097}

#n_test=5000
#data_train=pd.read_csv("file_data.csv",index_col=0)
data_test=pd.read_csv("file_test.csv",index_col=0)
data_kde=pd.read_csv("file_kde.csv",index_col=0)

data_test=data_test.drop('timestamp',axis=1)
data_test=data_test.drop('UnitStatus',axis=1)
data_kde=data_kde.drop('timestamp',axis=1)
data_kde=data_kde.drop('UnitStatus',axis=1)

host='52.169.220.43:9200'
client=Elasticsearch(hosts=[host])
test_is={}
for v in variables:
    test_is[v]=[]
errors={}
for q in residuals:
    errors[q]=[]
"""for d in dates_issue:
    data_issue=get_analytics(client,d[0],d[1],device,version,names_analysis)
    if data_issue!=[]:
        for v in variables:
            baselines=get_telemetry(client,d[0],d[1],device,v)
            for t in data_issue[0]['timestamp']:
               test_is[v].append(baselines[t]) 
        for i in range(len(data_issue)):
            errors[residuals[i]]=errors[residuals[i]]+data_issue[i]['models_error']
data_test=pd.DataFrame(test_is)   """  
#########################################################################
"""copy_test=copy.deepcopy(data_test)
from sklearn.preprocessing import MinMaxScaler
for var in copy_test.columns:
    scaler = MinMaxScaler()
    telem_data=copy_test[var].values.reshape(-1,1)
    scaler.fit(telem_data)
    copy_test[var] = scaler.transform(telem_data)"""
copy_train=copy.deepcopy(data_kde)
from sklearn.preprocessing import MinMaxScaler
scalers=[]
for var in copy_train.columns:
    scaler = MinMaxScaler()
    telem_data=copy_train[var].values.reshape(-1,1)
    scaler.fit(telem_data)
    scalers.append(scaler)
    data_kde[var] = scaler.transform(telem_data)
    data_test[var] = scaler.transform(data_test[var].values.reshape(-1,1))
    #telem_data=copy_test[var].values.reshape(-1,1)
    #copy_test[var] = scaler.transform(telem_data)
#########################################################################
#cov=data_kde.cov()
fig = plt.figure(figsize=(15.0, 15.0))
#sns.heatmap(cov, annot=True, norm=LogNorm())
#########################################################################
def plot_results(X, Y_, means, covariances, index, title,color_iter,inds):

    splot = plt.subplot(2, 3, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        red_cov=covar[[inds[0],inds[1]]]
        v, w = linalg.eigh(red_cov[:,[inds[0],inds[1]]])
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 2.5, color=color, alpha=0.5,label='G'+str(i))

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse([mean[inds[0]],mean[inds[1]]], v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.legend()
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    
def plot_results_simple(X, Y_, means, index, title,color_iter,inds):

    splot = plt.subplot(2, 3, 1 + index)
    for i, (mean,  color) in enumerate(zip(means,  color_iter)):

        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 2.5, color=color, alpha=0.5,label='G'+str(i))

        # Plot an ellipse to show the Gaussian componen

        ell = mpl.patches.Ellipse([mean[inds[0]],mean[inds[1]]], 0.1, 0.1, 180. , color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.legend()
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

print('Start')
cont_cond=['ExtTemp', 'InvInfoCirc1.Info_MotPwr', 'W_OutTempUser', 'WaterFlowMeter']
to_plot=[cont_cond[0],cont_cond[1]]

# scores
def silhouette_score(estimator, X):
    try:
        clusters = estimator.predict(X)
        score = metrics.silhouette_score(X, clusters, metric='euclidean')
    except:
        score=-1
    return score
def ch_score(estimator, X):
    try:
        clusters = estimator.predict(X)
        score = metrics.calinski_harabasz_score(X, clusters)
    except:
        score=-1
    return score
def db_score(estimator, X):
    try:
        clusters = estimator.predict(X)
        score = metrics.davies_bouldin_score(X, clusters)
    except:
        score=-1
    return score


# baseline no outlayers
new_df=data_kde[cont_cond]
no_outl=new_df[(np.abs(stats.zscore(new_df)) < 3).all(axis=1)]


# baseline MeanShift
bnd = estimate_bandwidth(no_outl.values, quantile=0.2, n_samples=1500)
grid = GridSearchCV(MeanShift(),
                    {'bandwidth': [bnd*0.4,bnd*0.6,bnd*0.8,bnd,bnd*1.2,bnd*1.4,bnd*1.6],'bin_seeding':[True]},n_jobs=3,scoring=silhouette_score) # 20-fold cross-validation

grid.fit(no_outl.values,y=None)
meanSh = MeanShift(bandwidth=grid.best_params_['bandwidth'], bin_seeding=True,n_jobs=2,max_iter=1000).fit(no_outl.values)
groups_ms=meanSh.predict(no_outl.values)
clusters=len(np.unique(groups_ms))
print(np.unique(groups_ms, return_counts=True))
color_iter = itertools.cycle([plt.cm.Set1(i) for i in range(clusters)])
i=-1
fig = plt.figure(figsize=(20.0, 15.0))  
for ind in range(len(cont_cond)-1):
    for ind_two in range(ind+1,len(cont_cond)):
        i=i+1
        tits=[cont_cond[ind],cont_cond[ind_two]]
        title='Training Data viewed with: '+traductor[cont_cond[ind]]+' & '+traductor[cont_cond[ind_two]]
        plot_results_simple(no_outl[tits].values, groups_ms, meanSh.cluster_centers_, i,title,color_iter,[ind,ind_two])
fig.suptitle('Mean Shift')
plt.show() 
      
# baseline K means
grid = GridSearchCV(MiniBatchKMeans(),
                    {'n_clusters': [5,6,7,8,9],'batch_size':[100,300,500,750,1000],'max_iter':[1200]},n_jobs=3,scoring=silhouette_score) # 20-fold cross-validation

grid.fit(no_outl.values)

kmeans = MiniBatchKMeans(n_clusters=grid.best_params_['n_clusters'], batch_size=grid.best_params_['batch_size'],max_iter=5000).fit(no_outl.values)
groups_km=kmeans.predict(no_outl.values)
print(np.unique(groups_km, return_counts=True))
color_iter = itertools.cycle([plt.cm.Set1(i) for i in range(len(np.unique(groups_km)))])
i=-1
fig = plt.figure(figsize=(20.0, 15.0))  
for ind in range(len(cont_cond)-1):
    for ind_two in range(ind+1,len(cont_cond)):
        i=i+1
        tits=[cont_cond[ind],cont_cond[ind_two]]
        title='Training Data viewed with: '+traductor[cont_cond[ind]]+' & '+traductor[cont_cond[ind_two]]
        plot_results_simple(no_outl[tits].values, groups_km, kmeans.cluster_centers_, i,title,color_iter,[ind,ind_two])
fig.suptitle('K-Means')
plt.show() 

# BGM Method
grid = GridSearchCV(BayesianGaussianMixture(),
                    {'weight_concentration_prior': [0.00001,0.001,0.1,10,100,1000,2000,4000,6000,8000],'n_components':[5,6,7,8,15],'max_iter':[1200],'init_params':['random']},n_jobs=3,scoring=silhouette_score) # 20-fold cross-validation

grid.fit(no_outl.values)

bnd=grid.best_params_['weight_concentration_prior']
grid_2 = GridSearchCV(BayesianGaussianMixture(),
                    {'weight_concentration_prior': [bnd*0.4,bnd*0.6,bnd*0.8,bnd,bnd*1.2,bnd*1.4,bnd*1.6],'n_components':[grid.best_params_['n_components']],'max_iter':[1200],'init_params':['random']},n_jobs=3,scoring=silhouette_score) # 20-fold cross-validation

grid_2.fit(no_outl.values)

bgm = BayesianGaussianMixture(n_components=grid_2.best_params_['n_components'], random_state=42,max_iter=5000,weight_concentration_prior=grid_2.best_params_['weight_concentration_prior'],init_params='random',weight_concentration_prior_type='dirichlet_process').fit(no_outl.values)
groups=bgm.predict(no_outl.values)
probs=bgm.predict_proba(no_outl.values)
regions=np.unique(groups)
color_iter = itertools.cycle([plt.cm.Set1(i) for i in range(len(regions))])
i=-1
fig = plt.figure(figsize=(20.0, 15.0))  
plt.title('Variational Bayesian Estimation of a Gaussian Mixture')
for ind in range(len(cont_cond)-1):
    for ind_two in range(ind+1,len(cont_cond)):
        i=i+1
        tits=[cont_cond[ind],cont_cond[ind_two]]
        title='Training Data viewed with: '+traductor[cont_cond[ind]]+' & '+traductor[cont_cond[ind_two]]
        plot_results(no_outl[tits].values, groups, bgm.means_, bgm.covariances_, i,title,color_iter,[ind,ind_two])
fig.suptitle('Variational Bayesian Estimation of a Gaussian Mixture')
plt.show()    
#########################################################################

msh_sc=[silhouette_score(meanSh, no_outl.values),db_score(meanSh, no_outl.values),ch_score(meanSh, no_outl.values)/10000]
km_sc=[silhouette_score(kmeans, no_outl.values),db_score(kmeans, no_outl.values),ch_score(kmeans, no_outl.values)/10000]
bgm_sc=[silhouette_score(bgm, no_outl.values),db_score(bgm, no_outl.values),ch_score(bgm, no_outl.values)/10000]

# Plot bars with results of clustering metrics


labels = ['Mean Shift','K-Means','VBGMM']
silh = [msh_sc[0], km_sc[0], bgm_sc[0]]
davb = [msh_sc[1], km_sc[1], bgm_sc[1]]
caha = [msh_sc[2], km_sc[2], bgm_sc[2]]

scores_types=len(labels)
x = np.arange(scores_types)  # the label locations
width = 0.36  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width/scores_types, silh, width/scores_types, label='Silhouette')
rects2 = ax.bar(x - 0.5*width/scores_types, davb, width/scores_types, label='Davies-Bouldin')
rects3 = ax.bar(x + 0.5*width/scores_types, caha, width/scores_types, label='Calinski-Harabasz')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by Clustering and Score Methods')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
fig.tight_layout()

plt.show()


##################################################################################################

mso_variables={0:["PumpPress"],
               4:["InvInfoCirc1.Info_MotPwr","Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val","SuctSH_Circ1","SubCoolCir1","EvapTempCirc1","CondTempCirc1","W_InTempUser","W_OutTempEvap"],
               8:['W_OutTempEvap', 'W_OutTempUser','W_InTempUser', 'DscgTempCirc1', 'InvInfoCirc1.Info_MotPwr', 'SubCoolCir1', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val'],
               72:['InvInfoCirc1.Info_MotPwr', 'EvapTempCirc1', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val', "CondTempCirc1", 'SubCoolCir1'],
               100:['InvInfoCirc1.Info_MotPwr', 'W_InTempUser','SuctSH_Circ1' , 'SubCoolCir1', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val', "W_OutTempUser", 'WaterFlowMeter'],
               124:['InvInfoCirc1.Info_MotPwr', 'DscgTempCirc1', 'EvapTempCirc1', 'CondTempCirc1', 'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val',"W_OutTempUser" , 'WaterFlowMeter'],
               135:['InvInfoCirc1.Info_MotPwr', 'SuctSH_Circ1', 'DscgTempCirc1', 'CondTempCirc1', 'SubCoolCir1', 'W_OutTempEvap', 'W_OutTempUser', "W_InTempUser"]}    

mso_outs={0:"FiltPress", 4:"W_OutTempUser", 8:"CondTempCirc1", 72:'DscgTempCirc1', 100:'EvapTempCirc1', 124:'W_OutTempEvap', 135:'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val'}

residuals=[4,8,72,100]
if True:
    regions=np.unique(groups_km)
    clustering=kmeans
    models={}
    pred_weighted={}
    train_groups=clustering.predict(data_kde[cont_cond].values)
    test_groups=clustering.predict(data_test[cont_cond].values)
    #probs_test=clustering.predict_proba(data_test[cont_cond].values)
    train_data_groups={}
    test_data_groups={}
    covs={}
    for t in regions:
        locats=np.where(train_groups == t)[0]
        train_data_groups[t]=data_kde.iloc[locats]
        covs[t]=data_kde.iloc[locats].cov()
        locats=np.where(test_groups == t)[0]
        test_data_groups[t]=data_test.iloc[locats]
#regions=[1]
save_to_plot=test_data_groups[regions[0]].iloc[:500]
for mso in residuals:
    new_model_set={}
    for t in regions:
        
        new_model={}
        X_train = train_data_groups[t][mso_variables[mso]]
        y_train = train_data_groups[t][mso_outs[mso]]
        X_test = test_data_groups[t][mso_variables[mso]].iloc[:500]
        y_test = test_data_groups[t][mso_outs[mso]].iloc[:500]
        
        lin_reg_mod = ElasticNetCV(l1_ratio=[.1, .2, .4, .5, ],cv=30,n_jobs=3,max_iter=2000)
        
        lin_reg_mod.fit(X_train, y_train)
        if t in test_groups:
            pred = lin_reg_mod.predict(X_test)
            test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
            test_set_r2 = r2_score(y_test, pred)
        else:
            pred=[]
            test_set_rmse = 0
            test_set_r2 = 0
        
        new_model['model']=lin_reg_mod
        new_model['r2_score']=test_set_r2
        new_model['rmse_score']=test_set_rmse
        new_model['y_test']=y_test
        new_model['pred']=pred
        new_model['coeff']=lin_reg_mod.coef_
        new_model['offset']=lin_reg_mod.intercept_
        new_model_set[t]=new_model
    models[mso]=new_model_set



fig = plt.figure(figsize=(15.0, 20.0))
colors=['r','orange','y','g','purple']

check_df=[]
t=-1
for mso in residuals:
    t=t+1
    point=0
    for j in regions:
        x=np.arange(start=point, stop=point+len(models[mso][j]['y_test'].values), step=1)
        df=pd.DataFrame({'a':models[mso][j]['pred'],'b':models[mso][j]['y_test'].values})
        df=df.sort_values('b')
        check_df.append(df)
        ax1 = fig.add_subplot(4,1,t+1)
        if j==0:
            ax1.plot(x,df['b'].values,c='b',linewidth=2, alpha=0.5, label='Measured')
        else:
            ax1.plot(x,df['b'].values,c='b',linewidth=2, alpha=0.5)
        ax1.plot(x,df['a'].values,c=colors[j],linewidth=2, alpha=0.7, label='Prediction_Region_'+str(j))
        ax1.legend()
        point=point+len(models[mso][j]['y_test'].values)
        ax1.title.set_text('MSO #'+str(mso)+' | Target: '+traductor[mso_outs[mso]])
fig.suptitle("Resulting fit for test data") 
plt.show()

fig = plt.figure(figsize=(15.0, 20.0))
for mso in residuals:

    # Weights per region
    coefs={}
    for j in regions:
        coefs[j]=np.around(models[mso][j]['coeff'],decimals=2)
    labels=[]
    var_coef={}
    i=-1
    for n in mso_variables[mso]:
        i=i+1
        var_coef[n]=[]
        for j in regions:
            var_coef[n].append(coefs[j][i])
        labels.append(traductor[n])

    x = np.arange(len(labels))  # the label locations
    width = 0.8  # the width of the bars
    
    fig, ax = plt.subplots()
    rects={}
    N=len(regions)
    for j in regions:
        pl=-N/2+(N-j-1)
        rects[j] = ax.bar(x - pl*width/N, coefs[j], width/N, label='Region #'+str(j),color=colors[j])

    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Weights')
    ax.set_title('Weights by Region in MSO #'+str(mso)+' | Target: '+traductor[mso_outs[mso]])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    for j in regions:
        autolabel(rects[j])

    fig.tight_layout()
    plt.show()
"""
#easy plot for doc
fig = plt.figure(figsize=(20.0, 10.0))
t=-1
check_df=[]
for mso in residuals[1:]:
    t=t+1
    df=pd.DataFrame({'a':models[mso][regions[0]]['model'].predict(save_to_plot[mso_variables[mso]]),'b':save_to_plot[mso_outs[mso]].values})
    df=df.sort_values('b')
    check_df.append(df)
    ax1 = fig.add_subplot(2,3,t+1)
    ax1.plot(df['a'].values,c='r',linewidth=2, alpha=0.7, label='Prediction')
    ax1.plot(df['b'].values,c='b',linewidth=2, alpha=0.5, label='Measured')
    ax1.legend()
    ax1.title.set_text('MSO #'+str(mso)+' | Target: '+traductor[mso_outs[mso]])
fig.suptitle("Resulting fit for test data") 
diffs=[]
for d in range(1,len(check_df)):
    diffs.append(check_df[0].values-check_df[d].values)
fig = plt.figure(figsize=(20.0, 10.0))
t=-1
for mso in residuals[1:]:
    t=t+1
    ax1 = fig.add_subplot(2,3,t+1)
    weights=models[mso][regions[0]]['coeff']
    labels=[]
    for n in mso_variables[mso]:
        labels.append(traductor[n])
    ax1.bar(labels,weights)
    ax1.legend()
    ax1.title.set_text('MSO #'+str(mso)+' | Target: '+traductor[mso_outs[mso]])
fig.suptitle("Resulting weights") 

sts=copy_train[cont_cond].describe()
cent=kmeans.cluster_centers_[1]
i=-1
for m in residuals:
    print('MSO #'+str(m))
    print(models[m][regions[0]]['r2_score'])
    print(models[m][regions[0]]['rmse_score'])"""


#############################################################
# old plots to show comparisons in BGM mixtures
if False:   
    # test how it would perform the weighted mean
    order_weight_plot={}
    for mso in residuals:
        new_mso_weighted={}
        order_weight_plot[mso]={}
        for t in regions:
            new_model={}
            X_test = data_test[mso_variables[mso]]
            y_test = data_test[mso_outs[mso]]
            
    
            pred = models[mso][t]['model'].predict(X_test)
    
            new_mso_weighted[t]=pred
        pred_weighted[mso]=pd.DataFrame(new_mso_weighted).values*probs_test
        
    for t in regions:
        locats=np.where(test_groups == t)[0]
        for mso in residuals:
            train_data_groups[t]=data_kde.iloc[locats]  
            order_weight_plot[mso][t]=pred_weighted[mso][locats].sum(axis=1)


    root=[]
    #colors = iter([plt.cm.tab20(i) for i in range(20)])
    #root=r'V:\PL\Projects\Shared\LAUDA Cloud\LUC - Industrial PhD\Follow Up\Meetings\Figures Evaluation of FNN and ConfInt\regression'
    root=r'V:\PL\Projects\Shared\LAUDA Cloud\LUC - Industrial PhD\Follow Up\Meetings\Figures Evaluation of FNN and ConfInt\regression_BGM_3'
    #root=[]
    for mso in range(len(residuals)):
        if mso>0:
            plot_n=mso+1
            fig = plt.figure(figsize=(20.0, 10.0))
            colors = iter([plt.cm.tab20(i) for i in range(20)])
            #ax1 = fig.add_subplot(2,2,1)
            point=0
            for t in regions:
                if t in test_groups:
                    x=np.arange(start=point, stop=point+len(models[residuals[mso]][t]['y_test'].values), step=1)
                    plt.plot(x,models[residuals[mso]][t]['y_test'].values,c='k',linewidth=0.8, label='Prediction')
                    plt.plot(x,models[residuals[mso]][t]['pred'],c=next(colors),linewidth=0.6, label=traductor[mso_outs[residuals[mso]]]+"_RegL_G"+str(t))
                    #plt.plot(x,order_weight_plot[residuals[mso]][t],c='crimson',linewidth=0.6, label=traductor[mso_outs[residuals[mso]]]+"_RegL_Weighted"+str(t))
                    point=point+len(models[residuals[mso]][t]['y_test'].values)
                    #ax1.plot(np.array(errors[residuals[mso]])*mso_std[residuals[mso]]+mso_mean[residuals[mso]]+models[residuals[mso]]['y_test'],c='r',linewidth=1.2, label=mso_outs[residuals[mso]]+"_FNN")
                    #ax1.xlabel('Samples')
                    #ax1.ylabel('Errors')
                    #ax1.title('MSO #'+str(residuals[mso])+' Errors')

            
            #ax1.legend()
            plt.title('Model Performance')

            plt.show()
            if root!=[] and False:
                file=r"\comparison_MSO_"+str(residuals[mso])+".png"
                fig.savefig(root+file)
        """else:
            n=mso_variables[residuals[mso]][0]
            fig = plt.figure(figsize=(15.0, 10.0))
            plt.plot(models[residuals[mso]]['y_test'].values,c='g',linewidth=0.8, label='Measured')
            plt.plot(models[residuals[mso]]['pred'],c='b',linewidth=1, label=mso_outs[residuals[mso]]+"_RegL")
            plt.plot(np.array(errors[residuals[mso]])*mso_std[residuals[mso]]+mso_mean[residuals[mso]]+models[residuals[mso]]['y_test'],c='r',linewidth=1, label=mso_outs[residuals[mso]]+"_FNN")
            plt.plot(data_test[n].values,linewidth=0.4, label=n)
            plt.legend()
            plt.title("Results for MSO : "+str(residuals[mso]))
            plt.show()"""
            
    for mso in range(1,len(residuals)):
        fig = plt.figure(figsize=(20.0, 15.0)) 
        for t in regions:
            ax1 = fig.add_subplot(2,4,t+1)
            weights=models[residuals[mso]][t]['coeff']
            sek=np.argsort(abs(weights))
            names=[mso_outs[residuals[mso]],mso_variables[residuals[mso]][sek[-1]],mso_variables[residuals[mso]][sek[-2]]]

            labels=[]
            for n in mso_variables[residuals[mso]]:
                labels.append(traductor[n])
            ax1.bar(labels,weights)
            #ax1.tick_params(labelrotation=45)
            tit='Len Reg Coeff in MSO#'+str(residuals[mso])+' | Group: '+str(t)
            ax1.title.set_text(tit)
        fig.suptitle("Analysis for MSO: "+str(residuals[mso])+' | Goal: '+mso_outs[residuals[mso]])
        plt.show()
        """for j in range(len(names)):
            ax1 = fig.add_subplot(2,2,j+2)
            labels=[]
            for n in cov[names[j]].index:
                labels.append(traductor[n])
            ax1.bar(labels,cov[names[j]].values)
            #ax1.tick_params(labelrotation=45)
            ax1.title.set_text('Covariance for '+names[j])
        
        fig.suptitle("Analysis for MSO: "+str(residuals[mso]))
        #fig.tight_layout(pad=2.0)
        plt.show()
        #if root!=[]:
                #file=r"\analysis_MSO_"+str(residuals[mso])+".png"
                #fig.savefig(root+file)
    
    fig = plt.figure(figsize=(20.0, 15.0))             
    for mso in range(1,len(residuals)):
        ax1 = fig.add_subplot(2,3,mso)
        err_reg=models[residuals[mso]]['y_test'].values-models[residuals[mso]]['pred']
        #err_fnn=-(np.array(errors[residuals[mso]])*mso_std[residuals[mso]]+mso_mean[residuals[mso]])
        sup=max([max(err_reg)])
        inf=min([min(err_reg)])
        ax1.hist(err_reg,bins=100,range=[inf, sup], alpha=0.5, label='Prediction Linear Regresion')
        #ax1.hist(err_fnn,bins=100,range=[inf, sup], alpha=0.5, label='Prediction FNN')
        ax1.title.set_text('Error distribution for MSO#'+str(residuals[mso]))
        ax1.legend()
    plt.show()"""
    #if root!=[]:
        #file=r"\Errors Distributions.png"
        #fig.savefig(root+file)"""
   
