# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:11:02 2021

@author: sega01
"""

from elasticsearch import Elasticsearch
import pandas as pd
import json
import numpy as np
from matplotlib import cm as CM
import matplotlib.pyplot as plt
import datetime
from matplotlib.lines import Line2D
from sklearn.cluster import MiniBatchKMeans

def get_index_analytics(date,ty):
    index=ty+date[5:7]+date[0:4]
    return index 

#client,d[0],d[1],device,mso_outs[residuals[m]]
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
##############################################################################
# get clusters
"""clusters={}
for m in fm.mso_set:
    clusters[m]=fm.models[m].bgm.cluster_centers_
 
stats={}
for m in fm.mso_set:
    stats[m]={'mean':{},'std':{}}
    for c in fm.cont_cond:
        stats[m]['mean'][c]=fm.models[m].train_stats.loc[c,'mean']
        stats[m]['std'][c]=fm.models[m].train_stats.loc[c,'std']
"""

        
##############################################################################
def norm(x,train_stats):
    y={}
    for name in x.columns:
        y[name]=[]
        for i in x.index:
            a=float((x.loc[i,name]-train_stats['mean'][name])/train_stats['std'][name])
            y[name].append(a)  #.apply(lambda num: (num-train_stats.loc[name,'mean'])/train_stats.loc[name,'std'])
    return pd.DataFrame(y)
host='137.116.224.197:9200'
client=Elasticsearch(hosts=[host])
device=74124
d=["2021-05-27T20:30:00.000Z","2021-05-29T23:00:00.000Z"]
residuals=[10, 16, 30, 41, 52, 86]  
variables=["WaterFlowMeter","ExtTemp","W_InTempUser","ControlRegCompAC.VarFrequencyHzMSK"]
versions=['_test_VI_StabilityFilt_090621','_test_I_Redo_100621','_test_II_Redo_100621']
cluster_centers={}
cluster_centers['_test_VI_StabilityFilt_090621']={10: np.array([[ 0.30684264,  0.94088268,  0.6670714 ,  1.1277444 ],
       [-1.03673857, -0.51033848, -0.91576194, -0.18966042],
       [ 0.61917753, -0.7428548 , -1.06384103,  0.3267061 ],
       [-1.06903427,  0.16733252,  0.77898846, -0.49355676],
       [ 0.77155083,  0.1345264 ,  0.39396185, -0.71745328]]), 
       16: np.array([[-1.19285638,  0.17538972,  0.95502663, -0.39778545],
       [ 0.71427986,  0.08354405,  0.54068062, -0.74592587],
       [ 0.75128661, -0.22186031, -1.10664764,  0.78511581],
       [-0.9345732 , -0.11864735, -0.822042  , -0.81076784],
       [ 0.68936904,  0.99403141,  0.99243065,  1.06663444],
       [-0.71987172,  0.39102057, -0.3360628 ,  0.99080884],
       [ 0.16802399, -1.61709003, -0.94000941, -0.58466705]]), 
       30: np.array([[-0.87983913,  0.86140885,  0.63242688,  0.93039714],
       [ 0.61498497, -0.5395575 , -0.16549317, -0.93243954],
       [-1.14706591, -0.11666065,  0.14279949, -0.75103242],
       [ 0.19946653, -0.34805445, -1.03923588,  0.8125772 ],
       [ 0.83240262,  0.70746069,  0.8708061 ,  0.49792251]]), 
       41: np.array([[-0.87983913,  0.86140885,  0.63242688,  0.93039714],
       [ 0.61498497, -0.5395575 , -0.16549317, -0.93243954],
       [-1.14706591, -0.11666065,  0.14279949, -0.75103242],
       [ 0.19946653, -0.34805445, -1.03923588,  0.8125772 ],
       [ 0.83240262,  0.70746069,  0.8708061 ,  0.49792251]]), 
       52: np.array([[ 0.39518255, -0.16529471, -1.03740197,  1.09686544],
       [-1.06383448,  0.45786125,  0.81969242,  0.17205161],
       [ 0.18244401, -0.98032416, -0.76592492, -0.72115088],
       [ 0.70228644,  0.94352577,  0.93279078,  1.18033302],
       [ 0.74073047,  0.11470362,  0.69395466, -0.63949105],
       [-1.20311504,  0.03069439, -0.44360074, -0.64836609]]), 
       86: np.array([[-1.04256775, -0.37934038, -0.87009815, -0.12049621],
       [ 0.7171508 ,  0.10514389,  0.36253924, -0.82126797],
       [ 0.59630944, -0.7796793 , -1.06338986,  0.28350835],
       [ 0.38266062,  0.88174331,  0.72408821,  1.03029934],
       [-1.14909417,  0.16087902,  0.83897292, -0.50494744]])}

cluster_centers['_test_I_Redo_100621']={10: np.array([[ 0.30684264,  0.94088268,  0.6670714 ,  1.1277444 ],
       [-1.03673857, -0.51033848, -0.91576194, -0.18966042],
       [ 0.61917753, -0.7428548 , -1.06384103,  0.3267061 ],
       [-1.06903427,  0.16733252,  0.77898846, -0.49355676],
       [ 0.77155083,  0.1345264 ,  0.39396185, -0.71745328]]), 
       16: np.array([[-1.19285638,  0.17538972,  0.95502663, -0.39778545],
       [ 0.71427986,  0.08354405,  0.54068062, -0.74592587],
       [ 0.75128661, -0.22186031, -1.10664764,  0.78511581],
       [-0.9345732 , -0.11864735, -0.822042  , -0.81076784],
       [ 0.68936904,  0.99403141,  0.99243065,  1.06663444],
       [-0.71987172,  0.39102057, -0.3360628 ,  0.99080884],
       [ 0.16802399, -1.61709003, -0.94000941, -0.58466705]]), 
       30: np.array([[-0.87983913,  0.86140885,  0.63242688,  0.93039714],
       [ 0.61498497, -0.5395575 , -0.16549317, -0.93243954],
       [-1.14706591, -0.11666065,  0.14279949, -0.75103242],
       [ 0.19946653, -0.34805445, -1.03923588,  0.8125772 ],
       [ 0.83240262,  0.70746069,  0.8708061 ,  0.49792251]]), 
       41: np.array([[-0.87983913,  0.86140885,  0.63242688,  0.93039714],
       [ 0.61498497, -0.5395575 , -0.16549317, -0.93243954],
       [-1.14706591, -0.11666065,  0.14279949, -0.75103242],
       [ 0.19946653, -0.34805445, -1.03923588,  0.8125772 ],
       [ 0.83240262,  0.70746069,  0.8708061 ,  0.49792251]]), 
       52: np.array([[ 0.39518255, -0.16529471, -1.03740197,  1.09686544],
       [-1.06383448,  0.45786125,  0.81969242,  0.17205161],
       [ 0.18244401, -0.98032416, -0.76592492, -0.72115088],
       [ 0.70228644,  0.94352577,  0.93279078,  1.18033302],
       [ 0.74073047,  0.11470362,  0.69395466, -0.63949105],
       [-1.20311504,  0.03069439, -0.44360074, -0.64836609]]), 
       86: np.array([[ 0.56853384,  0.12198583, -0.76844678,  1.09272166],
       [-0.48389024,  0.83311912,  0.66648255,  1.03298276],
       [ 0.77808876,  0.20003061,  0.64520406, -0.59120745],
       [-1.15784511,  0.060045  ,  0.47130719, -0.66478158],
       [-0.06955607, -1.05189166, -1.06992028, -0.51048404]])}

cluster_centers['_test_II_Redo_100621']={10: np.array([[ 0.30684264,  0.94088268,  0.6670714 ,  1.1277444 ],
       [-1.03673857, -0.51033848, -0.91576194, -0.18966042],
       [ 0.61917753, -0.7428548 , -1.06384103,  0.3267061 ],
       [-1.06903427,  0.16733252,  0.77898846, -0.49355676],
       [ 0.77155083,  0.1345264 ,  0.39396185, -0.71745328]]), 
       16: np.array([[-1.19285638,  0.17538972,  0.95502663, -0.39778545],
       [ 0.71427986,  0.08354405,  0.54068062, -0.74592587],
       [ 0.75128661, -0.22186031, -1.10664764,  0.78511581],
       [-0.9345732 , -0.11864735, -0.822042  , -0.81076784],
       [ 0.68936904,  0.99403141,  0.99243065,  1.06663444],
       [-0.71987172,  0.39102057, -0.3360628 ,  0.99080884],
       [ 0.16802399, -1.61709003, -0.94000941, -0.58466705]]), 
       30: np.array([[-0.87983913,  0.86140885,  0.63242688,  0.93039714],
       [ 0.61498497, -0.5395575 , -0.16549317, -0.93243954],
       [-1.14706591, -0.11666065,  0.14279949, -0.75103242],
       [ 0.19946653, -0.34805445, -1.03923588,  0.8125772 ],
       [ 0.83240262,  0.70746069,  0.8708061 ,  0.49792251]]), 
       41: np.array([[-0.87983913,  0.86140885,  0.63242688,  0.93039714],
       [ 0.61498497, -0.5395575 , -0.16549317, -0.93243954],
       [-1.14706591, -0.11666065,  0.14279949, -0.75103242],
       [ 0.19946653, -0.34805445, -1.03923588,  0.8125772 ],
       [ 0.83240262,  0.70746069,  0.8708061 ,  0.49792251]]), 
       52: np.array([[ 0.39518255, -0.16529471, -1.03740197,  1.09686544],
       [-1.06383448,  0.45786125,  0.81969242,  0.17205161],
       [ 0.18244401, -0.98032416, -0.76592492, -0.72115088],
       [ 0.70228644,  0.94352577,  0.93279078,  1.18033302],
       [ 0.74073047,  0.11470362,  0.69395466, -0.63949105],
       [-1.20311504,  0.03069439, -0.44360074, -0.64836609]]), 
       86: np.array([[-1.04256775, -0.37934038, -0.87009815, -0.12049621],
       [ 0.7171508 ,  0.10514389,  0.36253924, -0.82126797],
       [ 0.59630944, -0.7796793 , -1.06338986,  0.28350835],
       [ 0.38266062,  0.88174331,  0.72408821,  1.03029934],
       [-1.14909417,  0.16087902,  0.83897292, -0.50494744]])}

training_stats={}
training_stats['_test_VI_StabilityFilt_090621']={10: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 16: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 30: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 41: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 52: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 86: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}}
training_stats['_test_I_Redo_100621']={10: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 16: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 30: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 41: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 52: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 86: {'mean': {'WaterFlowMeter': 47.083149995454434, 'ExtTemp': 22.349501227303158, 'W_InTempUser': 15.886939345603608, 'ControlRegCompAC.VarFrequencyHzMSK': 108.78530294139524}, 'std': {'WaterFlowMeter': 12.808062998991348, 'ExtTemp': 1.1874285097051451, 'W_InTempUser': 14.977173285123403, 'ControlRegCompAC.VarFrequencyHzMSK': 46.541000059661854}}}
training_stats['_test_II_Redo_100621']={10: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 16: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 30: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 41: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 52: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}, 86: {'mean': {'WaterFlowMeter': 46.84279402540024, 'ExtTemp': 22.414790134455767, 'W_InTempUser': 15.017071064282405, 'ControlRegCompAC.VarFrequencyHzMSK': 108.55561959654176}, 'std': {'WaterFlowMeter': 13.009840400662714, 'ExtTemp': 1.2046713259650426, 'W_InTempUser': 14.965715439944347, 'ControlRegCompAC.VarFrequencyHzMSK': 46.8723239345942}}}

telemetry={}
for var in variables:
    telemetry[var]=get_telemetry(client,d[0],d[1],device,var)
df=pd.DataFrame(telemetry)
normed_cc=norm(df, training_stats['_test_VI_StabilityFilt_090621'][10])
#df.to_csv('telemetry_cont_cond_hot.csv')
clusts={}
v='_test_II_Redo_100621'
versions=['_test_II_Redo_100621']
groups=[]
for m in residuals:
    clusts[m]=MiniBatchKMeans(n_clusters=len(cluster_centers[v][m]))
    clusts[m].cluster_centers_=cluster_centers[v][m]
    groups.append(clusts[m].predict(normed_cc.values))
    
""" 
for v in versions:
    colors=['crimson','gold','orchid','limegreen','royalblue','chocolate','slategray','purple']
    fig = plt.figure(figsize=(30.0, 20.0))
    #color=CM.rainbow(np.linspace(0,1,(len(faults))))
    for mso in range(len(residuals)):
        c=colors[mso]
        ax1 = fig.add_subplot(3,2,mso+1)
        bars=[]
        for f in range(cluster_centers[v][residuals[mso]].shape[0]):
            dists=(normed_cc-cluster_centers[v][residuals[mso]][f,:])**2
            dists=np.sqrt(dists.sum(axis=1))
            ax1.plot(dists.values,color=colors[f],linewidth=1.2,alpha=0.7,label='Dist. Cluster #'+str(f))
        ax1.title.set_text('MSO #'+str(residuals[mso]))
        ax1.legend()
    fig.suptitle("Distances for each MSO in vers "+v+' | High Temp data')
    #fig.savefig('HOT_distances_by_Cluster_'+v+'.png')
    plt.show()"""
    
for v in versions:
    colors=['crimson','gold','orchid','limegreen','royalblue','chocolate','slategray','purple']
    fig = plt.figure(figsize=(30.0, 20.0))
    #color=CM.rainbow(np.linspace(0,1,(len(faults))))
    for mso in range(len(residuals)):
        c=colors[mso]
        ax1 = fig.add_subplot(3,2,mso+1)
        bars=[]
        by_var=[]
        for f in range(cluster_centers[v][residuals[mso]].shape[0]):
            dists=(normed_cc-cluster_centers[v][residuals[mso]][f,:])
            #dists=np.sqrt(dists.sum(axis=1))
            if len(by_var)==0:
                by_var=dists
            else:
                by_var=by_var+dists
        i=-1
        for f in variables: 
            i=i+1
            ax1.plot(by_var[f].values/cluster_centers[v][residuals[mso]].shape[0],color=colors[i],linewidth=1.2,alpha=0.7,label='Dist. '+f)
        ax1.title.set_text('MSO #'+str(residuals[mso]))
        ax1.legend()
    fig.suptitle("Distances for each MSO in vers "+v+' | High Temp data')
    #fig.savefig('HOT_distances_by_Var_'+v+'.png')
    plt.show()